from metaflow.decorators import StepDecorator
from metaflow import current
import functools
import os
import threading

from .vllm_manager import VLLMManager
from .status_card import VLLMStatusCard, CardDecoratorInjector



class VLLMDecorator(StepDecorator, CardDecoratorInjector):
    """
    This decorator is used to run vllm APIs as Metaflow task sidecars.

    User code call
    --------------
    @vllm(
        model="...",
        ...
    )

    Valid backend options
    ---------------------
    - 'local': Run as a separate process on the local task machine.
    
    Valid model options
    -------------------
    Any HuggingFace model identifier, e.g. 'meta-llama/Llama-3.2-1B'

    NOTE: vLLM's OpenAI-compatible server serves ONE model per server instance.
    If you need multiple models, you must create multiple @vllm decorators.

    Parameters
    ----------
    model: str
        HuggingFace model identifier to be served by vLLM.
    backend: str
        Determines where and how to run the vLLM process.
    debug: bool
        Whether to turn on verbose debugging logs.
    """

    name = "vllm"
    defaults = {
        "model": None,
        "backend": "local",
        "debug": False,
        "card_refresh_interval": 10,
    }

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        super().step_init(
            flow, graph, step_name, decorators, environment, flow_datastore, logger
        )

        # Validate that a model is specified
        if not self.attributes["model"]:
            raise ValueError(
                f"@vllm decorator on step '{step_name}' requires a 'model' parameter. "
                f"Example: @vllm(model='meta-llama/Llama-3.2-1B')"
            )

        # Attach the vllm status card
        self.attach_card_decorator(
            flow,
            step_name,
            "vllm_status",
            "blank",
            refresh_interval=self.attributes["card_refresh_interval"],
        )

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        @functools.wraps(step_func)
        def vllm_wrapper():
            self.vllm_manager = None
            self.status_card = None
            self.card_monitor_thread = None

            try:
                self.status_card = VLLMStatusCard(
                    refresh_interval=self.attributes["card_refresh_interval"]
                )

                def monitor_card():
                    try:
                        self.status_card.on_startup(current.card["vllm_status"])

                        while not getattr(
                            self.card_monitor_thread, "_stop_event", False
                        ):
                            try:
                                self.status_card.on_update(
                                    current.card["vllm_status"], None
                                )
                                import time

                                time.sleep(self.attributes["card_refresh_interval"])
                            except Exception as e:
                                if self.attributes["debug"]:
                                    print(f"[@vllm] Card monitoring error: {e}")
                                break
                    except Exception as e:
                        if self.attributes["debug"]:
                            print(f"[@vllm] Card monitor thread error: {e}")
                        self.status_card.on_error(current.card["vllm_status"], str(e))

                self.card_monitor_thread = threading.Thread(
                    target=monitor_card, daemon=True
                )
                self.card_monitor_thread._stop_event = False
                self.card_monitor_thread.start()

                self.vllm_manager = VLLMManager(
                    model=self.attributes["model"],
                    backend=self.attributes["backend"],
                    debug=self.attributes["debug"],
                    status_card=self.status_card,
                )

                if self.attributes["debug"]:
                    print(
                        "[@vllm] VLLMManager initialized."
                    )

            except Exception as e:
                if self.status_card:
                    self.status_card.add_event(
                        "error", f"Initialization failed: {str(e)}"
                    )
                    try:
                        self.status_card.on_error(current.card["vllm_status"], str(e))
                    except:
                        pass
                print(f"[@vllm] Error initializing VLLMManager: {e}")
                raise

            try:
                if self.status_card:
                    self.status_card.add_event("info", "Starting user step function")
                step_func()
                if self.status_card:
                    self.status_card.add_event(
                        "success", "User step function completed successfully"
                    )
            finally:
                if self.vllm_manager:
                    self.vllm_manager.terminate_models()

                if self.card_monitor_thread and self.status_card:
                    import time

                    try:
                        self.status_card.on_update(current.card["vllm_status"], None)
                    except Exception as e:
                        if self.attributes["debug"]:
                            print(f"[@vllm] Final card update error: {e}")
                    time.sleep(2)

                if self.card_monitor_thread:
                    self.card_monitor_thread._stop_event = True
                    self.card_monitor_thread.join(timeout=5)
                    if self.attributes["debug"]:
                        print("[@vllm] Card monitoring thread stopped.")

        return vllm_wrapper