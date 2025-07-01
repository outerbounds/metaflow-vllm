from metaflow import (
    FlowSpec,
    step,
    vllm,
    kubernetes,
    secrets,
    Config,
    config_expr,
    current,
    card,
    pypi,
)
from metaflow.profilers import gpu_profile
from metaflow.cards import Table


class HelloVLLM(FlowSpec):

    config = Config("config", default="smol-config.json")

    @step
    def start(self):
        self.next(self.test_first_model, self.test_second_model)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(**config_expr("config.inference_environment_resources"))
    @vllm(
        model=config_expr("config.model_a"),
        engine_args=config_expr("config.vllm_engine_args"),
        openai_api_server=True,
    )
    @step
    def test_first_model(self):
        """
        An introduction to using the @vllm decorator in API server mode.

        This example uses openai_api_server=True for backward compatibility.
        The decorator starts a vLLM subprocess server and provides an OpenAI-compatible API.

        For better performance, consider using the native engine mode (see flow_native.py).

        NOTE: vLLM serves one model per server instance. If you need multiple
        models, create separate steps with separate @vllm decorators.
        """
        import openai # pylint: disable=import-error

        # Use OpenAI-compatible API to communicate with vLLM server
        client = openai.OpenAI(
            base_url=current.vllm.local_endpoint,
            api_key=current.vllm.local_api_key,
        )

        # Using completions API instead of chat completions to avoid chat template issues
        self.responses = {
            "model": [self.config.model_a] * len(self.config.prompts),
            "prompts": self.config.prompts,
            "responses": [],
        }
        for prompt in self.config.prompts:
            response = client.completions.create(
                model=self.config.model_a,
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
            )
            self.responses["responses"].append(response.choices[0].text)
        self.next(self.join)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(**config_expr("config.inference_environment_resources"))
    @vllm(
        model=config_expr("config.model_b"),
        engine_args=config_expr("config.vllm_engine_args"),
        openai_api_server=True,
    )
    @step
    def test_second_model(self):
        """
        Demonstrates how to use a second model in API server mode.
        Each @vllm decorator with openai_api_server=True creates a separate server instance.
        """
        import openai # pylint: disable=import-error

        # Use OpenAI-compatible API to communicate with vLLM server
        client = openai.OpenAI(
            base_url=current.vllm.local_endpoint,
            api_key=current.vllm.local_api_key,
        )

        self.responses = {
            "model": [self.config.model_b] * len(self.config.prompts),
            "prompts": self.config.prompts,
            "responses": [],
        }
        for prompt in self.config.prompts:
            response = client.completions.create(
                model=self.config.model_b,
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
            )
            self.responses["responses"].append(response.choices[0].text)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.responses = [i.responses for i in inputs]
        self.next(self.end)

    @card
    @pypi(packages={"pandas": "2.3.0"})
    @step
    def end(self):
        import pandas as pd # pylint: disable=import-error

        prompts = self.responses[0]["prompts"]
        model_a_responses = self.responses[0]["responses"]
        model_b_responses = self.responses[1]["responses"]
        self.df = pd.DataFrame(
            {
                "prompt": prompts,
                "model_a": model_a_responses,
                "model_b": model_b_responses,
            }
        )
        current.card.append(Table.from_dataframe(self.df))


if __name__ == "__main__":
    HelloVLLM()
