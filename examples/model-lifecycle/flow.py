from metaflow import (
    FlowSpec,
    step,
    pypi,
    vllm,
    model,
    kubernetes,
    secrets,
    environment,
    huggingface_hub,
    current,
    Config,
    config_expr,
)
from metaflow.profilers import gpu_profile


class HubToVLLMTestFlow(FlowSpec):

    config = Config("config", default="smol-config.json")

    @secrets(sources=["outerbounds.eddie-hf"])
    @pypi(**config_expr("config.huggingface_hub.environment"))
    @huggingface_hub
    @step
    def start(self):
        self.hf_model_for_vllm = current.huggingface_hub.snapshot_download(
            repo_id=self.config.huggingface_hub.repo_id,
            force_download=self.config.huggingface_hub.force_download,
            allow_patterns=(
                "*"
                if self.config.huggingface_hub.allow_patterns is None
                else self.config.huggingface_hub.allow_patterns.split(",")
            ),
        )
        self.next(self.test_model)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(**config_expr("config.inference_environment_resources"))
    @environment(vars={"HF_HOME": config_expr("config.huggingface_hub.hf_home")}) # vLLM will look here to load weights.
    @model(
        load=[
            (
                "hf_model_for_vllm",
                config_expr(
                    "config.huggingface_hub.hf_home + '/models--' + config.huggingface_hub.repo_id.replace('/', '--')"
                ),
            )
        ]
    )
    @vllm(
        model=config_expr("config.huggingface_hub.repo_id"),
        engine_args=config_expr("config.vllm_engine_args"),
    )
    @step
    def test_model(self):
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1250,
            top_p=0.9
        )

        # Generate response using native engine
        prompt = "Identify the top 5 comedy bands of all time. What makes them great?"
        outputs = current.vllm.llm.generate([prompt], sampling_params)
        
        response_text = outputs[0].outputs[0].text

        print(
            f"\n\n[@test] Response from {self.config.huggingface_hub.repo_id}: {response_text}\n\n"
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HubToVLLMTestFlow()
