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
    Parameter,  
    Config,
    config_expr
)
from metaflow.profilers import gpu_profile


class HubToVLLMTestFlow(FlowSpec):

    config = Config("config", default="config.json")

    @pypi(**config_expr("config.huggingface_hub.environment"))
    @huggingface_hub
    @step
    def start(self):

        self.hf_model_for_vllm = current.huggingface_hub.snapshot_download(
            repo_id=self.config.huggingface_hub.repo_id,
            force_download=self.config.huggingface_hub.force_download,
            allow_patterns="*"
            if self.config.huggingface_hub.allow_patterns is None
            else self.config.huggingface_hub.allow_patterns.split(","),
        )
        self.next(self.test_model)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(gpu=2, cpu=8, memory=16_000)
    @pypi(**config_expr("config.inference_environment"))
    @environment(vars={"HF_HOME": config_expr("config.huggingface_hub.hf_home")})
    @model(load=[("hf_model_for_vllm", config_expr("config.huggingface_hub.hf_home + '/models--' + config.huggingface_hub.repo_id.replace('/', '--')"))])
    @vllm(model=config_expr("config.huggingface_hub.repo_id"), engine_args={"tensor_parallel_size": 1})
    @step
    def test_model(self):
        import openai
        import httpx

        # Use OpenAI-compatible API to communicate with vLLM server
        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="test123",    
            http_client=httpx.Client(proxies={}),
        )

        # Using completions API instead of chat completions to avoid chat template issues
        response = client.completions.create(
            model=self.config.huggingface_hub.repo_id,
            prompt="What are the leading Chinese tech companies?",
            max_tokens=150,
            temperature=0.7,
        )
        
        print(f"\n\n[@test] Response from Llama-3.2-1B: {response.choices[0].text}\n\n")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HubToVLLMTestFlow()
