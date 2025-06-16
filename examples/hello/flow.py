from metaflow import FlowSpec, step, pypi, vllm, kubernetes, secrets, environment
from metaflow.profilers import gpu_profile


class HelloVLLM(FlowSpec):

    @step
    def start(self):
        self.next(self.test_first_model, self.test_second_model)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(gpu=2, cpu=32, memory=24000)
    @pypi(packages={"vllm": "0.6.1", "openai": "1.52.0", "httpx": "0.27.0", "setuptools": "<81"})
    @vllm(model = "meta-llama/Llama-3.2-1B", debug=True)
    @step
    def test_first_model(self):
        """
        An introduction to using the @vllm decorator.

        Notice that the @kubernetes decorator uses default base image.
        We install the vLLM package and OpenAI client using normal @pypi usage.
        This dependency management approach contrasts that of the end step.

        This step serves one model via a vLLM server.
        It also turns on debugging for verbose logs.
        
        NOTE: vLLM serves one model per server instance. If you need multiple
        models, create separate steps with separate @vllm decorators.
        """
        import openai
        import httpx

        # Use OpenAI-compatible API to communicate with vLLM server
        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",  # vLLM doesn't require real API key
            http_client=httpx.Client(proxies={}),
        )

        # Using completions API instead of chat completions to avoid chat template issues
        response = client.completions.create(
            model="meta-llama/Llama-3.2-1B",
            prompt="What are the leading Chinese tech companies?",
            max_tokens=150,
            temperature=0.7,
        )
        
        print(f"\n\n[@test] Response from Llama-3.2-1B: {response.choices[0].text}\n\n")
        self.next(self.join)

    @secrets(sources=["outerbounds.eddie-hf"])
    @gpu_profile(interval=1)
    @kubernetes(gpu=2, cpu=32, memory=24000)
    @pypi(packages={"vllm": "0.6.1", "openai": "1.52.0", "httpx": "0.27.0", "setuptools": "<81"})
    @vllm(model="Qwen/Qwen2.5-0.5B", debug=True)
    @step
    def test_second_model(self):
        """
        Demonstrates how to use a second model.
        Each @vllm decorator creates a separate server instance.
        """
        import openai
        import httpx

        # Use OpenAI-compatible API to communicate with vLLM server
        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",  # vLLM doesn't require real API key
            http_client=httpx.Client(proxies={}),
        )

        response = client.completions.create(
            model="Qwen/Qwen2.5-0.5B",
            prompt="What are the leading Chinese tech companies?",
            max_tokens=150,
            temperature=0.7,
        )
        
        print(f"\n\n[@test] Response from Qwen2.5-0.5B: {response.choices[0].text}\n\n")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HelloVLLM()
