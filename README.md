# Metaflow vLLM Extension

A Metaflow decorator for running vLLM inference servers as task sidecars.
This package is a thin wrapper around vLLMs robust batch inference APIs. 
The primary functionalities are to:
1. automate runtime operation, 
2. make complex profiling routines easy, and
3. provide templates for designing scalable batch inference deployments.

## Key Design Principles

- **One Model Per Server**: vLLM's OpenAI-compatible server serves one model per instance
- **Multiple Models**: Use separate `@vllm` decorators on different steps to serve multiple models
- **Local Backend**: Runs vLLM server as a subprocess on the task machine

## Usage

```python
from metaflow import FlowSpec, step, vllm

class MyFlow(FlowSpec):
    
    @vllm(model="meta-llama/Llama-3.2-1B")
    @step
    def start(self):
        import openai
        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123"
        )
        
        response = client.completions.create(
            model="meta-llama/Llama-3.2-1B",
            prompt="Hello, world!",
            max_tokens=50
        )
        
        print(response.choices[0].text)
        self.next(self.end)
        
    @step
    def end(self):
        pass
```

## Multiple Models

To use multiple models, create separate steps with separate `@vllm` decorators:

```python
@vllm(model="meta-llama/Llama-3.2-1B")
@step
def step_a(self):
    # Use Llama model
    pass

@vllm(model="Qwen/Qwen2.5-0.5B")  
@step  
def step_b(self):
    # Use Qwen model
    pass
```
