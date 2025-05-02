"""
Local model implementation using vLLM.

This module defines the VLLMModel class that provides an interface
to run local models using vLLM for efficient inference.
"""

from typing import List, Dict

try:
    import vllm
    from vllm import SamplingParams, LLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not installed. Install with 'pip install vllm' to use local models.")

from models.base_model import LLMModel

class VLLMModel(LLMModel):
    """Class for local LLM interfaces using vLLM."""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 50,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with 'pip install vllm'.")
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **kwargs,
        )
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine = LLMEngine.from_engine_args(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                prompt += f"{content}\n"
        return prompt.strip()

    def generate(self, prompt_messages: List[Dict[str, str]]) -> str:
        prompt = self._convert_messages_to_prompt(prompt_messages)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        outputs = self.engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""

if __name__ == "__main__":
    # Example usage
    model = VLLMModel(model_name="meta-llama/Llama-3-8B-Instruct", temperature=0.0, max_tokens=32)
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    output = model.generate(prompt_messages)
    print("Generated text:", output)
