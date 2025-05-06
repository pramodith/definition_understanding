"""
Local model implementation using vLLM.

This module defines the VLLMModel class that provides an interface
to run local models using vLLM for efficient inference.
"""

from typing import List, Dict

try:
    import vllm
    from vllm import SamplingParams, LLM
    from transformers import AutoTokenizer
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype='half'
        )

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        tokenized_messages = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return tokenized_messages

    def batch_generate(self, prompts_messages: list[list[dict]], batch_size: int = 5):
        """
        Batch inference for a list of prompt_messages (each is a list of dicts).
        Returns a list of lists (top-k per prompt, but here just one per prompt).
        """
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            top_k=1,
        )
        # Convert each message list to a prompt string
        prompts = [self._convert_messages_to_prompt(msgs) for msgs in prompts_messages]
        outputs = self.llm.generate(prompts, sampling_params)
        # For compatibility with top-k, return a list of lists
        batch_results = []
        for output in outputs:
            if output.outputs:
                batch_results.append([out.text.strip() for out in output.outputs])
            else:
                batch_results.append([""])
        return batch_results


if __name__ == "__main__":
    # Example usage
    model = VLLMModel(model_name="Qwen/Qwen3-0.6B", temperature=0.0, max_tokens=50)
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    output = model.batch_generate([prompt_messages])
    print("Generated text:", output)
