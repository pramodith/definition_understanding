"""
Model factory module using LiteLLM.

This module provides a factory function to create instances of LLM models
using LiteLLM for a unified interface to various LLM providers.
"""

from models.base_model import LLMModel
from models.vllm_model import VLLMModel


def get_model(
    model_name: str, api_key: str | None = None, is_local: bool = True, **kwargs
) -> LLMModel:
    """
    Get an instance of the specified model using LiteLLM.

    Args:
        model_name: Name of the model to use (in LiteLLM format)
        api_key: API key for the model service (if needed)
        is_local: Whether to use a local inference engine
        **kwargs: Additional model-specific parameters

    Returns:
        An instance of the LLMModel configured for the specified model

    Examples:
        # OpenAI models
        model = get_model("gpt-3.5-turbo")
        model = get_model("gpt-4")

        # Anthropic models
        model = get_model("anthropic/claude-3-opus-20240229")
        model = get_model("anthropic/claude-3-sonnet-20240229")

        # Together AI models
        model = get_model("together/llama-2-70b-chat")
        model = get_model("together/mistral-7b-instruct")
    """
    if is_local:
        return VLLMModel(model_name=model_name, api_key=api_key, **kwargs)
    else:
        # Map common model names to LiteLLM format if needed
        litellm_model_name = model_name

        # If the model name doesn't include a provider prefix, add it
        if model_name.startswith("gpt-"):
            litellm_model_name = f"openai/{model_name}"
        elif model_name.startswith("claude"):
            litellm_model_name = f"anthropic/{model_name}"
        elif model_name.startswith("gemini"):
            litellm_model_name = f"gemini/{model_name}"
        elif model_name.startswith("phi"):
            litellm_model_name = f"microsoft/{model_name}"
        # Together AI models for open-source models
        elif any(
            name in model_name.lower() for name in ["meta-llama", "mistral", "qwen"]
        ):
            litellm_model_name = f"together_ai/{model_name}"

        # Create and return the LLMModel instance
        return LLMModel(model_name=litellm_model_name, api_key=api_key, **kwargs)
