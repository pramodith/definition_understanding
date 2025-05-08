from unittest.mock import patch

from models.base_model import LLMModel
from models.model_factory import get_model


# Dummy classes to avoid real model instantiation for tests
class DummyVLLMModel:
    def __init__(self, model_name, api_key=None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs


def test_get_model_openai():
    model = get_model("gpt-4", api_key="key", is_local=False)
    assert isinstance(model, LLMModel)
    assert model.model_name == "openai/gpt-4"


def test_get_model_anthropic():
    model = get_model("claude-3-sonnet", api_key="key", is_local=False)
    assert isinstance(model, LLMModel)
    assert model.model_name == "anthropic/claude-3-sonnet"


def test_get_model_together_ai():
    model = get_model("meta-llama-3-8b", api_key="key", is_local=False)
    assert isinstance(model, LLMModel)
    assert model.model_name == "together_ai/meta-llama-3-8b"


@patch("models.model_factory.VLLMModel", new=DummyVLLMModel)
def test_get_model_vllm():
    model = get_model("Qwen/Qwen3-0.6B", api_key="key", is_local=True)
    assert isinstance(model, DummyVLLMModel)
    assert model.model_name == "Qwen/Qwen3-0.6B"
