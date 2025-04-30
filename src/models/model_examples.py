"""
Examples of using the LiteLLM-based model interface.

This module provides examples of how to use the model factory with different LLM providers.
"""

from dotenv import load_dotenv

from models.model_factory import get_model


def openai_example(api_key: str = None) -> None:
    """
    Example of using OpenAI models with LiteLLM.
    
    Args:
        api_key: OpenAI API key (optional, can use environment variable)
    """
    # Use OpenAI GPT-3.5 Turbo
    model = get_model(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
        temperature=0.0,
        max_tokens=50
    )
    
    # Generate a response
    prompt = "What is the capital of France?"
    response = model.generate(prompt)
    
    print(f"Model: {model.name}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
    
    # Use OpenAI GPT-4
    model = get_model(
        model_name="gpt-4",
        api_key=api_key,
        temperature=0.0,
        max_tokens=50
    )
    
    # Generate a response
    prompt = "Explain quantum computing in one sentence."
    response = model.generate(prompt)
    
    print(f"Model: {model.name}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)


def anthropic_example(api_key: str = None) -> None:
    """
    Example of using Anthropic models with LiteLLM.
    
    Args:
        api_key: Anthropic API key (optional, can use environment variable)
    """
    # Use Anthropic Claude 3 Sonnet
    model = get_model(
        model_name="anthropic/claude-3-sonnet-20240229",
        api_key=api_key,
        temperature=0.0,
        max_tokens=50
    )
    
    # Generate a response
    prompt = "What is the capital of France?"
    response = model.generate(prompt)
    
    print(f"Model: {model.name}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
    
    # Use Anthropic Claude 3 Opus
    model = get_model(
        model_name="anthropic/claude-3-opus-20240229",
        api_key=api_key,
        temperature=0.0,
        max_tokens=50
    )
    
    # Generate a response
    prompt = "Explain quantum computing in one sentence."
    response = model.generate(prompt)
    
    print(f"Model: {model.name}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)


def together_ai_example(api_key: str = None) -> None:
    """
    Example of using Together AI models with LiteLLM.
    
    Args:
        api_key: Together AI API key (optional, can use environment variable)
    """
    # Use Together AI Llama 2
    model = get_model(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        api_key=api_key,
        temperature=0.0,
        max_tokens=50
    )
    
    # Generate a response
    prompt = "What is the capital of France?"
    response = model.generate(prompt)
    
    print(f"Model: {model.name}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
    


def run_all_examples() -> None:
    """Run all model examples."""
    print("Running OpenAI examples...")
    try:
        openai_example()
    except Exception as e:
        print(f"Error running OpenAI examples: {e}")
    
    print("\nRunning Anthropic examples...")
    try:
        anthropic_example()
    except Exception as e:
        print(f"Error running Anthropic examples: {e}")
    
    print("\nRunning Together AI examples...")
    try:
        together_ai_example()
    except Exception as e:
        print(f"Error running Together AI examples: {e}")


if __name__ == "__main__":
    load_dotenv()
    run_all_examples()
