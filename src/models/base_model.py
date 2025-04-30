"""
Base model class for LLM interfaces using LiteLLM.

This module defines the LLMModel class that provides a unified interface
to various LLM providers through LiteLLM.
"""

import os
from typing import Dict, List, Optional, Union

import litellm


class LLMModel:
    """Base class for LLM interfaces using LiteLLM."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 50,
        **kwargs
    ):
        """
        Initialize the LLM model using LiteLLM.
        
        Args:
            model_name: Name of the model (provider/model format for LiteLLM)
            api_key: API key for the model provider (if needed)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        
        # Set API key if provided or get from environment
        if api_key:
            # Set the appropriate environment variable based on the model provider
            if 'openai' in model_name.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'anthropic' in model_name.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            elif 'together' in model_name.lower():
                os.environ['TOGETHER_API_KEY'] = api_key
            # Add more providers as needed
        
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using LiteLLM.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The model's response
        """
        try:
            # Use LiteLLM to generate a response
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response from {self.model_name}: {e}")
            return f"Error: {str(e)}"
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self.model_name
