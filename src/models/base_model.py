"""
Base model class for LLM interfaces using LiteLLM.

This module defines the LLMModel class that provides a unified interface
to various LLM providers through LiteLLM.
"""

import os
import asyncio
from typing import Dict, List, Optional, Union, Any

import litellm
from litellm import batch_completion, acompletion


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
    
    def generate(self, prompt_messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using LiteLLM.
        
        Args:
            prompt_messages: The input prompt
            
        Returns:
            The model's response
        """
        try:
            # Use LiteLLM to generate a response
            response = litellm.completion(
                model=self.model_name,
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response from {self.model_name}: {e}")
            return f"Error: {str(e)}"
            
    async def agenerate(self, prompt_messages: List[Dict[str, str]]) -> str:
        """
        Generate a response asynchronously using LiteLLM.
        
        Args:
            prompt_messages: The input prompt
            
        Returns:
            The model's response
        """
        try:
            # Use LiteLLM to generate a response asynchronously
            response = await acompletion(
                model=self.model_name,
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating async response from {self.model_name}: {e}")
            return f"Error: {str(e)}"
            
    def batch_generate(self, prompts_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate responses for multiple prompts in a batch using LiteLLM.
        
        Args:
            prompts_messages: List of input prompts
            
        Returns:
            List of model responses
        """
        try:
            # Format messages for batch completion
            messages_list = prompts_messages
            
            # Use LiteLLM batch completion
            responses = batch_completion(
                model=self.model_name,
                messages=messages_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            # Extract content from responses
            results = [resp.choices[0].message.content.strip() for resp in responses]
            return results
            
        except Exception as e:
            print(f"Error generating batch responses from {self.model_name}: {e}")
            return [f"Error: {str(e)}"] * len(prompts)
            
    async def abatch_generate(self, prompts_messages: List[List[Dict[str, str]]], batch_size: int = 20, delay: float = 0.5) -> List[str]:
        """
        Generate responses for multiple prompts asynchronously in batches using LiteLLM.
        
        This method processes prompts in smaller batches to avoid overwhelming the API
        and to handle large numbers of prompts efficiently.
        
        Args:
            prompts_messages: List of input prompts
            batch_size: Number of prompts to process in each batch
            
        Returns:
            List of model responses in the same order as the input prompts
        """
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts_messages), batch_size):
            batch = prompts_messages[i:i+batch_size]
            batch_tasks = [self.agenerate(prompt_message) for prompt_message in batch]
            
            try:
                # Run batch of async tasks concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle exceptions in results
                processed_results = []
                for res in batch_results:
                    if isinstance(res, Exception):
                        processed_results.append(f"Error: {str(res)}")
                    else:
                        processed_results.append(res)
                        
                results.extend(processed_results)
                
            except Exception as e:
                print(f"Error in batch processing from {self.model_name}: {e}")
                # If the entire batch fails, add error messages for all prompts in this batch
                results.extend([f"Error: {str(e)}"] * len(batch))
            
            await asyncio.sleep(delay)
        
        return results
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self.model_name
