"""
Base model class for LLM interfaces using LiteLLM.

This module defines the LLMModel class that provides a unified interface
to various LLM providers through LiteLLM.
"""

import asyncio
import os

import litellm
from litellm import acompletion, batch_completion


class LLMModel:
    """Base class for LLM interfaces using LiteLLM."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 50,
        top_logprobs: int = 5,
        **kwargs,
    ):
        """
        Initialize the LLM model using LiteLLM.

        Args:
            model_name: Name of the model (provider/model format for LiteLLM)
            api_key: API key for the model provider (if needed)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            top_logprobs: Number of top log-probabilities to return
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name

        # Set API key if provided or get from environment
        if api_key:
            # Set the appropriate environment variable based on the model provider
            if "openai" in model_name.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in model_name.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "together" in model_name.lower():
                os.environ["TOGETHER_API_KEY"] = api_key
            # Add more providers as needed

        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.kwargs = kwargs

    def _extract_topk_tokens_from_logprobs(self, choice) -> list[str]:
        """
        Helper to extract top-k tokens from a LiteLLM choice logprobs dict.
        Returns a list of top-k tokens (best guess first) or an empty list if not available.
        """
        topk_tokens = []
        if hasattr(choice, "logprobs") and choice.logprobs and "top_logprobs" in choice.logprobs:
            top_logprobs = choice.logprobs["top_logprobs"]
            if top_logprobs and len(top_logprobs) > 0:
                first_token_probs = top_logprobs[0]
                sorted_tokens = sorted(first_token_probs.items(), key=lambda x: x[1], reverse=True)
                topk_tokens = [token for token, _ in sorted_tokens]
        return topk_tokens

    def generate(self, prompt_messages: list[dict[str, str]]) -> list[str]:
        """
        Generate a response using LiteLLM and extract top-k predicted tokens.

        Args:
            prompt_messages: The input prompt

        Returns:
            List of top-k predicted tokens/words (best guess first)
        """
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **self.kwargs,
            )
            choice = response.choices[0]
            topk_tokens = self._extract_topk_tokens_from_logprobs(choice)
            if not topk_tokens:
                topk_tokens = [choice.message.content.strip()]
            return topk_tokens
        except Exception as e:
            print(f"Error generating response from {self.model_name}: {e}")
            return [f"Error: {str(e)}"]

    async def agenerate(self, prompt_messages: list[dict[str, str]]) -> list[str]:
        """
        Generate a response asynchronously using LiteLLM and extract top-k predicted tokens.

        Args:
            prompt_messages: The input prompt

        Returns:
            List of top-k predicted tokens/words (best guess first)
        """
        try:
            response = await acompletion(
                model=self.model_name,
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **self.kwargs,
            )
            choice = response.choices[0]
            topk_tokens = self._extract_topk_tokens_from_logprobs(choice)
            if not topk_tokens:
                topk_tokens = [choice.message.content.strip()]
            return topk_tokens
        except Exception as e:
            print(f"Error generating async response from {self.model_name}: {e}")
            return [f"Error: {str(e)}"]

    def batch_generate(self, prompts_messages: list[list[dict[str, str]]]) -> list[list[str]]:
        """
        Generate responses for multiple prompts in a batch using LiteLLM and extract top-k predicted tokens.

        Args:
            prompts_messages: List of input prompts

        Returns:
            List of lists of top-k predicted tokens/words (best guess first)
        """
        try:
            messages_list = prompts_messages
            responses = batch_completion(
                model=self.model_name,
                messages=messages_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **self.kwargs,
            )
            results = []
            for resp in responses:
                choice = resp.choices[0]
                topk_tokens = self._extract_topk_tokens_from_logprobs(choice)
                if not topk_tokens:
                    topk_tokens = [choice.message.content.strip()]
                results.append(topk_tokens)
            return results
        except Exception as e:
            print(f"Error generating batch responses from {self.model_name}: {e}")
            return [[f"Error: {str(e)}"]] * len(prompts_messages)

    async def abatch_generate(
        self,
        prompts_messages: list[list[dict[str, str]]],
        batch_size: int = 20,
        delay: float = 0.5,
    ) -> list[list[str]]:
        """
        Generate responses for multiple prompts asynchronously in batches using LiteLLM and extract top-k predicted tokens.

        This method processes prompts in smaller batches to avoid overwhelming the API
        and to handle large numbers of prompts efficiently.

        Args:
            prompts_messages: List of input prompts
            batch_size: Number of prompts to process in each batch

        Returns:
            List of lists of top-k predicted tokens/words (best guess first)
        """
        results = []
        for i in range(0, len(prompts_messages), batch_size):
            batch = prompts_messages[i : i + batch_size]
            batch_tasks = [self.agenerate(prompt_message) for prompt_message in batch]
            try:
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                processed_results = []
                for res in batch_results:
                    if isinstance(res, Exception):
                        processed_results.append([f"Error: {str(res)}"])
                    else:
                        processed_results.append(res)
                results.extend(processed_results)
            except Exception as e:
                print(f"Error in batch processing from {self.model_name}: {e}")
                results.extend([[f"Error: {str(e)}"]] * len(batch))
            await asyncio.sleep(delay)
        return results

    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self.model_name
