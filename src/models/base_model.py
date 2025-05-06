"""
Base model class for LLM interfaces using LiteLLM.

This module defines the LLMModel class that provides a unified interface
to various LLM providers through LiteLLM.
"""

import asyncio

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
        logprobs: bool = False,
        top_logprobs: int | None = None,
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
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.kwargs = kwargs
        # The lower this value the more likely we get greedy sampling
        self.top_p = 0.001

    def _extract_topk_tokens_from_logprobs(self, choice) -> list[str]:
        """
        Helper to extract top-k tokens from a LiteLLM choice logprobs dict.
        Returns a list of top-k tokens (best guess first) or an empty list if not available.
        """
        try:
            if hasattr(choice, "logprobs") and choice.logprobs:
                topk_tokens = ["" for _ in range(self.top_logprobs)]
                if not hasattr(choice.logprobs, "content"):    
                    for content in choice.logprobs.content:
                        top_logprobs = content["top_logprobs"]
                        for ind, t in enumerate(top_logprobs):
                            topk_tokens[ind] += t.token
                else:
                    for i in range(len(choice.logprobs.top_logprobs)):
                        for ind, (key, value) in enumerate(choice.logprobs.top_logprobs[i].items()):
                            topk_tokens[ind] += key
        except Exception as e:
            print(f"Error extracting top-k tokens from logprobs: {e}")
            return []

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
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                top_p = self.top_p,
                **self.kwargs,
            )
            choice = response.choices[0]
            if self.top_logprobs is not None:
                topk_tokens = self._extract_topk_tokens_from_logprobs(choice)
            else:
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
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                top_p = self.top_p,
                **self.kwargs,
            )
            choice = response.choices[0]
            topk_tokens = self._extract_topk_tokens_from_logprobs(choice)
            if not topk_tokens:
                topk_tokens = [choice.message.content.strip()]
            return topk_tokens
        except Exception as e:
            print(f"Error generating async response from {self.model_name}: {e}")
            print(response)
            return [f"Error: {str(e)}"]

    def batch_generate(
        self, prompts_messages: list[list[dict[str, str]]]
    ) -> list[list[str]]:
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
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                top_p = self.top_p,
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
