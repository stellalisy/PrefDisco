"""
api_client.py

Implements the client interface for language model services in the Causal Preference Evolution Framework.
Modified to use the specified load_model function.
"""

import json
import logging
import time
import random
import re
from typing import Dict, Any, List, Optional, Union, Callable

from llm.agents.load_model import load_model

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for accessing language model services. Provides a unified interface
    for text generation, prompt formatting, and response handling.
    """

    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Initialize the LLM client.

        Args:
            config: Configuration parameters containing model name and other settings
        """
        self.model_name = config.get("model", "gpt-4o")

        # Set default configuration or update with provided config
        self.config = {
            "max_retries": 3,
            "retry_delay": 2,  # seconds
            "max_tokens": 4096,
            "temperature": 0.7,
            "cache_responses": True,
            "log_prompts": False,
            "log_responses": False,
            "random_seed": 42,
            "reasoning_effort": "high"
        }

        # Update with provided config
        self.config.update(config)

        # Extract any additional kwargs for load_model
        model_kwargs = config.get("model_kwargs", {})

        # Initialize model
        self.llm = load_model(self.model_name, **model_kwargs)

        # Cache for responses
        self._response_cache = {}

        # Request counter for logging and rate limiting
        self._request_count = 0

        random.seed(self.config["random_seed"])

        logger.info(f"LLMClient initialized with model: {self.model_name}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        regenerate_if_unfinished: bool = False,
        enable_thinking: bool = True
    ) -> str:
        """
        Chat with the model.

        Args:
            messages: List of messages to chat with (system, user, assistant, user, ...)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
        """
        if self.config["cache_responses"]:
            cache_key = self._create_cache_key(messages, max_tokens, temperature, response_format)
            if cache_key in self._response_cache:
                logger.debug("Using cached response")
                return self._response_cache[cache_key]

        # Set up generation parameters
        tokens = max_tokens or self.config["max_tokens"]
        temp = temperature or self.config["temperature"]
        effort = reasoning_effort or self.config.get("reasoning_effort", "high")

        # Set json_mode if response_format is json
        json_mode = response_format == "json"
        if response_format == "json" and "Please respond with clean JSON" not in messages[-1]["content"]:
            messages[-1]["content"] = f"{messages[-1]['content']}\n\nPlease respond with clean JSON only, without explanations or code blocks, do not include \"```json\"."

        # Log the messages if enabled
        if self.config["log_prompts"]:
            # for each message, print the first 100 characters of the content
            abbrev_messages = '[\n'
            for message in messages:
                abbrev_messages += f"    {{'role': '{message['role']}', 'content': '{message['content'][:100]}...{message['content'][-100:]}'}}\n" if len(message['content']) > 200 else f"    {{'role': '{message['role']}', 'content': '{message['content']}'}}\n"
            abbrev_messages += ']'
            logger.debug(f"Messages: {abbrev_messages}")

        # Retry logic
        retries = 0
        while retries <= self.config["max_retries"]:
            try:
                # Increment request counter
                self._request_count += 1

                # Use the model to generate a response
                response = self.llm.interact(
                    prompt=messages,
                    max_tokens=tokens,
                    temperature=temp,
                    response_format=response_format,
                    reasoning_effort=effort,
                    json_mode=json_mode,
                    enable_thinking=enable_thinking
                )
                response_text, input_tokens, cached_tokens, output_tokens, reasoning_tokens = response["response_text"], response["input_tokens"], response["cached_tokens"], response["output_tokens"], response.get("reasoning_tokens", 0)

                if response_format == "json":
                    response_text = re.sub(r"^```(?:json)?\n|\n```$", "", response_text.strip())

                # Log the response if enabled
                if self.config["log_responses"]:
                    logger.debug(f"Response: {response_text[:200]}...")

                ret = {
                    "response_text": response_text,
                    "input_tokens": input_tokens,
                    "cached_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens
                }

                if "finish_reason" in response:
                    ret["finish_reason"] = response["finish_reason"]
                    if regenerate_if_unfinished and ret["finish_reason"] != "stop":
                        logger.warning(f"Response did not finish with stop reason (got '{ret['finish_reason']}'). Using truncated response.")

                # Cache the response if enabled
                if self.config["cache_responses"] and cache_key:
                    self._response_cache[cache_key] = ret

                return ret

            except Exception as e:
                retries += 1
                logger.warning(f" [{time.strftime('%Y-%m-%d %H:%M:%S')}] API request failed (attempt {retries}/{self.config['max_retries']}): {e}")

                if retries <= self.config["max_retries"]:
                    # Add jitter to retry delay (between 0.5 and 1.5 times the base delay)
                    jitter = random.uniform(0.5, 1.5)
                    delay = self.config["retry_delay"] * jitter * retries
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f" [{time.strftime('%Y-%m-%d %H:%M:%S')}] API request failed after {self.config['max_retries']} retries.")
                    raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        regenerate_if_unfinished: bool = False,
        enable_thinking: bool = True
    ) -> str:
        """
        Generate text based on a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            response_format: Optional format specification (e.g., "json")
            system_prompt: Optional system prompt to override default

        Returns:
            Generated text

        Raises:
            Exception: If API request fails after retries
        """
        # Check cache if enabled
        if self.config["cache_responses"]:
            cache_key = self._create_cache_key(prompt, max_tokens, temperature, response_format, system_prompt)
            if cache_key in self._response_cache:
                logger.debug("Using cached response")
                return self._response_cache[cache_key]

        # Set up generation parameters
        tokens = max_tokens or self.config.get("max_tokens", 512)
        temp = temperature or self.config.get("temperature", None)
        effort = reasoning_effort or self.config.get("reasoning_effort", "high")

        # Prepare full prompt with system message if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Set json_mode if response_format is json
        json_mode = (response_format == "json" or isinstance(response_format, dict))
        if response_format == "json" and "Please respond with clean JSON" not in full_prompt:
            full_prompt = f"{full_prompt}\n\nPlease respond with clean JSON only, without explanations or code blocks, do not include \"```json\"."

        # Log the prompt if enabled
        if self.config["log_prompts"]:
            logger.debug(f"Prompt: {full_prompt[:200]}...")

        # Retry logic
        retries = 0
        while retries < self.config["max_retries"]:
            try:
                # Increment request counter
                self._request_count += 1
                # Use the model to generate a response
                response = self.llm.interact(
                    prompt=full_prompt,
                    max_tokens=tokens,
                    temperature=temp,
                    response_format=response_format,
                    reasoning_effort=effort,
                    json_mode=json_mode,
                    enable_thinking=enable_thinking
                )
                response_text = response.get("response_text")

                # Handle None response from API
                if response_text is None:
                    retries += 1
                    logger.warning(f"API returned None response (attempt {retries}/{self.config['max_retries']})")
                    if retries < self.config["max_retries"]:
                        time.sleep(self.config["retry_delay"])
                        continue
                    else:
                        raise ValueError("API returned None response after all retries")

                if json_mode:
                    response_text = re.sub(r"^```(?:json)?\n|\n```$", "", response_text.strip())

                # Log the response if enabled
                if self.config["log_responses"]:
                    logger.debug(f"Response: {response_text[:200]}...")

                response["response_text"] = response_text

                if "finish_reason" in response:
                    if regenerate_if_unfinished:
                        assert response["finish_reason"] == "stop", "Response did not finish with stop reason"

                # Cache the response if enabled
                if self.config["cache_responses"] and cache_key:
                    self._response_cache[cache_key] = response

                return response

            except Exception as e:
                retries += 1
                logger.warning(f" [{time.strftime('%Y-%m-%d %H:%M:%S')}] API request failed (attempt {retries}/{self.config['max_retries']}): {e}")
                # breakpoint()
                if retries < self.config["max_retries"]:
                    # Add jitter to retry delay (between 0.5 and 1.5 times the base delay)
                    jitter = random.uniform(0.5, 1.5)
                    delay = self.config["retry_delay"] * jitter * retries
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f" [{time.strftime('%Y-%m-%d %H:%M:%S')}] API request failed after {self.config['max_retries']} retries.")
                    raise

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt to override default

        Returns:
            List of generated texts
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            results.append(result)

        return results

    def generate_with_context(
        self,
        context: str,
        query: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text with separate context and query.

        Args:
            context: Context information
            query: Query or instruction
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt to override default

        Returns:
            Generated text
        """
        # Combine context and query into a single prompt
        prompt = f"""
        Context information:
        {context}

        Query:
        {query}
        """

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt
        )

    def format_as_json(self, content: Dict[str, Any]) -> str:
        """
        Format dictionary content as a JSON string.

        Args:
            content: Dictionary to format

        Returns:
            JSON-formatted string
        """
        return json.dumps(content, indent=2)

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse a JSON response string.

        Args:
            response: JSON string

        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self._response_cache = {}
        logger.debug("Cleared response cache")

    def _create_cache_key(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        response_format: Optional[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a cache key for the request.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            response_format: Response format specification
            system_prompt: System prompt

        Returns:
            Cache key string
        """
        # Combine parameters into a single string
        key_components = [
            prompt if isinstance(prompt, str) else json.dumps(prompt),
            str(max_tokens or self.config["max_tokens"]),
            str(temperature or self.config["temperature"]),
            json.dumps(response_format, sort_keys=True) if isinstance(response_format, dict) else (response_format or "default"),
            system_prompt or "default"
        ]

        # Join with a separator and return
        return "||".join(key_components)