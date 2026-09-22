# https://github.com/openai/openai-python
import os
import json
import time
import asyncio
import openai
import backoff
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )  # for exponential backoff
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseAgent
import yaml

# Import OpenAI exceptions with compatibility for different library versions
try:
    from openai import RateLimitError, APIError, Timeout, APIConnectionError, BadRequestError
except ImportError:
    try:
        # Try importing from openai.error (older versions)
        from openai.error import RateLimitError, APIError, Timeout, APIConnectionError, BadRequestError
    except (ImportError, AttributeError):
        # Fallback to Exception if imports fail
        RateLimitError = Exception
        APIError = Exception
        Timeout = Exception
        APIConnectionError = Exception
        BadRequestError = Exception

OPENAI_RETRY_EXCEPTIONS = (RateLimitError, APIError, Timeout, APIConnectionError)

USAGE_TRACKING_DIR = os.path.join(os.path.dirname(__file__), 'usage_tracking')
os.makedirs(USAGE_TRACKING_DIR, exist_ok=True)

STRUCTURED_OUTPUT_MODELS = ("gpt-4o-mini", "gpt-4o-2024-08-06")

class GPT3BaseAgent(BaseAgent):
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

        if not hasattr(self.args, 'api_info'):
            raise ValueError("args.api_info is required")
        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found (current directory: {os.getcwd()})")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        if self.api_info.get("api_type", "openai") == "openai":
            self.client = OpenAI(api_key=self.api_info.get("api_key", os.getenv('OPENAI_API_KEY')))
        elif self.api_info.get("api_type", "openai") == "azure":
            self.client = AzureOpenAI(api_key=self.api_info.get("api_key", os.getenv('AZURE_OPENAI_API_KEY')),
                                      azure_endpoint=self.api_info.get("api_base", os.getenv("AZURE_OPENAI_ENDPOINT")),
                                      api_version=self.api_info.get("api_version", "2024-06-01"))
        else:
            raise ValueError(f"API type {self.api_info.get('api_type')} not supported")

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-3.5-turbo-instruct"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0
        if not hasattr(self.args, 'n'):
            self.args.n = 1

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate(self, prompt, temperature=None, max_tokens=None):
        completion = self.client.completions.create(model=self.args.model,
                                                    prompt=prompt,
                                                    temperature=self.args.temperature if temperature is None else temperature,
                                                    max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                    top_p=self.args.top_p,
                                                    frequency_penalty=self.args.frequency_penalty,
                                                    presence_penalty=self.args.presence_penalty,
                                                    stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                                                    logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                                                    echo=self.args.echo if hasattr(self.args, 'echo') else False,
                                                    n=self.args.n if hasattr(self.args, 'n') else 1)

        return completion

    def preprocess_input(self, text):
        return text

    def _postprocess_output(self, outputs):
        responses = [c.text.strip() for c in outputs.choices]

        try: cached_tokens = outputs.usage.prompt_tokens_details.cached_tokens
        except: cached_tokens = 0
        input_tokens = outputs.usage.prompt_tokens - cached_tokens
        output_tokens = outputs.usage.completion_tokens
        finish_reasons = [c.finish_reason for c in outputs.choices]

        return {
            "response_text": responses[0],
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reasons[0]
        }

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt, temperature=None, max_tokens=None, **kwargs):
        outputs = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
        responses = self._postprocess_output(outputs)

        return responses

class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-4o"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _json_generate(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
        ):
        if "gpt-5" in self.args.model:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=[
                                                            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                            {"role": "user", "content": f"{prompt}"}
                                                            ],
                                                         )
        else:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=[
                                                            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                            {"role": "user", "content": f"{prompt}"}
                                                            ],
                                                         temperature=self.args.temperature if temperature is None else temperature,
                                                         max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _json_generate_from_messages(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
            ):
        if "gpt-5" in self.args.model:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=messages,
                                                         )
        else:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=messages,
                                                         temperature=self.args.temperature if temperature is None else temperature,
                                                         max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _structured_generate(self, prompt, temperature=None, max_tokens=None, response_format=None):
        completion = self.client.beta.chat.completions.parse(model=self.args.model,
                                                             messages=[{"role": "user", "content": f"{prompt}"}],
                                                             temperature=self.args.temperature if temperature is None else temperature,
                                                             max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                             response_format=response_format)

        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _structured_generate_from_messages(self, messages, temperature=None, max_tokens=None, response_format=None):
        completion = self.client.beta.chat.completions.parse(model=self.args.model,
                                                             messages=messages,
                                                             temperature=self.args.temperature if temperature is None else temperature,
                                                             max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                             response_format=response_format)
        return completion

    def _postprocess_output(self, outputs, response_format=None):
        if self.args.model in STRUCTURED_OUTPUT_MODELS and response_format:
            # Get outputs that could be formatted into provided response_format
            # TODO: No usage logging for now
            responses = [c.message.parsed for c in outputs.choices if c.message.parsed]
            failures = [c.message.refusal for c in outputs.choices if not c.message.parsed]
            if len(responses) == 0:
                responses = [{"error": "Couldn't parse output into response_format", "failures": failures}]
        else:
            responses = [c.message.content.strip() for c in outputs.choices]
            try: cached_tokens = outputs.usage.prompt_tokens_details.cached_tokens
            except: cached_tokens = 0
            input_tokens = outputs.usage.prompt_tokens - cached_tokens
            output_tokens = outputs.usage.completion_tokens
            usage_track_file = os.path.join(USAGE_TRACKING_DIR, f"{self.args.model}_usage_generation.jsonl")
            with open(usage_track_file, "a") as f:
                f.write(json.dumps({'input': input_tokens, 'output': output_tokens}) + "\n")
            finish_reasons = [c.finish_reason for c in outputs.choices]

        return {
            "response_text": responses[0],
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reasons[0]
        }

    def interact(self, prompt, temperature=0, max_tokens=256, history=None, json_mode=False, response_format=None, **kwargs):
        if json_mode:
            if isinstance(prompt, str):
                outputs = self._json_generate(prompt, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._json_generate_from_messages(prompt, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
            if isinstance(prompt, str):
                outputs = self._structured_generate(prompt, temperature=temperature, max_tokens=max_tokens, response_format=response_format)
            elif isinstance(prompt, list):
                outputs = self._structured_generate_from_messages(prompt, temperature=temperature, max_tokens=max_tokens, response_format=response_format)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        else:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, temperature=temperature, max_tokens=max_tokens, history=history)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        responses = self._postprocess_output(outputs, response_format=response_format)

        return responses

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate(self, prompt, temperature=None, max_tokens=None, history=None):
        retries = 3
        while retries > 0:
            try:
                messages = []
                if history is not None:
                    assert len(history) % 2 == 0, "History must have an even number of messages"
                    for idx, msg in enumerate(history):
                        if idx % 2 == 0:
                            messages.append({"role": "user", "content": f"{msg}"})
                        else:
                            messages.append({"role": "assistant", "content": f"{msg}"})
                messages.append({"role": "user", "content": f"{prompt}"})
                if "gpt-5" in self.args.model:
                    completion = self.client.chat.completions.create(model=self.args.model,
                                                            messages=messages,
                                                            max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                else:
                    completion = self.client.chat.completions.create(model=self.args.model,
                                                            messages=messages,
                                                            temperature=self.args.temperature if temperature is None else temperature,
                                                            max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                return completion
            except BadRequestError as e:
                if "context_length_exceeded" in str(e) or "maximum context length" in str(e):
                    if retries > 1:
                        if "maximum context length is " in str(e) and "your messages resulted in " in str(e):
                            max_context_length = int(str(e).split("maximum context length is ")[1].split()[0])
                            prompt_length = int(str(e).split("your messages resulted in ")[1].split()[0])
                            temp_max_tokens = self.args.max_tokens if max_tokens is None else max_tokens
                            amount_to_truncate = int((prompt_length - max_context_length + temp_max_tokens + 10)/prompt_length * len(prompt))
                            prompt = prompt[amount_to_truncate:]
                        else:
                            prompt = prompt[int(len(prompt) * 0.2):]
                    else:
                        prompt = prompt[int(len(prompt) * 0.5):]
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")


    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_from_messages(self, messages, temperature=None, max_tokens=None):
        if "gpt-5" in self.args.model:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         messages=messages,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        else:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         messages=messages,
                                                         temperature=self.args.temperature if temperature is None else temperature,
                                                         max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        return completion


    def batch_interact(self, prompts, temperature=1, max_tokens=256, response_format=None):
        raise NotImplementedError


class AsyncConversationalGPTBaseAgent(ConversationalGPTBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        if self.api_info.get("api_type", "openai") == "openai":
            self.client = AsyncOpenAI(api_key=self.api_info.get("api_key", os.getenv('OPENAI_API_KEY')))
        elif self.api_info.get("api_type", "openai") == "azure":
            self.client = AsyncAzureOpenAI(api_key=self.api_info.get("api_key", os.getenv('AZURE_OPENAI_API_KEY')),
                                            azure_endpoint=self.api_info.get("api_base", os.getenv("AZURE_OPENAI_ENDPOINT")),
                                            api_version=self.api_info.get("api_version", "2024-06-01"))
        else:
            raise ValueError(f"API type {self.api_info.get('api_type')} not supported")

    async def batch_generate(self, prompts, temperature=0, max_tokens=256, response_format=None):
        model = self.args.model
        if model in STRUCTURED_OUTPUT_MODELS and response_format:
            completions = await asyncio.gather(*[self.client.beta.chat.completions.parse(model=self.args.model,
                                                                                         messages=[{"role": "user", "content": f"{prompt}"}],
                                                                                         temperature=temperature,
                                                                                         max_tokens=max_tokens,
                                                                                         response_format=response_format)
                                                for prompt in prompts])
        else:
            completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                     messages=[{"role": "user", "content": f"{prompt}"}],
                                                                                     temperature=temperature,
                                                                                     max_tokens=max_tokens)
                                                for prompt in prompts])
        return completions

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def batch_interact(self, prompts, temperature=0, max_tokens=256, response_format=None):
        outputs = asyncio.run(self.batch_generate(prompts, temperature, max_tokens, response_format))
        responses = [self._postprocess_output(output, response_format) for output in outputs]

        return responses

    def interact(self, prompt, temperature=0, max_tokens=256, response_format=None, **kwargs):
        outputs = self.batch_interact([prompt], temperature, max_tokens, response_format)

        return outputs[0]







class ConversationalGPTReasoningAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "o3-mini"
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _json_generate(
            self,
            prompt: str,
            reasoning_effort: Optional[str] = "high",
            max_tokens: Optional[int] = None
        ):
        completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=[
                                                            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                            {"role": "user", "content": f"{prompt}"}
                                                            ],
                                                         reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                         )
        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _json_generate_from_messages(
            self,
            messages: List[Dict[str, str]],
            reasoning_effort: Optional[str] = "high",
            max_tokens: Optional[int] = None
            ):
        completion = self.client.chat.completions.create(model=self.args.model,
                                                         response_format={ "type": "json_object" },
                                                         messages=messages,
                                                         reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _structured_generate(self, prompt, reasoning_effort="high", max_tokens=None, response_format=None):
        completion = self.client.beta.chat.completions.parse(model=self.args.model,
                                                             messages=[{"role": "user", "content": f"{prompt}"}],
                                                             reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                             max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                             response_format=response_format)

        return completion

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _structured_generate_from_messages(self, messages, reasoning_effort="high", max_tokens=None, response_format=None):
        completion = self.client.beta.chat.completions.parse(model=self.args.model,
                                                             messages=messages,
                                                             reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                             max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                             response_format=response_format)
        return completion

    def _postprocess_output(self, outputs, response_format=None):
        if self.args.model in STRUCTURED_OUTPUT_MODELS and response_format:
            # Get outputs that could be formatted into provided response_format
            # TODO: No usage logging for now
            responses = [c.message.parsed for c in outputs.choices if c.message.parsed]
            failures = [c.message.refusal for c in outputs.choices if not c.message.parsed]
            if len(responses) == 0:
                responses = [{"error": "Couldn't parse output into response_format", "failures": failures}]
        else:
            responses = [c.message.content.strip() for c in outputs.choices]
            cached_tokens = outputs.usage.prompt_tokens_details.cached_tokens
            input_tokens = outputs.usage.prompt_tokens - cached_tokens
            reasoning_tokens = outputs.usage.completion_tokens_details.reasoning_tokens
            output_tokens = outputs.usage.completion_tokens - reasoning_tokens
            usage_track_file = os.path.join(USAGE_TRACKING_DIR, f"{self.args.model}_usage_generation.jsonl")
            with open(usage_track_file, "a") as f:
                f.write(json.dumps({'input': input_tokens, 'output': output_tokens}) + "\n")
            finish_reasons = [c.finish_reason for c in outputs.choices]

        return {
            "response_text": responses[0],
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": finish_reasons[0]
        }

    def interact(self, prompt, reasoning_effort="high", max_tokens=256, history=None, json_mode=False, response_format=None, **kwargs):
        if json_mode:
            if isinstance(prompt, str):
                outputs = self._json_generate(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._json_generate_from_messages(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
            if isinstance(prompt, str):
                outputs = self._structured_generate(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens, response_format=response_format)
            elif isinstance(prompt, list):
                outputs = self._structured_generate_from_messages(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens, response_format=response_format)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        else:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens, history=history)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        responses = self._postprocess_output(outputs, response_format=response_format)

        return responses

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate(self, prompt, reasoning_effort="high", max_tokens=None, history=None):
        retries = 5
        while retries > 0:
            try:
                messages = []
                if history is not None:
                    assert len(history) % 2 == 0, "History must have an even number of messages"
                    for idx, msg in enumerate(history):
                        if idx % 2 == 0:
                            messages.append({"role": "user", "content": f"{msg}"})
                        else:
                            messages.append({"role": "assistant", "content": f"{msg}"})
                messages.append({"role": "user", "content": f"{prompt}"})
                if "o1-mini" in self.args.model:
                    completion = self.client.chat.completions.create(model=self.args.model,
                                                         messages=messages,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                else:
                    completion = self.client.chat.completions.create(model=self.args.model,
                                                                messages=messages,
                                                                reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                                max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)

                return completion
            except BadRequestError as e:
                if "context_length_exceeded" in str(e):
                    if retries > 1:
                        prompt = prompt[int(len(prompt) * 0.2):]
                    else:
                        prompt = prompt[int(len(prompt) * 0.4):]
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")


    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_from_messages(self, messages, reasoning_effort="high", max_tokens=None):
        if "o1-mini" in self.args.model:
            if len(messages) == 1:
                messages = [{"role": "user", "content": f"{messages[0]['content']}"}]
            else:
                messages[1] = {"role": "user", "content": f"{messages[0]['content']}\n\n{messages[1]['content']}"}
                messages.pop(0)
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         messages=messages,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        else:
            completion = self.client.chat.completions.create(model=self.args.model,
                                                         messages=messages,
                                                         reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                         max_completion_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
        return completion


    def batch_interact(self, prompts, reasoning_effort="high", max_tokens=256, response_format=None):
        raise NotImplementedError


class AsyncConversationalGPTReasoningAgent(ConversationalGPTReasoningAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        if self.api_info.get("api_type", "openai") == "openai":
            self.client = AsyncOpenAI(api_key=self.api_info.get("api_key", os.getenv('OPENAI_API_KEY')))
        elif self.api_info.get("api_type", "openai") == "azure":
            self.client = AsyncAzureOpenAI(api_key=self.api_info.get("api_key", os.getenv('AZURE_OPENAI_API_KEY')),
                                            azure_endpoint=self.api_info.get("api_base", os.getenv("AZURE_OPENAI_ENDPOINT")),
                                            api_version=self.api_info.get("api_version", "2024-06-01"))
        else:
            raise ValueError(f"API type {self.api_info.get('api_type')} not supported")

    async def batch_generate(self, prompts, reasoning_effort="high", max_tokens=256, response_format=None):
        model = self.args.model
        if model in STRUCTURED_OUTPUT_MODELS and response_format:
            completions = await asyncio.gather(*[self.client.beta.chat.completions.parse(model=self.args.model,
                                                                                         messages=[{"role": "user", "content": f"{prompt}"}],
                                                                                         reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                                                         max_completion_tokens=max_tokens,
                                                                                         response_format=response_format)
                                                for prompt in prompts])
        else:
            completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                     messages=[{"role": "user", "content": f"{prompt}"}],
                                                                                     reasoning_effort=self.args.reasoning_effort if reasoning_effort is None else reasoning_effort,
                                                                                     max_completion_tokens=max_tokens)
                                                for prompt in prompts])
        return completions

    # @backoff.on_exception(backoff.expo, OPENAI_RETRY_EXCEPTIONS) # Commented out - using LLMClient retry logic instead
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def batch_interact(self, prompts, reasoning_effort="high", max_tokens=256, response_format=None):
        outputs = asyncio.run(self.batch_generate(prompts, reasoning_effort=reasoning_effort, max_tokens=max_tokens, response_format=response_format))
        responses = [self._postprocess_output(output, response_format) for output in outputs]

        return responses

    def interact(self, prompt, reasoning_effort="high", max_tokens=256, response_format=None, **kwargs):
        outputs = self.batch_interact([prompt], reasoning_effort, max_tokens, response_format)

        return outputs[0]
