import os
from types import SimpleNamespace

import boto3
import json
import time

from .base import AsyncBaseAgent
from pydantic import BaseModel
import yaml


class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]


# Models that support structured output
STRUCTURED_OUTPUT_MODELS = ["claude-3-5-sonnet-20241022-v1:0", "claude-3-5-haiku-20241007-v1:0", "claude-3-5-sonnet-20250219-v1:0"]


class AsyncClaudeAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", self.api_info.get("aws_access_key_id", ""))
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", self.api_info.get("aws_secret_access_key", ""))
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN", self.api_info.get("aws_session_token", ""))
        # if not aws_access_key_id: aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        # if not aws_secret_access_key: aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        # if not aws_session_token: aws_session_token = os.environ["AWS_SESSION_TOKEN"]

        from botocore.config import Config

        config = Config(read_timeout=1000)

        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=config,
        )

    def interact(self, prompt, temperature=0, max_tokens=256, history=None, json_mode=False, response_format=None, **kwargs):
        if json_mode:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")

        elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
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

    def _generate(self, prompt, json_mode=False, temperature=None, max_tokens=None, history=None, response_format=None):
        # Prepare messages

        retries = 3
        while retries > 0:
            try:
                messages = []
                # Add history if provided
                if history:
                    messages.extend(history[len(history)-3+retries:])

                # Add current prompt
                if isinstance(prompt, str):
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages.extend(prompt)

                # Prepare body for Bedrock API
                body = {
                    "messages": messages,
                    "anthropic_version": "bedrock-2023-05-31"
                }

                # Add parameters
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                elif hasattr(self.args, 'max_tokens'):
                    body["max_tokens"] = self.args.max_tokens

                if temperature is not None:
                    body["temperature"] = temperature
                elif hasattr(self.args, 'temperature'):
                    body["temperature"] = self.args.temperature

                # Handle JSON mode
                if json_mode:
                    body["response_format"] = {"type": "json_object"}

                # Handle structured output
                elif response_format:
                    body["response_format"] = response_format
                    # Note: Claude's structured output is handled differently than Gemini
                    # You might need to add the schema to the prompt or use a different approach

                # Generate content using Bedrock
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=f"arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.{self.args.model}"
                )
                return response
            except Exception as e:
                if "Input is too long" in str(e) or "exceed context limit" in str(e):
                    if retries > 1:
                        prompt = prompt[int(len(prompt) * 0.2):]
                    else:
                        prompt = prompt[int(len(prompt) * 0.5):]
                elif "please wait" in str(e):
                    time.sleep(20)
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")


    def _generate_from_messages(self, messages, json_mode=False, temperature=None, max_tokens=None, response_format=None):
        # Messages are already in the correct format for Claude
        return self._generate(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, response_format=response_format)

    def preprocess_input(self, text):
        return text

    def _postprocess_output(self, output, response_format=None):
        response_body = json.loads(output.get("body").read())
        content = response_body.get("content", [])

        # Extract the text from the content array
        response_text = ""
        if content and len(content) > 0:
            response_text = content[0].get("text", "")

        # Extract usage information
        usage = response_body.get("usage", {})

        return {
            "response_text": response_text,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "finish_reason": response_body.get("stop_reason", None),
            "cached_tokens": usage.get("cached_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
        }