import os
import re
from types import SimpleNamespace

# import google.generativeai as genai
from google import genai

from .base import AsyncBaseAgent
from pydantic import BaseModel
import yaml


class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]


# Models that support structured output
STRUCTURED_OUTPUT_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.0-pro"]


class AsyncGeminiAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # self.model = genai.GenerativeModel(self.args.model)
        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        api_key = self.api_info.get("api_key", "")
        if not api_key: api_key = os.environ["GOOGLE_API_KEY"]
        self.client = genai.Client(api_key=api_key)

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
        # Prepare generation config
        config = {}

        if max_tokens is not None:
            config['max_output_tokens'] = max_tokens
        elif hasattr(self.args, 'max_tokens'):
            config['max_output_tokens'] = self.args.max_tokens

        if temperature is not None:
            config['temperature'] = temperature
        elif hasattr(self.args, 'temperature'):
            config['temperature'] = self.args.temperature

        # Handle JSON mode
        if json_mode:
            config['response_mime_type'] = 'application/json'
        #     config['response_schema'] = list[Recipe]  # Default schema, can be made configurable

        # # Handle structured output
        elif response_format:
            config['response_mime_type'] = 'application/json'
        #     config['response_schema'] = response_format

        # Prepare contents
        if isinstance(prompt, str):
            contents = prompt
        else:
            contents = prompt

        # Generate content
        output = self.client.models.generate_content(
            model=self.args.model,
            contents=contents,
            config=config if config else None
        )

        return output

    def _generate_from_messages(self, messages, json_mode=False, temperature=None, max_tokens=None, response_format=None):
        # Convert messages to the format expected by Gemini
        contents = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get('role', 'user')
                if role == "system": role = "user"
                content = message.get('content', '')
                contents.append({
                    'role': role,
                    'parts': [{'text': content}]
                })
            else:
                contents.append({'parts': [{'text': str(message)}]})

        # Use the same generation logic as _generate
        return self._generate(contents, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, response_format=response_format)

    def preprocess_input(self, text):
        return text

    def _postprocess_output(self, output, response_format=None):
        if response_format == "json":
            output_text = re.sub(r"^```(?:json)?\n|\n```$", "", output.text.strip())
        else:
            output_text = output.text
        return {
            "response_text": output_text,
            "input_tokens": output.usage_metadata.prompt_token_count if hasattr(output, 'usage_metadata') and output.usage_metadata else 0,
            "output_tokens": output.usage_metadata.candidates_token_count if hasattr(output, 'usage_metadata') and output.usage_metadata else 0,
            "cached_tokens": output.usage_metadata.cached_content_token_count if hasattr(output, 'usage_metadata') and output.usage_metadata else 0,
            "reasoning_tokens": output.usage_metadata.thoughts_token_count if hasattr(output, 'usage_metadata') and output.usage_metadata else 0,
            "finish_reason": output.candidates[0].finish_reason if hasattr(output, 'candidates') and output.candidates[0].finish_reason else None,
        }