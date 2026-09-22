import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import BaseAgent

class HuggingFaceAgent(BaseAgent):
    def __init__(self, kwargs: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8 #kwargs['batch_size']

        print(f"Loading tokenizer for {kwargs['model']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model'], padding_side='left', truncation_side='left')

        print(f"Loading model for {kwargs['model']}...")
        # Load with memory-safe parameters
        self.model = AutoModelForCausalLM.from_pretrained(
            kwargs['model'],
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            low_cpu_mem_usage=True
        )

        # Only move to device if device_map wasn't used
        if self.device.type != 'cuda':
            self.model = self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token

    def preprocess_input(self, prompts):

        # case 1: list of messages:
        if isinstance(prompts, list) and isinstance(prompts[0], list) and isinstance(prompts[0][0], dict):
            messages = prompts
        # singular message:
        elif isinstance(prompts, list) and isinstance(prompts[0], dict):
            messages = [prompts]
        # list of prompts:
        elif isinstance(prompts, list) and isinstance(prompts[0], str):
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        # singular prompt:
        elif isinstance(prompts, str):
            messages = [[{"role": "user", "content": prompts}]]
        else:
            raise ValueError("Prompts must be a list of messages or a string")

        input_texts = [self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        return input_texts


    def _postprocess_output(self, outputs, inputs=None, json_mode=False):
        if inputs is not None:
            num_input_tokens = inputs["input_ids"].shape[-1]
            outputs = outputs[:, num_input_tokens:]
        else:
            num_input_tokens = 0

        num_output_tokens = outputs.shape[-1]
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return {
            "response_text": outputs_text,
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens,
            "finish_reason": "stop",
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    def interact(self, prompt, max_tokens=256, temperature=0, do_sample=True, json_mode=False, **kwargs):
        return self._generate(prompt, max_tokens, temperature, do_sample, json_mode)

    def _generate(self, prompt, max_tokens, temperature, do_sample, json_mode=False):
        input_text = self.preprocess_input(prompt)[0]
        inputs = self.tokenizer(input_text, add_special_tokens=False, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, pad_token_id=self.tokenizer.eos_token_id)
        return self._postprocess_output(outputs, inputs, json_mode)


    def batch_interact(self, prompts, max_tokens=256, do_sample=True):
        return self._generate_batch(prompts, max_tokens, do_sample)

    def _generate_batch(self, prompts, max_tokens, temperature, do_sample, json_mode=False):
        batch_input_text = self.preprocess_input(prompts)
        batch_inputs = self.tokenizer(batch_input_text, add_special_tokens=False, truncation=True, return_tensors="pt").to(self.device)
        batch_outputs = self.model.generate(**batch_inputs, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, pad_token_id=self.tokenizer.eos_token_id)
        return [self._postprocess_output(outputs, batch_inputs, json_mode) for outputs in batch_outputs]



    def batch_compute_likelihood(self, input_texts, target_data):
        """ Compute the log-likelihood of the target data given the input text. """
        # We should pad after concatenating with target_outputs
        prompts = [self.preprocess_input(text) for text in input_texts] # apply those chat-specific templates
        data_appended_prompt = [p + d for p, d in zip(prompts, target_data)] # append the target responses to the prompts
        encoded_texts = self.tokenizer(data_appended_prompt, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        encoded_data = self.tokenizer(target_data, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.device) # this is actually for getting the attention mask to know which part of the input is the response

        with torch.no_grad():
            outputs = self.model(**encoded_texts, return_dict=True)

        vocab_distribution = torch.log_softmax(outputs.logits, dim=-1)
        data_token_logprobs = torch.gather(vocab_distribution[:,:-1,:], 2, encoded_data.input_ids.unsqueeze(-1)[:,1:,:])
        true_data_token_logprobs = (data_token_logprobs * encoded_data.attention_mask.unsqueeze(-1)[:, 1:, :]).squeeze(-1) # get only the logprobs of the response tokens
        data_log_likelihood = true_data_token_logprobs.sum(dim=1) / encoded_data.attention_mask.sum(dim=1)

        return data_log_likelihood

    def compute_data_likelihood(self, input_text, target_datum):
        return self.batch_compute_likelihood([input_text], [target_datum])[0]


class OssAgent(HuggingFaceAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _postprocess_outputs(self, outputs, inputs=None, json_mode=False):
        num_input_tokens = inputs["input_ids"].shape[-1]
        outputs = outputs[:, num_input_tokens:]
        num_output_tokens = outputs.shape[-1]
        outputs_text = self.tokenizer.batch_decode(outputs)[0]

        # Parse the special GPT-oss response format
        reasoning_text = ""
        final_message = outputs_text

        # Look for analysis channel content
        analysis_start = outputs_text.find('<|channel|>analysis<|message|>')
        if analysis_start != -1:
            analysis_end = outputs_text.find('<|end|>', analysis_start)
            if analysis_end != -1:
                reasoning_text = outputs_text[analysis_start + 32:analysis_end]  # 32 is length of '<|channel|>analysis<|message|>'

        # Check if the response ends with <|return|> tag (before cleaning)
        finish_reason = "stop" if outputs_text.endswith('<|return|>') else "length"

        # Look for final message in assistant channel
        final_start = outputs_text.find('<|start|>assistant<|channel|>final<|message|>')
        if final_start != -1:
            final_end = outputs_text.find('<|end|>', final_start)
            if final_end != -1:
                # Normal case: found both start and end tags
                final_message = outputs_text[final_start + 42:final_end]  # 42 is length of '<|start|>assistant<|channel|>final<|message|>'
            else:
                # Edge case: found start tag but no end tag (response might be cut off)
                final_message = outputs_text[final_start + 42:]  # Take everything after the start tag

        # Clean the final message by removing any remaining tags
        if final_message:
            # Remove common GPT-oss tags that might remain
            final_message = final_message.replace('<|return|>', '').strip()

        # Calculate reasoning tokens
        reasoning_tokens = len(self.tokenizer.encode(reasoning_text)) if reasoning_text else 0

        return {
            "response_text": final_message,
            "reasoning_text": reasoning_text,
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens - reasoning_tokens,
            "finish_reason": finish_reason,
            "cached_tokens": 0,
            "reasoning_tokens": reasoning_tokens,
        }