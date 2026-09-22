import os
import json
import yaml
import random
import argparse
import base64
import io
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import load_dataset
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
from PIL import Image
from llm.api_client import LLMClient

class PersonalizedBenchmarkGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark generator.

        Args:
            config: Configuration dictionary loaded from YAML file
        """

        # Initialize multiple LLM clients for diversity
        self.llm_clients = {}
        llm_configs = config.get("llm_configs", {})

        if llm_configs:
            # Initialize multiple clients
            for client_name, llm_config in llm_configs.items():
                if llm_config.get("model", "") == "" and os.path.exists(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml")):
                    with open(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml"), "r") as f:
                        api_info = yaml.safe_load(f)
                    llm_config["model"] = api_info.get(llm_config.get("model_kwargs", {}).get("api_account", "openai"), {}).get("model_name", "gpt-4o")
                print(f"Initializing {client_name} client with config: {llm_config}")
                self.llm_clients[client_name] = LLMClient(config=llm_config)

            # Set default client (for backward compatibility)
            self.client = list(self.llm_clients.values())[0] if self.llm_clients else None
        else:
            # Fallback to single client for backward compatibility
            llm_config = config.get("llm_config", {})
            if llm_config.get("model", "") == "" and os.path.exists(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml")):
                with open(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml"), "r") as f:
                    api_info = yaml.safe_load(f)
                llm_config["model"] = api_info.get(llm_config.get("model_kwargs", {}).get("api_account", "openai"), {}).get("model_name", "gpt-4o")
            print(f"llm_config: {llm_config}")
            self.client = LLMClient(config=llm_config)
            self.llm_clients = {"default": self.client}


        # Extract configuration values
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.dataset_split = config['dataset']['split']
        self.sample_size = config['dataset']['sample_size']
        self.problem_fields = config['dataset'].get('problem_fields', ['question', 'problem'])  # Default to common fields
        self.choices_field = config['dataset'].get('choices_field', None)  # Default to None
        self.answer_field = config['dataset'].get('answer_field', None)  # Default to None
        self.image_field = config['dataset'].get('image_field', None)  # Default to None
        self.num_personas_per_problem = config['personas']['per_problem']
        self.model_temperature = config.get('model', {}).get('temperature', 0.7)

        # Handle output directory and construct full paths
        self.output_dir = config['paths'].get('output_dir', 'data')
        self.output_file = os.path.join(self.output_dir, config['paths']['output_file'])
        self.all_prefs_filepath = config['paths']['all_prefs_filepath']
        self.dataset_preference_filepath = os.path.join(self.output_dir, config['paths']['dataset_preference_filepath'])
        self.persona_preferences_filepath = config['paths']['persona_preferences_filepath']

        # Handle persona library path - if it contains a directory separator, treat as relative to project root
        persona_lib_path = config['paths']['persona_library_filepath']
        if '/' in persona_lib_path or '\\' in persona_lib_path:
            # Path contains directory separator, treat as relative to project root
            self.persona_library_filepath = persona_lib_path
        else:
            # Simple filename, put in output directory
            self.persona_library_filepath = os.path.join(self.output_dir, persona_lib_path)
        self.persona_library = {}

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create persona library directory if it doesn't exist
        persona_lib_dir = os.path.dirname(self.persona_library_filepath)
        if persona_lib_dir:
            os.makedirs(persona_lib_dir, exist_ok=True)
        self.initial_num_personas = config['personas']['initial_num_personas']
        self.new_persona_prob = config['personas']['new_persona_prob']
        self.random_seed = config['system']['random_seed']

        # Preference generation configuration
        self.min_dimensions_per_problem = config['preference_generation']['min_dimensions_per_problem']
        self.consistency_weight = config['preference_generation']['consistency_weight']
        self.transferability_threshold = config['preference_generation']['transferability_threshold']

        # Evaluation configuration
        self.weight_tolerance = config['evaluation']['weight_tolerance']
        self.require_exact_preference_match = config['evaluation']['require_exact_preference_match']

        # Set random seed
        random.seed(self.random_seed)

        # Verbose mode (default to False, will be set by main script)
        self.verbose = False

        # Initialize credential refresh tracking for Claude clients (if enabled)
        self.reload_keys = config.get("reload_keys", False)
        if self.reload_keys:
            self.last_refresh_time = time.time()
            self.credential_script_path = config.get("credential_script_path", "")

        # Initialize timer for stopping after specified duration
        self.stop_after = config.get("stop_after", None)  # seconds
        self.start_time = time.time()

        # Parallelization settings
        self.num_workers = config.get("parallelization", {}).get("num_workers", 4)
        self.batch_dedup_size = config.get("parallelization", {}).get("batch_dedup_size", 20)

        # Thread-safe locks for shared state
        self._dimensions_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._progress_lock = threading.Lock()

        # Legacy API info handling (now handled in LLM client initialization)

        # if api_info.get("api_type", "openai") == "openai":
        #     self.client = OpenAI(api_key=api_info.get("api_key", os.getenv('OPENAI_API_KEY')))
        # elif api_info.get("api_type", "openai") == "azure":
        #     self.client = AzureOpenAI(api_key=api_info.get("api_key", os.getenv('AZURE_OPENAI_API_KEY')),
        #                               azure_endpoint=api_info.get("api_base", os.getenv("AZURE_OPENAI_ENDPOINT")),
        #                               api_version=api_info.get("api_version", "2024-06-01"))
        # else:
        #     raise ValueError(f"API type {api_info.get('api_type')} not supported")

        # if not self.client:
        #     raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

        # Load dataset
        print(f"Loading dataset {self.dataset_name}...")
        if self.dataset_name == "lighteval/mmlu":
            self.dataset = load_dataset("lighteval/mmlu", "all")
        else: self.dataset = load_dataset(self.dataset_name)

        # Sample problems
        self.problems = self._sample_problems()

        # Initialize storage
        self.all_dimensions = {}
        self.general_preferences = {}

        self.personalized_problems = []

        # Load or initialize persona preferences tracking
        self.persona_preferences_data = self._load_persona_preferences_data()

    def set_verbose(self, verbose: bool):
        """Set verbose mode for detailed output."""
        self.verbose = verbose

    def _sample_problems(self) -> List[Dict]:
        """Sample problems from the dataset."""
        if self.dataset_split not in self.dataset:
            raise ValueError(f"Split {self.dataset_split} not found in dataset")

        dataset_size = len(self.dataset[self.dataset_split])
        sample_size = min(self.sample_size, dataset_size)
        indices = random.sample(range(dataset_size), sample_size)

        return [self.dataset[self.dataset_split][i] for i in indices]

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load PIL Image from file path."""
        if os.path.isabs(image_path):
            # Absolute path
            full_path = image_path
        else:
            # Relative path - make it relative to output directory
            full_path = os.path.join(self.output_dir, image_path)

        return Image.open(full_path)

    def _encode_image_to_base64(self, image: Union[Image.Image, str]) -> str:
        """Convert PIL Image or file path to base64 string for API calls."""
        if isinstance(image, str):
            # If it's a file path, load the image first
            image = self._load_image_from_path(image)

        if hasattr(image, 'save'):
            # If it's a PIL Image
            buffered = io.BytesIO()
            # Convert to RGB if necessary (handles RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _should_stop_generation(self) -> bool:
        """Check if generation should stop due to time limit."""
        if self.stop_after is None:
            return False
        return (time.time() - self.start_time) >= self.stop_after

    def _refresh_claude_credentials_if_needed(self):
        """Refresh Claude credentials every hour if reload_keys is enabled."""
        if not self.reload_keys or (time.time() - self.last_refresh_time) < 3600:
            return

        try:
            import subprocess
            if os.path.exists(self.credential_script_path):
                # Execute credential script and capture environment variables
                result = subprocess.run(['bash', '-c', f'source {self.credential_script_path} && env'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse environment variables from output
                    for line in result.stdout.strip().split('\n'):
                        if '=' in line and line.startswith('AWS_'):
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                    # Re-initialize Claude clients with new environment variables
                    for client_name, llm_config in self.config.get("llm_configs", {}).items():
                        api_account = llm_config.get("model_kwargs", {}).get("api_account", "")
                        if "claude" in api_account.lower():
                            self.llm_clients[client_name] = LLMClient(config=llm_config)

                    self.last_refresh_time = time.time()
                    logging.info("Claude credentials refreshed successfully.")
                else:
                    logging.error(f"Credential script failed: {result.stderr}")
                    self.last_refresh_time = time.time()
        except Exception as e:
            logging.error(f"Failed to refresh Claude credentials: {e}")
            self.last_refresh_time = time.time()

    def _get_random_llm_client(self) -> LLMClient:
        """Randomly select an LLM client for diversity."""
        # Check if Claude credentials need refresh (only if reload_keys is enabled)
        if self.reload_keys:
            self._refresh_claude_credentials_if_needed()

        if len(self.llm_clients) > 1:
            client_name = random.choice(list(self.llm_clients.keys()))
            if self.verbose:
                print(f"Using {client_name} client for this API call")
            return self.llm_clients[client_name]
        else:
            return self.client

    def _call_llm(self, prompt: str, temperature: Optional[float] = None, image: Optional[Union[Image.Image, str]] = None, use_random_client: bool = True) -> Dict:
        """Call the LLM with a prompt and optional image, return parsed JSON."""
        if temperature is None:
            temperature = self.model_temperature

        # Select client - either random for diversity or default
        client = self._get_random_llm_client() if use_random_client else self.client

        try:
            # Prepare messages based on whether image is provided
            if image is not None:
                # Encode image to base64
                base64_image = self._encode_image_to_base64(image)

                # Construct multimodal message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that outputs clean, valid, parsable JSON. When analyzing images, describe what you see that's relevant to the question."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ]
            else:
                # Text-only message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that outputs clean, valid, parsable JSON."},
                    {"role": "user", "content": prompt}
                ]

            # try:
            response = client.chat(
                    messages=messages,
                    temperature=temperature,
                    response_format="json"
                )
            # except:
            #     breakpoint()
            #     response = client.chat(
            #         messages=messages,
            #         temperature=temperature,
            #         response_format="json"
            #     )

            # Parse JSON response
            response_text = response["response_text"]
            if isinstance(response_text, str):
                try:
                    parsed_response = json.loads(response_text)
                    # Ensure we return a dictionary
                    if isinstance(parsed_response, dict):
                        return parsed_response
                    else:
                        print(f"Warning: LLM returned non-dict JSON: {type(parsed_response)}")
                        print(f"Raw response: {response_text[:500]}...")
                        return {}
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response: {response_text[:500]}...")
                    return {}
            elif isinstance(response_text, dict):
                return response_text
            else:
                print(f"Warning: LLM returned unexpected response type: {type(response_text)}")
                print(f"Raw response: {str(response_text)[:500]}...")
                return {}
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"Prompt: {prompt}")
            if image is not None:
                print(f"Image provided: {type(image)}")
            return {}


    def load_all_preferences(self, filepath: Optional[str] = None) -> Dict[str, Dict]:
        """Load all preferences from a file."""
        if filepath and os.path.exists(filepath):
            # Check if file is empty
            if os.path.getsize(filepath) == 0:
                print(f"Warning: {filepath} is empty, starting with empty preferences")
                self.all_dimensions = {}
                return

            print(f"Loading all preferences from {filepath}")
            try:
                if filepath.endswith(".json"):
                    with open(filepath, 'r') as f:
                        self.all_dimensions = json.load(f)
                elif filepath.endswith(".jsonl"):
                    with open(filepath, 'r') as f:
                        self.all_dimensions = {json.loads(line)["name"]: json.loads(line) for line in f if line.strip()}
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {filepath}: {e}")
                print("Starting with empty preferences")
                self.all_dimensions = {}

    def save_all_dimensions(self, filepath: Optional[str] = None) -> None:
        """Save all dimensions to a file."""
        if filepath is None:
            filepath = self.all_prefs_filepath
        if filepath.endswith(".json"):
            with open(filepath, "w+") as f:
                json.dump(self.all_dimensions, f, indent=4)
        elif filepath.endswith(".jsonl"):
            with open(filepath, "w+") as f:
                for dim_name, dim_info in self.all_dimensions.items():
                    f.write(json.dumps({"name": dim_name, **dim_info}) + "\n")
        if self.verbose: print(f"Saved all dimensions to {filepath}")


    def load_dataset_preferences(self, filepath: Optional[str] = None) -> Dict[str, Dict]:
        """
        Load hardcoded dataset-level preference dimensions or generate them if not provided.
        """
        if filepath and os.path.exists(filepath):
            print(f"Loading dataset preferences from {filepath}")
            with open(filepath, 'r') as f:
                dimensions_data = json.load(f)
                self.general_preferences = {dim["name"]: dim for dim in dimensions_data.get("dimensions", [])}
        else:
            print("Generating dataset-level preferences...")
            # Select random examples to include in the prompt
            example_indices = random.sample(range(len(self.problems)), min(5, len(self.problems)))
            examples = [self.problems[i] for i in example_indices]

            # Determine task type and check for multimodal content
            has_images = any('image' in ex and ex['image'] is not None for ex in examples)
            multimodal_note = "\nNOTE: This dataset contains multimodal problems with both text and images. Consider preferences that would apply to explaining visual content and text-image relationships." if has_images else ""

            if 'problem' in examples[0]:  # Math-like
                example_texts = [ex.get('problem', '') for ex in examples]
                domain = "math problems"
            elif 'question' in examples[0]:  # QA-like
                example_texts = [ex.get('question', '') for ex in examples]
                domain = "questions"
            else:  # Generic
                example_texts = [str(ex) for ex in examples]
                domain = "tasks"

            prompt = (
                f"You are an expert in personalization systems and educational psychology. Your task is to identify the key dimensions for personalizing AI responses to {domain} from the dataset '{self.dataset_name}'.{multimodal_note}\n\n"
                f"DATASET ANALYSIS:\n"
                f"Here are representative examples from this dataset:\n\n"
                f"Example 1: {example_texts[0][:200]}{'...' if len(example_texts[0]) > 200 else ''}\n"
                f"Example 2: {example_texts[1][:200]}{'...' if len(example_texts[1]) > 200 else ''}\n"
                f"Example 3: {example_texts[2][:200] if len(example_texts) > 2 else 'N/A'}{'...' if len(example_texts) > 2 and len(example_texts[2]) > 200 else ''}\n\n"
                f"TASK: Generate exactly 4-5 GENERAL personalization dimensions that apply across ALL problems in this dataset.\n\n"
                f"REQUIREMENTS:\n"
                f"1. **Universal Applicability**: Each dimension must be relevant to ANY problem in the dataset\n"
                f"2. **High Impact**: Focus on dimensions that significantly affect user experience and learning\n"
                f"3. **Clear Differentiation**: Ensure minimal overlap between dimensions\n"
                f"4. **Actionable**: Each dimension should guide how responses are generated\n\n"
                f"DIMENSION SPECIFICATIONS:\n"
                f"- **Name**: Concise, descriptive (2-4 words)\n"
                f"- **Description**: Clear explanation of what this dimension captures and why it matters\n"
                f"- **Value Range**: Either 1-5 scale with clear anchors OR specific categorical values\n"
                f"- **Global Importance**: 1-5 rating of how critical this dimension is generally\n\n"
                f"EXAMPLES:\n"
                f"For Math Problems:\n"
                f"{{\n"
                f'  "name": "Explanation Depth",\n'
                f'  "description": "The level of detail preferred in explanations, from concise overviews to comprehensive step-by-step breakdowns with reasoning.",\n'
                f'  "value_range": "1-5 (1: Very concise, 3: Balanced, 5: Extremely detailed)",\n'
                f'  "global_importance": 5\n'
                f"}}\n\n"
                f"For Any Domain:\n"
                f"{{\n"
                f'  "name": "Communication Style",\n'
                f'  "description": "The formality and tone of communication, from academic and formal to casual and conversational.",\n'
                f'  "value_range": "1-5 (1: Very formal/academic, 3: Professional, 5: Very casual/friendly)",\n'
                f'  "global_importance": 4\n'
                f"}}\n\n"
                f"CRITICAL: Generate ONLY general dimensions that apply to ALL problems in this dataset, not problem-specific preferences.\n\n"
                f"OUTPUT FORMAT (JSON):\n"
                f"{{\n"
                f'    "dimensions": [\n'
                f'        {{\n'
                f'            "name": "Dimension Name",\n'
                f'            "description": "Comprehensive description of what this dimension represents and its impact",\n'
                f'            "value_range": "Clear scale or categorical values with specific anchors",\n'
                f'            "global_importance": 1-5\n'
                f'        }}\n'
                f'    ]\n'
                f"}}\n"
            )

            dimensions_data = self._call_llm(prompt)
            self.general_preferences = {dim["name"]: dim for dim in dimensions_data.get("dimensions", [])}

            for dim in dimensions_data.get("dimensions", []):
                print(f"    - '{dim['name']}': {dim['description']} (Range: {dim['value_range']}) (Importance: {dim['global_importance']})")
            print()

            # Save to file if filepath is provided
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(dimensions_data, f, indent=2)

        # Add to all_dimensions
        self.all_dimensions.update(self.general_preferences)
        self.save_all_dimensions()
        print(f"Loaded {len(self.general_preferences)} general dataset-level preferences")


    def generate_persona_preferences_for_problem(self, problem: Dict, persona: Dict, problem_index: int, max_tries: int = 3) -> Tuple[Dict[str, Dict], bool]:
        """Generate personalized preferences for how this specific persona would want this problem explained."""

        # Extract problem content including potential image
        problem_text, domain, image = self._extract_problem_content(problem)

        # Step 1: Sample preference dimensions from the persona's problem
        raw_dimensions, success1 = self._sample_pref_dimensions_from_persona_problem(
            persona, problem_index, problem_text, domain, image, try_idx=0, max_tries=max_tries
        )

        if not success1 or not raw_dimensions:
            return {}, False

        # Step 2: Deduplicate dimensions using update_preferences
        unique_dimensions, success2 = self.update_preferences(raw_dimensions)
        if not success2 or not unique_dimensions:
            return {}, False

        self.save_all_dimensions()

        # Step 3: Instantiate persona preferences for this problem
        persona_preferences, success3 = self._instantiate_persona_preferences(
            unique_dimensions, persona, problem_index, problem_text, domain, try_idx=0, max_tries=max_tries
        )

        if not success3 or not persona_preferences:
            return {}, False

        return persona_preferences, True


    def _instantiate_persona_preferences(self, unique_dimensions: Dict[str, Dict], persona: Dict, problem_index: int, problem_text: str, domain: str, try_idx: int = 0, max_tries: int = 3) -> Tuple[Dict[str, Dict], bool]:
        """Instantiate persona preferences for a problem."""
        # Step 3: Get persona's existing preferences from previous problems
        existing_preferences = persona.get("accumulated_preferences", {})

        if try_idx >= max_tries:
            print(f"Error: Failed to instantiate persona preferences for {persona['name']} on problem {problem_index} after {max_tries} tries")
            return {}, False

        # Step 4: Generate persona-specific preference values
        preferences_prompt = (
            f"You are an expert in personalization systems and educational psychology. Given a persona and relevant preference dimensions, generate this persona's specific preference values.\n\n"
            f"PERSONA SUMMARY:\n"
            f"Name: {persona['name']}\n"
            f"Background: {persona.get('minimal_necessary_description', 'No description')}\n"
            f"Key Learning Characteristics:\n"
            f"- Cognitive Style: {persona.get('educational_profile', {}).get('cognitive_features', {}).get('learning_style', 'Not specified')}\n"
            f"- Problem Solving: {persona.get('educational_profile', {}).get('cognitive_features', {}).get('problem_solving', 'Not specified')}\n"
            f"- Confidence: {persona.get('educational_profile', {}).get('affective_features', {}).get('confidence', 'Not specified')}\n"
            f"- Personality Traits: Openness={persona.get('personality_big5', {}).get('openness', '?')}, Conscientiousness={persona.get('personality_big5', {}).get('conscientiousness', '?')}, Extraversion={persona.get('personality_big5', {}).get('extraversion', '?')}, Agreeableness={persona.get('personality_big5', {}).get('agreeableness', '?')}, Neuroticism={persona.get('personality_big5', {}).get('neuroticism', '?')}\n"
            f"- Domain Expertise: {', '.join(persona.get('domain_expertise', []))}\n\n"
            f"CURRENT PROBLEM:\n{problem_text}\n\n"
            f"RELEVANT PREFERENCE DIMENSIONS:\n"
        )

        for dim_name, dim_info in unique_dimensions.items():
            preferences_prompt += f"- {dim_name}: {dim_info['description']} (Range: {dim_info['value_range']})\n"

        preferences_prompt += f"\n"

        if existing_preferences:
            preferences_prompt += f"PERSONA'S EXISTING PREFERENCES FROM PREVIOUS PROBLEMS:\n"
            for pref_name, pref_data in existing_preferences.items():
                problem_context = pref_data.get('problem_context', 'Unknown problem type')
                preferences_prompt += f"- {pref_name}: {pref_data.get('value', 'Unknown')} (importance: {pref_data.get('local_importance', 'Unknown')}) [from {problem_context}]\n"
            preferences_prompt += f"\n"

        preferences_prompt += (
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. **Consistency**: If this persona has existing preferences for similar dimensions, maintain consistency with their established patterns\n"
            f"2. **Transferability**: Consider how preferences might transfer between problem types. For example:\n"
            f"   - Visual preferences might transfer well between geometry and coding\n"
            f"   - Communication tone preferences should be highly consistent across problems\n"
            f"   - Detail level preferences might vary based on problem complexity\n"
            f"3. **Persona-Specific**: Ground all preferences in this persona's characteristics and background\n"
            f"4. **Problem-Relevant**: Adjust importance and some values based on this specific problem type\n"
            f"5. **Justification**: Explain how you considered existing preferences and problem context\n\n"
            f"For each dimension, provide:\n"
            f"1. **Value**: Specific value that fits this persona's profile and is consistent with existing preferences\n"
            f"2. **Local Importance**: 1-5 scale of how much this persona cares about this aspect for THIS specific problem. Make sure to spread out the range of importance; fpr example you cannot say all dimensions are important.\n"
            f"3. **Justification**: Detailed explanation of your reasoning, including reference to existing preferences if relevant\n\n"
            f"Format as JSON:\n"
            f"{{\n"
            f'    "preferences": {{\n'
            f'        "Dimension Name": {{\n'
            f'            "name": " Copy the name of the preference dimension from the previous step",\n'
            f'            "description": "Copy the description of the preference dimension from the previous step",\n'
            f'            "value_range": "Copy the value range of the preference dimension from the previous step",\n'
            f'            "value": "specific value or number",\n'
            f'            "local_importance": 1-5,\n'
            f'            "justification": "Detailed reasoning considering persona characteristics, existing preferences, and problem context"\n'
            f'            "type": "expertise" or "personal"\n'
            f'        }}\n'
            f'    }}\n'
            f"}}\n"
        )

        preferences_data = self._call_llm(preferences_prompt)
        persona_preferences = preferences_data.get("preferences", {})

        # Validate preferences
        if not persona_preferences:
            print("Error: No preferences generated, trying again...")
            return self._instantiate_persona_preferences(unique_dimensions, persona, problem_index, problem_text, domain, try_idx + 1, max_tries)

        # Check for unknown dimensions
        unknown_dims = [k for k in persona_preferences.keys() if k not in unique_dimensions]
        if unknown_dims:
            print(f"Error: Instantiated preferences contain unknown dimensions: {unknown_dims}, trying again...")
            return self._instantiate_persona_preferences(unique_dimensions, persona, problem_index, problem_text, domain, try_idx + 1, max_tries)

        # Check for missing dimensions
        missing_dims = [k for k in unique_dimensions.keys() if k not in persona_preferences]
        if missing_dims:
            print(f"Error: Missing preferences for dimensions: {missing_dims}, trying again...")
            return self._instantiate_persona_preferences(unique_dimensions, persona, problem_index, problem_text, domain, try_idx + 1, max_tries)

        # Validate preference structure
        for pref_name, pref_data in persona_preferences.items():
            required_fields = ["value", "local_importance", "justification"]
            missing_fields = [field for field in required_fields if field not in pref_data or pref_data[field] is None]
            if missing_fields:
                print(f"Error: Preference '{pref_name}' missing fields: {missing_fields}, trying again...")
                return self._instantiate_persona_preferences(unique_dimensions, persona, problem_index, problem_text, domain, try_idx + 1, max_tries)

        # Update persona's accumulated preferences
        if "accumulated_preferences" not in persona:
            persona["accumulated_preferences"] = {}

        for pref_name, pref_data in persona_preferences.items():
            pref_data["problem_context"] = domain
            persona["accumulated_preferences"][pref_name] = pref_data

        if self.verbose: print(f"Generated {len(persona_preferences)} personalized preferences for {persona['name']} on problem {problem_index}")
        if self.verbose:
            for pref_name, pref_data in persona_preferences.items():
                print(f"    - '{pref_name}': {pref_data.get('value', 'N/A')} (importance: {pref_data.get('local_importance', 'N/A')})")
            print()
        return persona_preferences, True


    def _sample_pref_dimensions_from_persona_problem(self, persona: Dict, problem_index: int, problem_text: str, domain: str, image: Optional[Union[Image.Image, str]] = None, try_idx: int = 0, max_tries: int = 3) -> Tuple[Dict[str, Dict], bool]:
        """Sample preference dimensions from the persona's problem."""
        if try_idx >= max_tries:
            print(f"Error: Failed to sample preference dimensions after {max_tries} tries")
            return {}, False

                 # Step 1: Generate persona-specific preference dimensions for this problem
        # Add image context to prompt if available
        image_context = ""
        if image is not None:
            image_context = f"\nNOTE: This problem includes an image. Please consider how the visual content affects learning preferences and what personalization dimensions would be relevant for explaining content that involves both text and visual elements.\n"

        # Check if this is a medical caretaker scenario
        is_medical_caretaker = "medical consultation as a patient caretaker" in domain

        if is_medical_caretaker:
            dimensions_prompt = (
                f"Generate {self.min_dimensions_per_problem} preference dimensions for how a patient caretaker/family member would want medical information explained to them.\n\n"
                f"MEDICAL CARETAKER SCENARIO: {problem_text}{image_context}\n\n"
                f"REQUIRED DIMENSION TYPES FOR MEDICAL CARETAKER COMMUNICATION:\n"
                f"1. **Medical Knowledge/Comfort Dimensions (2-3 dimensions)**: Identify the key medical concepts involved and create dimensions about the caretaker's comfort level with medical terminology and concepts. Examples:\n"
                f"   - 'Comfort with Medical Terminology' (how much medical jargon they can handle)\n"
                f"   - 'Familiarity with Specific Condition' (their background knowledge about the patient's condition)\n"
                f"   - 'Understanding of Treatment Options' (their experience with medical procedures)\n"
                f"   Only include medical knowledge dimensions that are ACTUALLY relevant to this specific medical question.\n\n"
                f"2. **Caretaker-Specific Personal Dimensions (remaining dimensions)**: Focus on how a worried family member or caretaker would want to receive and understand medical information. Examples:\n"
                f"   - 'Emotional Support Needs' (how much reassurance and emotional context they need)\n"
                f"   - 'Decision-Making Role' (whether they're the primary decision maker or support person)\n"
                f"   - 'Communication with Patient' (how they plan to relay information to the patient)\n"
                f"   - 'Practical Care Concerns' (focus on day-to-day caregiving implications)\n\n"
            )
        else:
            dimensions_prompt = (
                f"Generate {self.min_dimensions_per_problem} preference dimensions for how someone would want this {domain} explained.\n\n"
                f"PROBLEM: {problem_text}{image_context}\n\n"
                f"REQUIRED DIMENSION TYPES:\n"
                f"1. **Expertise/Comfort Dimensions (2-3 dimensions)**: Identify the key skills/concepts needed for this specific problem and create dimensions about comfort level with those skills. Examples:\n"
                f"   - 'Comfort with Recursive Thinking' (for recursion problems)\n"
                f"   - 'Familiarity with Inequality Manipulation' (for inequality problems)\n"
                f"   - 'Experience with Algorithm Complexity' (for efficiency problems)\n"
                f"   Only include expertise dimensions for skills that are ACTUALLY needed for this problem.\n\n"
                f"2. **Personal Dimensions (remaining dimensions)**: Think beyond generic categories. Focus on authentic, personal aspects that would make someone think 'this person really gets how I need to learn.'\n\n"
            )

        # Add common sections for both medical and non-medical scenarios
        dimensions_prompt += (
                f"CREATIVE REFLECTION PROCESS:\n"
                f"- What unique ways might someone want information delivered based on their life experiences?\n"
                f"- How might someone's profession, personality, or background create unusual learning preferences?\n"
                f"- What would make someone think 'Yes, this explanation style is perfect for me'?\n\n"
                f"Each dimension should be:\n"
                f"- Specific and actionable (something that could guide how to explain)\n"
                f"- Personally meaningful (something someone would actually care about)\n"
                f"- Grounded in authentic human needs and experiences\n\n"
            )

        dimensions_prompt += (
            f"NOW, generate these dimensions from the perspective of this specific person:\n\n"
            f"PERSONA PROFILE:\n"
            f"Name: {persona['name']}\n"
            f"Summary: {persona.get('minimal_necessary_description', '')}\n"
            f"Age: {persona.get('demographics', {}).get('age', '?')}, Occupation: {persona.get('demographics', {}).get('occupation', 'Unknown')}\n"
            f"Location: {persona.get('demographics', {}).get('location', 'Unknown')}\n"
            f"Hobbies: {', '.join(persona.get('demographics', {}).get('hobbies', []))}\n"
            f"Domain expertise: {', '.join(persona.get('domain_expertise', []))}\n\n"
            f"LEARNING PROFILE:\n"
            f"- Knowledge level: {persona.get('educational_profile', {}).get('knowledge_level', 'Not specified')}\n"
            f"- Learning style: {persona.get('educational_profile', {}).get('cognitive_features', {}).get('learning_style', 'Not specified')}\n"
            f"- Problem-solving: {persona.get('educational_profile', {}).get('cognitive_features', {}).get('problem_solving', 'Not specified')}\n"
            f"- Confidence: {persona.get('educational_profile', {}).get('affective_features', {}).get('confidence', 'Not specified')}\n"
            f"- Anxiety triggers: {persona.get('educational_profile', {}).get('affective_features', {}).get('anxiety_triggers', 'Not specified')}\n"
            f"- Motivation: {persona.get('educational_profile', {}).get('affective_features', {}).get('motivation', 'Not specified')}\n\n"
            f"PERSONALITY:\n"
            f"- Openness: {persona.get('personality_big5', {}).get('openness', {}).get('description', 'Not specified') if isinstance(persona.get('personality_big5', {}).get('openness'), dict) else 'Not specified'}\n"
            f"- Conscientiousness: {persona.get('personality_big5', {}).get('conscientiousness', {}).get('description', 'Not specified') if isinstance(persona.get('personality_big5', {}).get('conscientiousness'), dict) else 'Not specified'}\n"
            f"- Social style: {persona.get('personality_big5', {}).get('extraversion', {}).get('description', 'Not specified') if isinstance(persona.get('personality_big5', {}).get('extraversion'), dict) else 'Not specified'}\n\n"
            f"BACKSTORY: {persona.get('backstory', 'No backstory provided')}\n\n"
        )

        if is_medical_caretaker:
            dimensions_prompt += (
                f"MEDICAL CARETAKER INSTRUCTIONS:\n"
                f"1. First, analyze the medical question to identify 2-3 key medical concepts or knowledge areas\n"
                f"2. Create medical knowledge dimensions for those specific areas based on this person's background and comfort with medical information\n"
                f"3. Then create caretaker-specific personal dimensions based on their unique characteristics as someone caring for a patient\n"
                f"  - Based on this person's background, personality, and relationship to healthcare, how would they want medical information explained?\n"
                f"  - Generate dimensions that capture the nuanced, personal ways THIS person would want medical information communicated as a caretaker.\n"
                f"  - Consider their emotional needs, decision-making role, and how they'll use this information to care for their loved one.\n\n"
                f"4. Ensure the total is exactly {self.min_dimensions_per_problem} dimensions\n\n"
            )
        else:
            dimensions_prompt += (
                f"INSTRUCTIONS:\n"
                f"1. First, analyze the problem to identify 2-3 key skills/concepts needed to solve it\n"
                f"2. Create expertise dimensions for those specific skills based on this person's background\n"
                f"3. Then create personal dimensions based on their unique characteristics\n"
                f"  - Based on this person's unique background, personality, experiences, and current situation, what specific aspects of an explanation would matter most to THEM? "
                f"  - Generate dimensions that capture the nuanced, personal ways THIS person would want the {domain} explained. "
                f"  - Think about how their professional experience, hobbies, personality, learning challenges, and life story create unique preferences that wouldn't apply to everyone.\n\n"
                f"4. Ensure the total is exactly {self.min_dimensions_per_problem} dimensions\n\n"
            )
        dimensions_prompt += (
            f"NAMING REQUIREMENTS:\n"
            f"- Use concise, direct names (e.g., 'Visual Learning Style', 'Technical Depth')\n"
            f"- DO NOT use prefixes like 'Preference for', 'Desire for', 'Need for'\n"
            f"- DO NOT use redundant words like 'Preference', 'Style', 'Approach' unless essential\n"
            f"- Make names specific and actionable (what the preference controls)\n\n"
            f"Format your response as a clean JSON object with exactly the following structure and nothing else:\n"
            f"{{\n"
            f'    "dimensions": [\n'
            f'        {{\n'
            f'            "name": "(example) Technical Depth",\n'
            f'            "description": "Why this matters and how it affects learning experience",\n'
            f'            "value_range": "1-5 scale or categorical options",\n'
            f'            "type": "expertise" or "personal"\n'
            f'        }}\n'
            f'    ]\n'
            f"}}\n\n"
            f"Generate dimensions that capture how THIS specific person would want this problem explained, considering both their technical comfort level and their unique personal learning needs.\n\n"
                f"PROBLEM: {problem_text}\n\n"
                f"Preferences:"
        )
        dimensions_data = self._call_llm(dimensions_prompt, image=image)
        raw_dimensions = {dim["name"]: dim for dim in dimensions_data.get("dimensions", [])}

        # Validate dimensions structure and content
        if not raw_dimensions:
            print(f"Error: No dimensions generated, trying again...")
            return self._sample_pref_dimensions_from_persona_problem(persona, problem_index, problem_text, domain, image, try_idx + 1, max_tries)

        # Check required fields
        missing_fields = []
        for dim_name, dim_data in raw_dimensions.items():
            required_fields = ["name", "description", "value_range", "type"]
            for field in required_fields:
                if field not in dim_data or not dim_data[field]:
                    missing_fields.append(f"{dim_name}.{field}")

        if missing_fields:
            print(f"Error: Missing required fields: {missing_fields}, trying again...")
            return self._sample_pref_dimensions_from_persona_problem(persona, problem_index, problem_text, domain, image, try_idx + 1, max_tries)

        # Check if we have the right number of dimensions
        if len(raw_dimensions) != self.min_dimensions_per_problem:
            print(f"Error: Expected {self.min_dimensions_per_problem} dimensions, got {len(raw_dimensions)}, trying again...")
            return self._sample_pref_dimensions_from_persona_problem(persona, problem_index, problem_text, domain, image, try_idx + 1, max_tries)

        return raw_dimensions, True


    def update_preferences(self, new_dimensions: Dict[str, Dict]) -> Tuple[Dict[str, Dict], bool]:
        """
        Check for semantic overlaps with existing preferences and rename if needed.
        Uses chunking to handle large numbers of existing dimensions.
        returns a dict of the user's relevant preferences for this problem with updated names
        """
        if not new_dimensions:
            return {}, True

        if len(self.all_dimensions) == 0:
            # If no existing dimensions, just add all new ones
            self.all_dimensions = new_dimensions
            return new_dimensions, True

        # Filter out new dimensions that already exist (exact name match)
        truly_new_dimensions = {k: v for k, v in new_dimensions.items() if k not in self.all_dimensions}

        if not truly_new_dimensions:
            # All new dimensions already exist, just return the existing ones
            return {k: self.all_dimensions[k] for k in new_dimensions.keys()}, True

        if self.verbose:
            print(f"Updating preferences with {len(truly_new_dimensions)} truly new dimensions and {len(self.all_dimensions)} existing dimensions")
            print(f"  - New dimensions: {sorted(list(truly_new_dimensions.keys()))}")

        # Chunk existing dimensions to avoid overwhelming the LLM
        chunk_size = 200  # Process 200 existing dimensions at a time
        existing_items = list(self.all_dimensions.items())

        updated_dimensions = {k: self.all_dimensions[k] for k in new_dimensions.keys() if k in self.all_dimensions}
        updated_dimensions.update(self.general_preferences)
        novel_dimensions = []
        remaining_new_dimensions = dict(truly_new_dimensions)

        # Process all new dimensions against existing dimensions in chunks efficiently
        # Check each chunk against ALL remaining new dimensions in a single API call
        for i in range(0, len(existing_items), chunk_size):
            if not remaining_new_dimensions:  # No more dimensions to check
                break

            chunk = dict(existing_items[i:i + chunk_size])

            if self.verbose and len(existing_items) > chunk_size:
                print(f"  Checking {len(remaining_new_dimensions)} new dimensions against chunk {i//chunk_size + 1}/{(len(existing_items) + chunk_size - 1)//chunk_size}")

            # Check ALL remaining new dimensions against this chunk in one API call
            mapping_results = self._check_dimensions_against_chunk(remaining_new_dimensions, chunk)

            # Process the results
            for new_name, result in mapping_results.items():
                if result and result.get("action") == "map_to_existing":
                    existing_name = result.get("existing_dimension")
                    if existing_name in self.all_dimensions:
                        if self.verbose:
                            print(f"    - '{new_name}': --> '{existing_name}'. {result.get('reason', '')}")
                        updated_dimensions[existing_name] = self.all_dimensions[existing_name]
                        # Remove from remaining dimensions since it's been mapped
                        remaining_new_dimensions.pop(new_name, None)

        # Add any remaining unmapped dimensions as new
        for new_name, new_dim in remaining_new_dimensions.items():
            if self.verbose:
                print(f"    - '{new_name}': Keeping as new dimension")
            updated_dimensions[new_name] = new_dim
            self.all_dimensions[new_name] = new_dim
            novel_dimensions.append(new_name)

        num_old_dimensions = len(self.all_dimensions) - len(novel_dimensions)
        print(f"Num old dimensions: {num_old_dimensions} | New dimensions ({len(novel_dimensions)}/{len(truly_new_dimensions)}): {novel_dimensions}")
        if self.verbose: print()

        return updated_dimensions, True

    def _check_dimensions_against_chunk(self, new_dimensions: Dict[str, Dict], existing_chunk: Dict[str, Dict]) -> Dict[str, Dict]:
        """Check multiple new dimensions against a chunk of existing dimensions in a single API call."""
        if not existing_chunk or not new_dimensions:
            return {name: {"action": "keep_new", "reason": "No existing dimensions to compare against"} for name in new_dimensions.keys()}

        # Format new dimensions for the prompt
        new_dims_text = "\n".join([f"- {name}: {dim.get('description', '')}" for name, dim in new_dimensions.items()])

        prompt = (
            f"You are comparing multiple new personalization dimensions against existing ones to identify semantic overlaps.\n\n"
            f"NEW DIMENSIONS TO CHECK:\n"
            f"{new_dims_text}\n\n"
            f"EXISTING DIMENSIONS IN THIS CHUNK:\n"
            f"{json.dumps({name: dim['description'] for name, dim in existing_chunk.items()}, indent=2)}\n\n"
            f"For each new dimension, determine if it is semantically equivalent or highly similar to ANY of the existing dimensions. "
            f"Two dimensions are similar if they measure essentially the same concept, even if phrased differently.\n\n"
            f"EXAMPLES OF SIMILARITY:\n"
            f'- "Visual Learning Style" and "Diagram Preference" would be similar (both about visual aids)\n'
            f'- "Explanation Detail" and "Verbosity Level" would be similar (both about amount of detail)\n'
            f'- "Mathematical Formality" and "Code Style" would be different (domain-specific concepts)\n\n'
            f"For each new dimension, if similar to an existing dimension, provide the existing dimension name.\n"
            f"If unique, mark it as new.\n\n"
            f"Format your response as JSON with one entry per new dimension:\n"
            f"{{\n"
            f'    "dimension_name_1": {{\n'
            f'        "action": "map_to_existing" or "keep_new",\n'
            f'        "existing_dimension": "exact name from existing list (only if mapping)",\n'
            f'        "reason": "brief explanation of your decision"\n'
            f'    }},\n'
            f'    "dimension_name_2": {{ ... }}\n'
            f"}}\n"
        )

        try:
            result = self._call_llm(prompt)

            # Validate and process results for each new dimension
            processed_results = {}
            for new_name in new_dimensions.keys():
                if new_name in result:
                    dim_result = result[new_name]
                    action = dim_result.get("action")
                    existing_dimension = dim_result.get("existing_dimension")

                    # Validate the result
                    if action == "map_to_existing":
                        if existing_dimension and existing_dimension in existing_chunk:
                            processed_results[new_name] = dim_result
                        else:
                            # Invalid mapping, treat as new
                            processed_results[new_name] = {"action": "keep_new", "reason": "Invalid existing dimension mapping"}
                    elif action == "keep_new":
                        processed_results[new_name] = dim_result
                    else:
                        # Invalid action, treat as new
                        processed_results[new_name] = {"action": "keep_new", "reason": "Invalid action returned"}
                else:
                    # Missing result for this dimension, treat as new
                    processed_results[new_name] = {"action": "keep_new", "reason": "No result provided for this dimension"}

            return processed_results

        except Exception as e:
            if self.verbose:
                print(f"    Error checking dimensions against chunk: {e}")
            # On error, treat all as new dimensions
            return {name: {"action": "keep_new", "reason": "Error during comparison"} for name in new_dimensions.keys()}

    def _check_dimension_against_chunk(self, new_name: str, new_dim: Dict, existing_chunk: Dict[str, Dict]) -> Optional[Dict]:
        """Legacy method - kept for backward compatibility. Use _check_dimensions_against_chunk for efficiency."""
        result = self._check_dimensions_against_chunk({new_name: new_dim}, existing_chunk)
        return result.get(new_name)

    def sample_personas(self, relevant_dimensions: Dict[str, Dict], num_personas: int = 2) -> List[Dict]:
        """Generate personas with different preference profiles from the relevant dimensions."""
        if not relevant_dimensions:
            raise ValueError("No relevant dimensions available to generate personas.")

        personas = []

        for i in range(num_personas):
            # Randomize dimension order to prevent order bias
            dimension_names = list(relevant_dimensions.keys())
            random.shuffle(dimension_names)

            # Build prompt for comprehensive persona generation
            existing_personas_context = ""
            if i > 0:
                existing_descriptions = [f"- {persona.get('minimal_necessary_description', 'No description available')}" for persona in personas]
                newline = "\n"
                existing_personas_context = (
                    f"EXISTING PERSONAS TO DIFFERENTIATE FROM:\n"
                    f"{newline.join(existing_descriptions)}\n"
                    f"Your new persona MUST be significantly different from all the above personas in terms of:\n"
                    f"- Demographics (age, occupation, background, life stage)\n"
                    f"- Personality traits (Big Five scores should vary considerably)\n"
                    f"- Educational and cognitive characteristics\n"
                    f"- Life experiences and challenges\n"
                    f"- Learning preferences and approaches\n\n"
                )

            prompt = (
                f"You are an expert cognitive psychologist and educational researcher specializing in learner diversity and individual differences. "
                f"Create a detailed, realistic persona that represents authentic human variation in learning preferences and characteristics.\n\n"
                f"{existing_personas_context}"
                f"Create a comprehensive persona following this framework:\n\n"
                f"## PERSONAL DEMOGRAPHICS & BACKGROUND\n"
                f"- Name: A culturally diverse, plausible name\n"
                f"- Age: Specific age (18-75 years)\n"
                f"- Occupation/Status: Current job, student status, or life situation\n"
                f"- Location: Geographic context that might influence perspectives\n"
                f"- Family/Social Context: Relevant family or social circumstances\n"
                f"- Hobbies/Interests: 2-3 specific interests that could inform analogies and explanations\n\n"
                f"## EDUCATIONAL PROFILE\n"
                f"Based on educational psychology research, include:\n\n"
                f"1. **Knowledge Level**: Current expertise and educational background\n"
                f"2. **Common Errors/Misconceptions**: Specific types of mistakes this person tends to make\n"
                f"3. **Cognitive Features**:\n"
                f"   - Learning style preferences (visual, auditory, kinesthetic, reading/writing)\n"
                f"   - Attention span and focus patterns\n"
                f"   - Memory strengths/weaknesses\n"
                f"   - Problem-solving approach\n"
                f"   - Critical thinking tendencies\n"
                f"   - Collaborative vs independent learning preference\n"
                f"4. **Affective Features**:\n"
                f"   - Confidence levels in learning\n"
                f"   - Anxiety patterns or triggers\n"
                f"   - Motivation sources\n"
                f"   - Attitude toward challenges and mistakes\n"
                f"5. **Meta-cognitive Features**:\n"
                f"   - Self-awareness of learning strengths/weaknesses\n"
                f"   - Self-regulation strategies\n"
                f"   - Reflection habits\n"
                f"   - Self-assessment abilities\n\n"
                f"## PERSONALITY PROFILE (Big Five Traits)\n"
                f"Rate each trait (1-5 scale) and explain how it affects learning:\n"
                f"- **Openness**: Creativity, curiosity, willingness to try new approaches\n"
                f"- **Conscientiousness**: Organization, attention to detail, persistence\n"
                f"- **Extraversion**: Social learning preferences, need for interaction\n"
                f"- **Agreeableness**: Cooperation, willingness to ask for help\n"
                f"- **Neuroticism**: Stress responses, emotional regulation during learning\n\n"
                f"## COMPELLING BACKSTORY\n"
                f"Create a rich narrative that connects all the above elements, explaining:\n"
                f"- Key formative experiences that shaped their learning approach\n"
                f"- Specific challenges they've overcome or still face\n"
                f"- Professional or personal situations that require continuous learning\n"
                f"- How their background influences their current learning needs\n\n"
                f"## DOMAIN EXPERTISE FOR ANALOGIES\n"
                f"Identify 2-3 domains they know well (from hobbies, work, life experience) that could be used for analogies and explanations.\n\n"
                f"Available personalization dimensions to address:\n"
            )

                        # Add dimensions details
            for dim_name in dimension_names:
                dim = relevant_dimensions[dim_name]
                prompt += f"- {dim_name}: {dim['description']} (Range: {dim['value_range']})\n"

            # Add instructions for preference generation
            prompt += (
                f"\n"
                f"For each personalization dimension, provide:\n"
                f"1. **Value**: Specific value within the specified range that fits this persona's profile\n"
                f"2. **Local Importance**: 1-5 scale of how much this person cares about this aspect\n"
                f"3. **Justification**: Detailed explanation connecting the preference to their educational profile, personality, and backstory\n\n"
                f"CRITICAL REQUIREMENTS:\n"
                f"- Ensure internal consistency across all persona elements\n"
                f"- Create realistic human variation - avoid stereotypes or overly perfect personas\n"
                f"- Ground preferences in the person's authentic characteristics and experiences\n"
                f"- Make personas feel like real people you might meet, not idealized examples\n"
                f"- Vary personality traits, backgrounds, and learning profiles significantly across different personas\n"
                f"- DO NOT create personas similar to the Maria Santos example - this is just a template showing the JSON structure\n"
                f"- Generate DIVERSE personas across age groups (18-75), occupations, cultural backgrounds, education levels, and life experiences\n"
                f"- Avoid creating multiple personas with similar demographics, personalities, or learning profiles\n"
                f"- Include personas with different socioeconomic backgrounds, life stages, and challenges\n"
                f"- Ensure variety in Big Five personality scores - don't make all personas high-achieving or similar in temperament\n"
                f"- Create authentic diversity in cognitive and affective learning characteristics\n\n"
                f"Format as JSON:\n"
                f"{{\n"
                f'    "name": "Maria Santos",\n'
                f'    "minimal_necessary_description": "28-year-old single mother transitioning from high school math teacher to software developer, highly conscientious and methodical learner who prefers thorough explanations",\n'
                f'    "demographics": {{\n'
                f'        "age": 28,\n'
                f'        "occupation": "High school math teacher transitioning to software development",\n'
                f'        "location": "Phoenix, Arizona",\n'
                f'        "family_context": "Single parent of 8-year-old daughter",\n'
                f'        "hobbies": ["rock climbing", "pottery", "board games"]\n'
                f'    }},\n'
                f'    "educational_profile": {{\n'
                f'        "knowledge_level": "Strong mathematical foundation but new to programming concepts",\n'
                f'        "common_errors": "Tends to overthink simple problems and miss obvious solutions when stressed",\n'
                f'        "cognitive_features": {{\n'
                f'            "learning_style": "Visual-kinesthetic learner who benefits from hands-on practice",\n'
                f'            "attention_span": "Good focus for 45-60 minutes, needs breaks",\n'
                f'            "memory_strengths": "Excellent pattern recognition, weaker with arbitrary symbol memorization",\n'
                f'            "problem_solving": "Methodical, likes to break down complex problems systematically",\n'
                f'            "collaboration": "Prefers initial independent work followed by group discussion"\n'
                f'        }},\n'
                f'        "affective_features": {{\n'
                f'            "confidence": "Confident in familiar domains, anxious when learning completely new topics",\n'
                f'            "anxiety_triggers": "Time pressure and fear of making mistakes in front of others",\n'
                f'            "motivation": "Driven by desire to be a good role model for daughter and career advancement",\n'
                f'            "challenge_attitude": "Views mistakes as learning opportunities but needs encouragement"\n'
                f'        }},\n'
                f'        "metacognitive_features": {{\n'
                f'            "self_awareness": "High awareness of own learning patterns and needs",\n'
                f'            "self_regulation": "Good at planning study sessions and setting realistic goals",\n'
                f'            "reflection": "Regularly reflects on what worked and what didn\'t",\n'
                f'            "self_assessment": "Tends to be overly critical of own performance"\n'
                f'        }}\n'
                f'    }},\n'
                f'    "personality_big5": {{\n'
                f'        "openness": {{"score": 4, "description": "Curious about new ideas but prefers structured approaches to learning them"}},\n'
                f'        "conscientiousness": {{"score": 5, "description": "Highly organized and persistent, always completes assignments"}},\n'
                f'        "extraversion": {{"score": 3, "description": "Moderately social, enjoys discussion but also needs quiet time to process"}},\n'
                f'        "agreeableness": {{"score": 4, "description": "Very supportive of others, quick to offer help"}},\n'
                f'        "neuroticism": {{"score": 3, "description": "Generally stable but can become anxious under time pressure"}}\n'
                f'    }},\n'
                f'    "backstory": "Maria taught high school algebra for 6 years and loved helping struggling students find their \'aha\' moments. However, budget cuts led to larger class sizes and less individual attention time. Inspired by educational technology and wanting better career stability, she decided to transition to software development. She\'s been learning programming for 8 months through online courses and bootcamps while still teaching part-time. Her teaching background makes her highly aware of different learning styles, and she often finds herself thinking about how to explain coding concepts to her future students. The career transition has been challenging as she balances single parenthood, work, and studying, but her daughter\'s admiration for her \'learning computers\' keeps her motivated.",\n'
                f'    "domain_expertise": ["mathematics education", "child development", "outdoor recreation"],\n'
                f'    "preferences": {{\n'
                f'        "Explanation Depth": {{\n'
                f'            "value": 4,\n'
                f'            "local_importance": 5,\n'
                f'            "justification": "As a teacher, Maria appreciates thorough explanations that help her understand not just the \'what\' but the \'why\' so she can later explain concepts to others. Her high conscientiousness means she wants to fully understand before moving on."\n'
                f'        }}\n'
                f'    }}\n'
                f"}}\n"
            )

            persona_data = self._call_llm(prompt)
            persona_data["id"] = f"persona_{len(personas) + 1}"
            personas.append(persona_data)

            print(f"Generated persona {persona_data['id']}: {persona_data['name']} ({persona_data['demographics']['age']}, {persona_data['demographics']['occupation']})")
            print(f"  Background: {persona_data.get('backstory', 'No backstory provided')[:100]}...")
            for k, v in persona_data['preferences'].items():
                print(f"    - '{k}': {v['value']} (importance: {v['local_importance']})")
            print()

        return personas

    def _create_diverse_persona_prompt(self, persona_library: Dict[str, Dict], iteration: int) -> str:
        """Create a diversity-enforcing prompt for persona generation using LLM-driven coherent sampling."""
        import random

        # Keep the diverse occupation list since you liked it
        unique_occupations = [
            # Unique trades & crafts
            "Beekeeper specializing in urban apiaries", "Traditional blacksmith making custom tools",
            "Mushroom cultivation specialist", "Wind turbine maintenance technician", "Glassblowing artist",
            "Underwater welding contractor", "Professional knife sharpener", "Locksmith and security consultant",
            "Arborist and tree surgeon", "Solar panel installation supervisor", "Tile restoration craftsperson",
            "Vintage furniture restoration specialist", "Leather crafting artisan", "Clock repair technician",

            # Creative & performance
            "Puppeteer for children's theater", "Voice actor for audiobooks", "Street chalk artist",
            "Perfume blending consultant", "Cake sculptor for special events", "Muralist for public art projects",
            "Circus aerial instructor", "Professional storyteller", "Jewelry repair specialist",
            "Vintage clothing restoration expert", "Calligraphy and hand lettering artist", "Mime performer",

            # Service & hospitality
            "Food truck owner (fusion cuisine)", "Wedding day coordinator", "Personal shopping consultant",
            "Professional dog walker and pet sitter", "Mobile massage therapist", "Historical walking tour guide",
            "Craft cocktail bartender", "Floral arrangement designer", "Event lighting technician",
            "Pet grooming salon owner", "House sitting and plant care service", "Personal chef for seniors",

            # Specialized professionals
            "Court sign language interpreter", "Forensic accounting investigator", "Sleep study technician",
            "Professional genealogy researcher", "Wine sommelier and educator", "Sports play-by-play announcer",
            "Antique appraisal specialist", "Patent application examiner", "Customs and border inspection officer",
            "Lighthouse maintenance keeper", "Archaeological site surveyor", "Professional mediator",

            # Modern/tech adjacent
            "Podcast producer and sound engineer", "Social media content strategist", "Drone photography operator",
            "3D printing prototyping service", "Virtual reality experience designer", "Cryptocurrency mining consultant",
            "Mobile app user experience tester", "Online course curriculum developer", "Digital nomad travel blogger",
            "Twitch stream moderator", "NFT marketplace curator", "AI training data labeler",

            # Community & social services
            "Community garden program coordinator", "Volunteer firefighter and EMT", "Neighborhood watch organizer",
            "Food bank logistics manager", "Youth recreational sports coach", "Senior center activity director",
            "Disaster relief coordination volunteer", "Animal shelter volunteer coordinator", "Crisis hotline counselor",

            # Unusual/niche specialties
            "Professional home organizer", "Color therapy and interior consultant", "Feng shui space arrangement advisor",
            "Animal behavior and training specialist", "Escape room puzzle designer", "Crossword puzzle constructor",
            "Professional cuddling therapist", "Ghost tour historian and guide", "Competitive eating coach",
            "Professional line-waiting service", "Mystery shopper and service evaluator", "Professional mourner"
        ]

        # Analyze existing personas for diversity gaps
        existing_occupations = set(persona.get('demographics', {}).get('occupation', '') for persona in persona_library.values())
        existing_names = set(persona.get('name', '') for persona in persona_library.values())
        existing_ages = [persona.get('demographics', {}).get('age', 0) for persona in persona_library.values()]
        existing_locations = set(persona.get('demographics', {}).get('location', '') for persona in persona_library.values())

        # Create diversity constraints
        age_ranges = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 65), (66, 75)]
        range_counts = {r: sum(1 for age in existing_ages if r[0] <= age <= r[1]) for r in age_ranges}
        underrepresented_ranges = [r for r, count in range_counts.items() if count == min(range_counts.values())] if range_counts else age_ranges

        # Create context about existing personas
        existing_personas_context = ""
        if persona_library:
            recent_personas = list(persona_library.values())[-8:]  # Show last 8 to avoid overwhelming
            existing_summaries = []
            for persona in recent_personas:
                summary = (f"- {persona.get('name', 'Unknown')}: {persona.get('demographics', {}).get('age', '?')} yr old "
                          f"{persona.get('demographics', {}).get('occupation', 'Unknown occupation')} "
                          f"from {persona.get('demographics', {}).get('location', 'Unknown location')}")
                existing_summaries.append(summary)

            existing_personas_context = (
                f"RECENT PERSONAS TO DIFFERENTIATE FROM:\n"
                f"{chr(10).join(existing_summaries)}\n\n"
                f"Your new persona MUST be completely different in demographics, personality, and life situation.\n\n"
            )

        # Use few-shot examples or constraints based on probability
        use_constraints = random.random() < 0.7  # 70% chance to use constraints, 30% pure LLM creativity

        if use_constraints:
            # Constrained generation for diversity
            available_occupations = [occ for occ in unique_occupations if occ not in existing_occupations]
            if not available_occupations:
                available_occupations = unique_occupations  # Reset if exhausted

            suggested_occupation = random.choice(available_occupations)
            suggested_age_range = random.choice(underrepresented_ranges)

            prompt = (
                f"Create a completely unique, authentic person with genuine human complexity. Ensure all attributes are coherent and realistic.\n\n"
                f"{existing_personas_context}"
                f"DIVERSITY GUIDANCE (adapt as needed for coherence):\n"
                f"- Suggested occupation: {suggested_occupation}\n"
                f"- Suggested age range: {suggested_age_range[0]}-{suggested_age_range[1]} years old\n"
                f"- Avoid names already used: {', '.join(list(existing_names)[-5:]) if existing_names else 'None'}\n"
                f"- Avoid locations already used: {', '.join(list(existing_locations)[-5:]) if existing_locations else 'None'}\n\n"
                f"CRITICAL: Ensure all attributes make sense together. If the suggested occupation doesn't fit the age range, "
                f"modify either the age or occupation to create a coherent, realistic person.\n\n"
            )
        else:
            # Pure LLM creativity with few-shot examples showcasing diverse career transitions
            example_personas = [
                "Carmen Vásquez (41, Food truck owner specializing in fusion cuisine from Austin, Texas): Former struggling actress who spent 15 years auditioning in LA, learned resilience through constant rejection, now channels creativity into culinary arts, prefers hands-on learning and visual demonstrations.",
                "Dmitri Petrov (29, Professional genealogy researcher from St. Petersburg, Russia): Ex-circus performer who traveled with his family's acrobatic troupe until a knee injury ended his career, meticulous attention to detail from years of precise physical training, learns best through systematic step-by-step processes.",
                "Asha Okonkwo (56, Crisis hotline counselor from Birmingham, Alabama): Former coal miner who worked underground for 20 years before the mine closed, developed deep listening skills and patience from dangerous work requiring constant communication, prefers collaborative learning and real-world examples."
            ]

            prompt = (
                f"Create a completely unique, authentic person with genuine human complexity. Draw inspiration from these examples of diverse, coherent personas:\n\n"
                f"EXAMPLES OF COHERENT DIVERSITY:\n"
                f"{chr(10).join(f'- {example}' for example in example_personas)}\n\n"
                f"{existing_personas_context}"
                f"Create someone equally unique and coherent. Ensure all attributes (age, occupation, background, personality) "
                f"form a realistic, internally consistent person.\n\n"
            )

        prompt += (
            f"## COHERENT PERSONA GENERATION\n"
            f"Create a realistic person where all attributes support each other:\n\n"
            f"### BASIC DEMOGRAPHICS (must be coherent)\n"
            f"- Name: Culturally diverse name that fits their background\n"
            f"- Age: Realistic age for their occupation and life experience\n"
            f"- Occupation: Should match their age, education, and life path\n"
            f"- Location: Specific place that influences their worldview and opportunities\n"
            f"- Cultural Background: Heritage that may influence their learning style and values\n\n"
            f"### LIFE SITUATION (must be realistic)\n"
            f"- Family Context: Family situation that makes sense for their age/culture\n"
            f"- Economic Reality: Financial situation consistent with their occupation\n"
            f"- Living Situation: Housing/lifestyle that fits their income and choices\n"
            f"- Daily Challenges: Real problems someone in their situation would face\n\n"
            f"### EDUCATIONAL JOURNEY (must be coherent)\n"
            f"- Formal Education: Educational path that led to their current occupation\n"
            f"- Career Path: How they got to where they are (career changes, etc.)\n"
            f"- Self-Taught Skills: Skills they developed outside formal education\n"
            f"- Learning Style: How they prefer to learn new things\n"
            f"- Knowledge Gaps: Areas where they lack confidence or knowledge\n\n"
            f"### PERSONALITY & PSYCHOLOGY (realistic complexity)\n"
            f"- Big Five Traits: Personality scores (1-5) that create authentic human complexity\n"
            f"- Strengths: What they're naturally good at\n"
            f"- Struggles: Realistic challenges and imperfections\n"
            f"- Motivations: What drives them in their current life stage\n"
            f"- Quirks: Unique behavioral patterns or habits\n\n"
            f"### PERSONAL STORY (compelling and coherent)\n"
            f"- Backstory: Life events that shaped who they are today\n"
            f"- Current Goals: What they're working toward\n"
            f"- Hidden Depths: Unexpected talents or interests\n"
            f"- Hobbies: 2-3 interests that fit their personality and situation\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. **Coherence**: All attributes must make sense together (age ↔ occupation ↔ experience)\n"
            f"2. **Authenticity**: Create someone who feels like a real person you could meet\n"
            f"3. **Complexity**: Include contradictions and imperfections that make them human\n"
            f"4. **Diversity**: Ensure they're genuinely different from existing personas\n"
            f"5. **Cultural Sensitivity**: Represent backgrounds authentically and respectfully\n\n"
            f"REQUIRED JSON FORMAT:\n"
            f"{{\n"
            f'    "name": "Full Name",\n'
            f'    "minimal_necessary_description": "Brief 1-sentence description capturing key characteristics",\n'
            f'    "demographics": {{\n'
            f'        "age": 35,\n'
            f'        "occupation": "Specific occupation that matches their age and background",\n'
            f'        "location": "Specific city, state/province, country",\n'
            f'        "family_context": "Family situation that makes sense for their age/culture",\n'
            f'        "hobbies": ["hobby1", "hobby2", "hobby3"]\n'
            f'    }},\n'
            f'    "educational_profile": {{\n'
            f'        "knowledge_level": "Current expertise and educational background",\n'
            f'        "common_errors": "Specific types of mistakes this person tends to make",\n'
            f'        "cognitive_features": {{\n'
            f'            "learning_style": "How they prefer to learn (visual, auditory, kinesthetic, etc.)",\n'
            f'            "attention_span": "Their focus patterns and attention capabilities",\n'
            f'            "memory_strengths": "What they remember well vs. struggle with",\n'
            f'            "problem_solving": "Their approach to solving problems",\n'
            f'            "collaboration": "Preference for independent vs collaborative learning"\n'
            f'        }},\n'
            f'        "affective_features": {{\n'
            f'            "confidence": "Their confidence levels in learning situations",\n'
            f'            "anxiety_triggers": "What makes them anxious or stressed",\n'
            f'            "motivation": "What drives them to learn and achieve",\n'
            f'            "challenge_attitude": "How they respond to difficulties and mistakes"\n'
            f'        }},\n'
            f'        "metacognitive_features": {{\n'
            f'            "self_awareness": "How well they understand their own learning",\n'
            f'            "self_regulation": "Their ability to manage their learning process",\n'
            f'            "reflection": "How they think about their learning experiences",\n'
            f'            "self_assessment": "How they evaluate their own performance"\n'
            f'        }}\n'
            f'    }},\n'
            f'    "personality_big5": {{\n'
            f'        "openness": {{"score": 4, "description": "Creativity, curiosity, willingness to try new approaches"}},\n'
            f'        "conscientiousness": {{"score": 3, "description": "Organization, attention to detail, persistence"}},\n'
            f'        "extraversion": {{"score": 2, "description": "Social learning preferences, need for interaction"}},\n'
            f'        "agreeableness": {{"score": 5, "description": "Cooperation, willingness to ask for help"}},\n'
            f'        "neuroticism": {{"score": 2, "description": "Stress responses, emotional regulation during learning"}}\n'
            f'    }},\n'
            f'    "backstory": "Rich narrative explaining their background, formative experiences, current situation, and how their past shapes their learning needs",\n'
            f'    "domain_expertise": ["domain1", "domain2", "domain3"]\n'
            f"}}\n\n"
            f"Fill out ALL fields with realistic, coherent information for your unique persona.\n"
        )

        return prompt

    def load_persona_library(self) -> List[Dict]:
        """Load or generate a library of diverse personas."""

        # Try to load existing persona library
        if os.path.exists(self.persona_library_filepath):
            print(f"Loading existing persona library from {self.persona_library_filepath}")
            with open(self.persona_library_filepath, 'r') as f:
                data = json.load(f)
                loaded_persona_library = data.get("personas", [])
                loaded_persona_library = {persona['id']: persona for persona in loaded_persona_library}
                print(f"Loaded {len(loaded_persona_library)} existing personas")
                self.persona_library.update(loaded_persona_library)

        # Check if we have exactly the required number of personas
        if len(self.persona_library) == self.initial_num_personas:
            print(f"Persona library already has the required {self.initial_num_personas} personas. Using existing library.")
        elif len(self.persona_library) > self.initial_num_personas:
            print(f"Persona library has {len(self.persona_library)} personas, more than required {self.initial_num_personas}. Using first {self.initial_num_personas} personas.")
            # Keep only the first N personas to maintain consistency
            persona_ids = list(self.persona_library.keys())[:self.initial_num_personas]
            self.persona_library = {pid: self.persona_library[pid] for pid in persona_ids}
        else:
            # Generate additional personas if needed
            personas_needed = self.initial_num_personas - len(self.persona_library)
            print(f"Generating {personas_needed} additional personas for the library using random model selection...")

            for i in range(personas_needed):
                # Use random model selection for persona generation
                print(f"Generating persona {len(self.persona_library) + 1}/{self.initial_num_personas}...")
                new_persona = self._generate_single_persona(len(self.persona_library))
                self.persona_library[new_persona['id']] = new_persona

        # Ensure we have exactly the required number
        assert len(self.persona_library) == self.initial_num_personas, f"Expected {self.initial_num_personas} personas, got {len(self.persona_library)}"

        # Save the updated persona library
        from datetime import datetime
        library_data = {
            "personas": list(self.persona_library.values()),
            "modified_at": datetime.now().isoformat(),
            "total_personas": len(self.persona_library),
            "generation_complete": True,
            "fixed_library": True
        }

        with open(self.persona_library_filepath, 'w') as f:
            json.dump(library_data, f, indent=2)

        print(f"Fixed persona library with exactly {len(self.persona_library)} personas saved to {self.persona_library_filepath}")
        print(f"  (This shared persona library can be reused across different tasks)")


    def _generate_single_persona(self, current_library_size: int) -> Dict:
        """Generate a single new persona to add to the library with coherence validation."""
        max_attempts = 3

        for attempt in range(max_attempts):
            prompt = self._create_diverse_persona_prompt(self.persona_library, current_library_size)
            persona_data = self._call_llm(prompt)

            # Validate that persona_data is a dictionary
            if not isinstance(persona_data, dict):
                print(f"Attempt {attempt + 1}: LLM returned invalid persona data type: {type(persona_data)}, retrying...")
                continue

            # Check for required fields
            if not persona_data.get('name') or not isinstance(persona_data.get('demographics'), dict):
                print(f"Attempt {attempt + 1}: Persona data missing required fields, retrying...")
                continue

            # Validate coherence
            if self._validate_persona_coherence(persona_data):
                persona_data["id"] = self._generate_persona_id(persona_data)
                persona_data["usage_count"] = 0

                print(f"Generated new persona {persona_data['id']}: {persona_data['name']} ({persona_data['demographics']['age']}, {persona_data['demographics']['occupation']})")
                return persona_data
            else:
                print(f"Attempt {attempt + 1}: Generated persona failed coherence validation, retrying...")

        # If all attempts fail, create a minimal fallback persona
        print("Warning: All attempts failed, creating fallback persona")
        fallback_persona = {
            "name": f"Fallback User {current_library_size + 1}",
            "minimal_necessary_description": "A basic user profile created as fallback",
            "demographics": {
                "age": 30,
                "occupation": "General user",
                "location": "Unknown",
                "family_context": "Not specified",
                "hobbies": ["reading", "learning"]
            },
            "educational_profile": {
                "knowledge_level": "Intermediate",
                "common_errors": "Standard learning challenges",
                "cognitive_features": {
                    "learning_style": "Mixed approach",
                    "attention_span": "Average",
                    "memory_strengths": "General",
                    "problem_solving": "Methodical",
                    "collaboration": "Flexible"
                },
                "affective_features": {
                    "confidence": "Moderate",
                    "anxiety_triggers": "Time pressure",
                    "motivation": "Learning and growth",
                    "challenge_attitude": "Positive"
                },
                "metacognitive_features": {
                    "self_awareness": "Good",
                    "self_regulation": "Adequate",
                    "reflection": "Regular",
                    "self_assessment": "Realistic"
                }
            },
            "personality_big5": {
                "openness": {"score": 3, "description": "Moderately open to new experiences"},
                "conscientiousness": {"score": 3, "description": "Reasonably organized and persistent"},
                "extraversion": {"score": 3, "description": "Balanced social preferences"},
                "agreeableness": {"score": 3, "description": "Generally cooperative"},
                "neuroticism": {"score": 3, "description": "Emotionally stable"}
            },
            "backstory": "A general user profile created when persona generation failed.",
            "domain_expertise": ["general knowledge"]
        }
        fallback_persona["id"] = self._generate_persona_id(fallback_persona)
        fallback_persona["usage_count"] = 0
        return fallback_persona

    def _validate_persona_coherence(self, persona_data: Dict) -> bool:
        """Validate that persona attributes are coherent and realistic."""
        try:
            demographics = persona_data.get('demographics', {})
            age = demographics.get('age', 0)
            occupation = demographics.get('occupation', '')
            name = persona_data.get('name', '')

            # Basic validation checks
            if not (18 <= age <= 75):
                print(f"Invalid age: {age}")
                return False

            if not name or len(name.split()) < 2:
                print(f"Invalid name format: {name}")
                return False

            if not occupation:
                print("Missing occupation")
                return False

            # Use LLM to check for coherence issues
            coherence_prompt = (
                f"Evaluate if this persona has any major coherence issues. Look for impossible combinations "
                f"or unrealistic attributes:\n\n"
                f"Name: {name}\n"
                f"Age: {age}\n"
                f"Occupation: {occupation}\n"
                f"Location: {demographics.get('location', 'Not specified')}\n"
                f"Educational Background: {persona_data.get('educational_profile', {}).get('knowledge_level', 'Not specified')}\n\n"
                f"Check for issues like:\n"
                f"- Age incompatible with occupation (e.g., 20-year-old retired surgeon)\n"
                f"- Impossible educational/career timeline\n"
                f"- Contradictory personality traits\n"
                f"- Unrealistic combinations\n\n"
                f"Respond with JSON: {{'coherent': true/false, 'issues': ['list of any problems found']}}\n"
            )

            validation_result = self._call_llm(coherence_prompt)
            is_coherent = validation_result.get('coherent', True)

            if not is_coherent:
                issues = validation_result.get('issues', ['Unknown coherence issues'])
                print(f"Coherence issues found: {', '.join(issues)}")

            return is_coherent

        except Exception as e:
            print(f"Error validating persona coherence: {e}")
            return True  # Default to accepting if validation fails

    def _save_persona_library_to_file(self):
        """Save the current persona library to file."""
        from datetime import datetime
        library_data = {
            "personas": list(self.persona_library.values()),
            "last_updated": datetime.now().isoformat(),
            "total_personas": len(self.persona_library)
        }

        with open(self.persona_library_filepath, 'w') as f:
            json.dump(library_data, f, indent=2)

    def _generate_raw_dimensions_only(self, problem: Dict, persona: Dict, problem_index: int) -> Tuple[Dict[str, Dict], bool]:
        """
        Generate raw preference dimensions WITHOUT deduplication.
        This is used in phase 1 of parallel processing.
        Returns raw dimensions that will be deduplicated later in batch.
        """
        problem_text, domain, image = self._extract_problem_content(problem)

        raw_dimensions, success = self._sample_pref_dimensions_from_persona_problem(
            persona, problem_index, problem_text, domain, image, try_idx=0, max_tries=3
        )

        return raw_dimensions, success

    def _batch_deduplicate_dimensions(self, all_raw_dimensions: List[Dict[str, Dict]]) -> Dict[str, Dict[str, str]]:
        """
        Deduplicate multiple sets of raw dimensions in a single batch operation.
        Returns a mapping from raw dimension names to canonical names.
        Much more efficient than deduplicating one at a time.
        """
        # Collect all unique dimension names from all batches
        all_new_dims = {}
        for raw_dims in all_raw_dimensions:
            for dim_name, dim_data in raw_dims.items():
                if dim_name not in self.all_dimensions and dim_name not in all_new_dims:
                    all_new_dims[dim_name] = dim_data

        if not all_new_dims:
            return {}  # Nothing to deduplicate

        print(f"  Batch deduplicating {len(all_new_dims)} new unique dimensions...")

        # Now check all new dimensions against existing ones in one call
        with self._dimensions_lock:
            mapping, _ = self.update_preferences(all_new_dims)

        # Create dimension name mapping (raw name -> canonical name)
        name_mapping = {}
        for dim_name in all_new_dims.keys():
            # Find the canonical name in mapping
            if dim_name in mapping:
                name_mapping[dim_name] = dim_name  # Kept as is
            else:
                # Check if it was mapped to an existing dimension
                for canonical_name in self.all_dimensions.keys():
                    if canonical_name != dim_name:
                        # This dimension was mapped to an existing one
                        name_mapping[dim_name] = canonical_name
                        break
                else:
                    name_mapping[dim_name] = dim_name  # Fallback

        return name_mapping

    def _generate_single_problem_parallel(self, task: Tuple[int, Dict, Dict, str, set]) -> Optional[Dict]:
        """
        Generate a single personalized problem (for parallel execution).
        This is a thread-safe version that returns results instead of writing directly.

        Args:
            task: Tuple of (problem_index, problem, persona, problem_id, existing_personas_for_problem)

        Returns:
            Dict with results or None if failed
        """
        problem_index, problem, persona, problem_id, _ = task

        try:
            # Phase 1: Generate raw dimensions (no deduplication yet)
            raw_dimensions, success1 = self._generate_raw_dimensions_only(problem, persona, problem_index)

            if not success1 or not raw_dimensions:
                return None

            # Phase 2: Deduplicate (thread-safe)
            with self._dimensions_lock:
                unique_dimensions, success2 = self.update_preferences(raw_dimensions)

            if not success2 or not unique_dimensions:
                return None

            # Phase 3: Instantiate persona preferences
            problem_text, domain, _ = self._extract_problem_content(problem)
            persona_preferences, success3 = self._instantiate_persona_preferences(
                unique_dimensions, persona, problem_index, problem_text, domain, try_idx=0, max_tries=3
            )

            if not success3 or not persona_preferences:
                return None

            # Phase 4: Generate evaluation rubric (can run in parallel - no shared state modified)
            rubric, success4 = self.generate_evaluation_rubric(
                problem, persona_preferences, persona, problem_id, try_idx=0, max_tries=3
            )

            if not success4 or not rubric:
                return None

            # Process problem for serialization
            processed_problem = self._process_problem_for_serialization(problem, problem_id, problem_index)

            # Create result
            result = {
                "type": "personalized_problem",
                "problem_id": problem_id,
                "original_problem": processed_problem,
                "persona": {k: v for k, v in persona.items() if k != "accumulated_preferences"},
                "persona_preferences": persona_preferences,
                "evaluation_rubric": rubric,
                "generation_success": True,
                "_problem_index": problem_index,  # For tracking
                "_persona_id": persona.get('id', self._generate_persona_id(persona))
            }

            return result

        except Exception as e:
            print(f"    Error generating {problem_id}: {e}")
            return None

    def create_personalized_benchmark_parallel(self) -> str:
        """Generate the complete personalized benchmark using parallel processing.

        Uses a two-phase approach for better parallelization:
        - Phase 1: Generate raw dimensions in parallel (no dedup bottleneck)
        - Phase 2: Batch deduplicate collected dimensions
        - Phase 3: Generate preferences and rubrics in parallel
        """

        # Step 1: Load or create persona library
        self.load_persona_library()

        # Step 2: Load or generate dataset-level preferences
        self.load_all_preferences(self.all_prefs_filepath)
        self.load_dataset_preferences(self.dataset_preference_filepath)

        # Load existing progress
        existing_problem_map = {}  # Maps problem_index -> set of persona_ids
        problem_ids = set()

        if os.path.exists(self.output_file):
            print(f"Found existing output file: {self.output_file}")
            print("Loading existing progress...")

            with open(self.output_file, 'r') as f:
                self.personalized_problems = []
                valid_problems = 0

                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') == 'personalized_problem':
                            problem_id = data.get('problem_id', '')

                            if self._validate_existing_problem(data):
                                self.personalized_problems.append(data)
                                problem_ids.add(problem_id)

                                problem_index, persona_id = self._parse_problem_id(problem_id)
                                if problem_index is not None and persona_id is not None:
                                    if problem_index not in existing_problem_map:
                                        existing_problem_map[problem_index] = set()
                                    existing_problem_map[problem_index].add(persona_id)
                                    valid_problems += 1
                    except:
                        pass

                print(f"Loaded {valid_problems} valid existing problems")
        else:
            print(f"No existing output file found. Starting fresh generation.")
            self.personalized_problems = []

            with open(self.output_file, 'w') as f:
                metadata = {
                    "type": "metadata",
                    "dataset": self.dataset_name,
                    "general_preferences": self.general_preferences,
                    "total_personas_in_library": len(self.persona_library),
                    "fixed_persona_library": True,
                    "personas_per_problem": self.num_personas_per_problem,
                    "sample_size": self.sample_size,
                    "parallel_workers": self.num_workers,
                }
                f.write(json.dumps(metadata) + '\n')

        # Build list of all tasks to process
        tasks = []
        for i, problem in enumerate(self.problems):
            existing_personas = existing_problem_map.get(i, set())

            # Get personas needed for this problem
            available_personas = [p for p in self.persona_library.values()
                                if self._generate_persona_id(p) not in existing_personas]

            personas_needed = self.num_personas_per_problem - len(existing_personas)

            if personas_needed <= 0:
                continue

            # Select personas for this problem
            selected_personas = random.sample(available_personas, min(personas_needed, len(available_personas)))

            for persona in selected_personas:
                if 'id' not in persona:
                    persona['id'] = self._generate_persona_id(persona)

                problem_id = f"problem_{i}_{persona['id']}"

                if problem_id not in problem_ids:
                    tasks.append((i, problem, persona, problem_id, existing_personas))

        print(f"\nTotal tasks to process: {len(tasks)}")
        print(f"Using {self.num_workers} parallel workers")
        print(f"Batch dedup size: {self.batch_dedup_size}")

        # Process in batches for better deduplication efficiency
        successful = 0
        failed = 0
        batch_size = self.batch_dedup_size

        for batch_start in range(0, len(tasks), batch_size):
            if self._should_stop_generation():
                print(f"\nStopping due to time limit")
                break

            batch_tasks = tasks[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(tasks) + batch_size - 1) // batch_size

            print(f"\n{'='*50}")
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_tasks)} tasks)")

            # PHASE 1: Generate raw dimensions in parallel (no lock contention)
            print(f"  Phase 1: Generating raw dimensions in parallel...")
            raw_dimensions_results = {}  # task_idx -> raw_dimensions

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {
                    executor.submit(self._generate_raw_dimensions_only, task[1], task[2], task[0]): idx
                    for idx, task in enumerate(batch_tasks)
                }

                for future in tqdm(as_completed(future_to_idx), total=len(batch_tasks), desc="  Raw dimensions"):
                    idx = future_to_idx[future]
                    try:
                        raw_dims, success = future.result()
                        if success and raw_dims:
                            raw_dimensions_results[idx] = raw_dims
                    except Exception as e:
                        print(f"    Error generating dimensions for task {idx}: {e}")

            print(f"  Phase 1 complete: {len(raw_dimensions_results)}/{len(batch_tasks)} successful")

            # PHASE 2: Batch deduplicate all dimensions at once (single operation)
            print(f"  Phase 2: Batch deduplicating dimensions...")
            all_raw_dims = list(raw_dimensions_results.values())
            if all_raw_dims:
                self._batch_deduplicate_dimensions(all_raw_dims)
            self.save_all_dimensions()

            # PHASE 3: Generate preferences and rubrics in parallel
            print(f"  Phase 3: Generating preferences and rubrics in parallel...")

            # Build phase 3 tasks (only for successful phase 1 results)
            phase3_tasks = [
                (idx, batch_tasks[idx], raw_dimensions_results[idx])
                for idx in raw_dimensions_results.keys()
            ]

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_task = {
                    executor.submit(self._generate_preferences_and_rubric, task_data): task_data
                    for task_data in phase3_tasks
                }

                for future in tqdm(as_completed(future_to_task), total=len(phase3_tasks), desc="  Prefs & rubrics"):
                    task_data = future_to_task[future]
                    idx, original_task, raw_dims = task_data
                    problem_id = original_task[3]

                    try:
                        result = future.result()

                        if result is not None:
                            # Write result to file (thread-safe)
                            with self._file_lock:
                                clean_result = {k: v for k, v in result.items()
                                              if not k.startswith('_')}

                                with open(self.output_file, 'a') as f:
                                    f.write(json.dumps(clean_result) + '\n')

                                self.personalized_problems.append(clean_result)

                            # Update tracking
                            self._update_persona_preferences_tracking(
                                result['_persona_id'],
                                result['persona'],
                                original_task[1],  # problem
                                result['persona_preferences'],
                                result['_problem_index']
                            )

                            successful += 1
                        else:
                            failed += 1

                    except Exception as e:
                        print(f"    Error processing {problem_id}: {e}")
                        failed += 1

            print(f"  Batch {batch_num} complete: {successful} successful, {failed} failed total")

        # Save dimensions
        self.save_all_dimensions()
        self._save_persona_library_to_file()

        print(f"\n{'='*50}")
        print(f"Parallel Generation Complete!")
        print(f"- Successful: {successful}")
        print(f"- Failed: {failed}")
        print(f"- Total scenarios: {len(self.personalized_problems)}")
        print(f"- Output file: {self.output_file}")

        return self.output_file

    def _generate_preferences_and_rubric(self, task_data: Tuple) -> Optional[Dict]:
        """
        Phase 3 worker: Generate preferences and rubric for a task with pre-deduplicated dimensions.
        This runs after batch deduplication, so no lock contention.
        """
        idx, original_task, raw_dimensions = task_data
        problem_index, problem, persona, problem_id, _ = original_task

        try:
            # Map raw dimensions to deduplicated names
            unique_dimensions = {}
            for dim_name, dim_data in raw_dimensions.items():
                # Check if this dimension exists in all_dimensions (may have been renamed)
                if dim_name in self.all_dimensions:
                    unique_dimensions[dim_name] = self.all_dimensions[dim_name]
                else:
                    # Find if it was mapped to an existing dimension
                    # For now, just use the raw dimension (it should have been added during dedup)
                    unique_dimensions[dim_name] = dim_data

            if not unique_dimensions:
                return None

            # Generate persona preferences
            problem_text, domain, _ = self._extract_problem_content(problem)
            persona_preferences, success = self._instantiate_persona_preferences(
                unique_dimensions, persona, problem_index, problem_text, domain, try_idx=0, max_tries=3
            )

            if not success or not persona_preferences:
                return None

            # Generate evaluation rubric
            rubric, success = self.generate_evaluation_rubric(
                problem, persona_preferences, persona, problem_id, try_idx=0, max_tries=3
            )

            if not success or not rubric:
                return None

            # Process problem for serialization
            processed_problem = self._process_problem_for_serialization(problem, problem_id, problem_index)

            return {
                "type": "personalized_problem",
                "problem_id": problem_id,
                "original_problem": processed_problem,
                "persona": {k: v for k, v in persona.items() if k != "accumulated_preferences"},
                "persona_preferences": persona_preferences,
                "evaluation_rubric": rubric,
                "generation_success": True,
                "_problem_index": problem_index,
                "_persona_id": persona.get('id', self._generate_persona_id(persona))
            }

        except Exception as e:
            print(f"    Error in phase 3 for {problem_id}: {e}")
            return None



    def generate_evaluation_rubric(self, problem: Dict, persona_preferences: Dict[str, Dict], persona: Dict, problem_id: str, try_idx: int = 0, max_tries: int = 3) -> Tuple[Dict, bool]:
        """Generate evaluation rubric for a personalized problem using three separate LLM calls."""
        print(f"Generating evaluation rubric for {problem_id} with {len(persona_preferences)} persona_preferences")

        if try_idx >= max_tries:
            print(f"Error: Failed to generate evaluation rubric for {problem_id} after {max_tries} tries")
            return {}, False

        # Extract problem content including image
        problem_text, problem_type, image = self._extract_problem_content(problem)

        if self.answer_field:
            if self.dataset_name == "allenai/social_i_qa":
                solution_context = "ABC"[int(problem.get(self.answer_field, ''))-1]
            elif self.dataset_name == "lighteval/mmlu":
                solution_context = problem.get(self.choices_field, ['']*4)[int(problem.get(self.answer_field, ''))]
            elif self.dataset_name == "lucasmccabe/logiqa":
                solution_context = "ABC"[int(problem.get(self.answer_field, ''))]
            else:
                solution_context = str(problem.get(self.answer_field, ''))

        # Calculate total importance for weight normalization using all stored preferences
        total_importance = sum(pref.get('local_importance', 1) for pref in persona_preferences.values())

        # Extract preferences by type for targeted rubric generation (use all stored preferences)
        expertise_prefs = {k: v for k, v in persona_preferences.items() if v.get('type') == 'expertise'}
        personal_prefs = {k: v for k, v in persona_preferences.items() if v.get('type') == 'personal'}
        other_prefs = {k: v for k, v in persona_preferences.items() if v.get('type') not in ['expertise', 'personal']}

        # Common context for all LLM calls
        image_note = "\nNOTE: This problem includes visual content that should be considered in evaluation criteria." if image is not None else ""

        base_context = (
            f"PROBLEM TO BE ANSWERED:\n"
            f"Type: {problem_type}\n"
            f"Problem: {problem_text}{image_note}\n"
        )

        if solution_context:
            base_context += f"Correct answer: {solution_context[:200]}...\n"

        base_context += (
            f"\nPERSONA CONTEXT:\n"
            f"User: {persona.get('name', 'Unknown')} - {persona.get('minimal_necessary_description', 'No description')}\n"
            f"Background: {persona.get('backstory', 'No backstory')[:150]}...\n\n"
        )

        all_criteria = []

        # Generate expertise criteria
        if expertise_prefs:
            expertise_criteria = self._generate_expertise_rubric_criteria(
                expertise_prefs, base_context, problem_text, problem_type, total_importance
            )
            if not isinstance(expertise_criteria, list):
                print(f"ERROR: expertise_criteria is not a list: {type(expertise_criteria)} = {expertise_criteria}")
                expertise_criteria = []
            all_criteria.extend(expertise_criteria)

        # Generate personal criteria
        if personal_prefs:
            personal_criteria = self._generate_personal_rubric_criteria(
                personal_prefs, base_context, problem_text, problem_type, total_importance, persona
            )
            if not isinstance(personal_criteria, list):
                print(f"ERROR: personal_criteria is not a list: {type(personal_criteria)} = {personal_criteria}")
                personal_criteria = []
            all_criteria.extend(personal_criteria)

        # Generate other criteria
        if other_prefs:
            other_criteria = self._generate_other_rubric_criteria(
                other_prefs, base_context, problem_text, problem_type, total_importance
            )
            if not isinstance(other_criteria, list):
                print(f"ERROR: other_criteria is not a list: {type(other_criteria)} = {other_criteria}")
                other_criteria = []
            all_criteria.extend(other_criteria)

        # Combine all criteria into final rubric
        rubric = {"evaluation_criteria": all_criteria}

        # Validate rubric against stored preferences (must match exactly)
        expected_preferences = set(persona_preferences.keys())
        actual_criteria = set(c.get("preference", "") for c in rubric.get("evaluation_criteria", []))

        if expected_preferences != actual_criteria:
            print(f"WARNING: Rubric criteria mismatch for {problem_id}")
            print(f"  - Expected (stored preferences): {sorted(expected_preferences)}")
            print(f"  - Generated: {sorted(actual_criteria)}")
            print(f"  - Missing: {sorted(expected_preferences - actual_criteria)}")
            print(f"  - Extra: {sorted(actual_criteria - expected_preferences)}")
            if len(actual_criteria - expected_preferences) > 0 and len(expected_preferences - actual_criteria) == 0:
                for pref in actual_criteria - expected_preferences:
                    rubric["evaluation_criteria"] = [c for c in rubric["evaluation_criteria"] if c.get("preference", "") != pref]
                actual_criteria = set(c.get("preference", "") for c in rubric.get("evaluation_criteria", []))
            if len(actual_criteria - expected_preferences) == 0 and len(expected_preferences - actual_criteria) > 0:
                missing_criteria = self._generate_other_rubric_criteria(
                    {k: v for k, v in persona_preferences.items() if k in set(expected_preferences - actual_criteria)},
                    base_context, problem_text, problem_type, total_importance
                    )
                rubric["evaluation_criteria"].extend(missing_criteria)
                actual_criteria = set(c.get("preference", "") for c in rubric.get("evaluation_criteria", []))
            if expected_preferences != actual_criteria:
                print(f"WARNING: Rubric criteria mismatch for {problem_id} after adding missing criteria")
                print(f"  - Expected (stored preferences): {sorted(expected_preferences)}")
                print(f"  - Generated: {sorted(actual_criteria)}")
                print(f"  - Missing: {sorted(expected_preferences - actual_criteria)}")
                print(f"  - Extra: {sorted(actual_criteria - expected_preferences)}")
                return self.generate_evaluation_rubric(problem, persona_preferences, persona, problem_id, try_idx + 1, max_tries)

        # Validate weights sum to 1.0
        total_weight = sum(c.get("weight", 0) for c in rubric.get("evaluation_criteria", []))
        if abs(total_weight - 1.0) > self.weight_tolerance:
            print(f"WARNING: Rubric weights sum to {total_weight:.3f}, not 1.0 for {problem_id}")

        return rubric, True

    def _generate_expertise_rubric_criteria(self, expertise_prefs: Dict[str, Dict], base_context: str, problem_text: str, problem_type: str, total_importance: int) -> List[Dict]:
        """Generate rubric criteria specifically for expertise/comfort preferences."""

        prompt = (
            f"You are creating evaluation criteria for EXPERTISE/COMFORT preferences - how well an AI response matches a user's technical skill level and comfort with specific concepts.\n\n"
            f"{base_context}"
            f"EXPERTISE/COMFORT PREFERENCES:\n"
        )

        for pref_name, pref_data in expertise_prefs.items():
            weight = pref_data.get('local_importance', 1) / total_importance
            prompt += f"- {pref_name}: {pref_data.get('value', 'N/A')} (importance: {pref_data.get('local_importance', 1)}, weight: {weight:.3f})\n"
            prompt += f"  Why this matters: {pref_data.get('justification', 'No justification')}\n\n"

        prompt += (
            f"EXPERTISE RUBRIC GUIDELINES:\n"
            f"- Focus on TECHNICAL APPROPRIATENESS: Does the response match the user's stated comfort/skill level?\n"
            f"- Evaluate CONCEPT COMPLEXITY: Are concepts presented at the right level of sophistication?\n"
            f"- Assess TERMINOLOGY USAGE: Is technical language appropriate for their experience level?\n"
            f"- Check ASSUMPTION LEVELS: Does the response assume the right background knowledge?\n\n"
            f"CRITICAL: Create exactly ONE criterion for each expertise preference listed above. Each criterion must use the EXACT preference name.\n\n"
            f"For each expertise preference, create criteria that measure how well the AI response calibrates its technical level to match the user's capabilities for this specific {problem_type}.\n\n"
            f"EXAMPLE EXPERTISE CRITERION:\n"
            f"{{\n"
            f'    "preference": "Comfort with Linear Algebra",\n'
            f'    "description": "How well the response matches the user\'s intermediate comfort level (value: 3) with linear algebra when explaining matrix operations in this problem",\n'
            f'    "weight": 0.25,\n'
            f'    "levels": [\n'
            f'        {{"score": 1, "description": "Uses linear algebra concepts far above (graduate level) or below (basic arithmetic) the user\'s intermediate level, making explanation inaccessible or patronizing"}},\n'
            f'        {{"score": 3, "description": "Mostly appropriate for intermediate level but some inconsistencies - occasionally too advanced or too basic for stated comfort level"}},\n'
            f'        {{"score": 5, "description": "Perfectly calibrated to intermediate linear algebra comfort - uses appropriate terminology, assumes right background knowledge, explains concepts at ideal complexity level"}}\n'
            f'    ]\n'
            f"}}\n\n"
            f"Format as JSON:\n"
            f"{{\n"
            f'    "criteria": [\n'
            f'        {{\n'
            f'            "preference": "Exact preference name",\n'
            f'            "description": "How well the response matches the user\'s technical level for this specific concept in this problem",\n'
            f'            "weight": 0.XX,\n'
            f'            "levels": [\n'
            f'                {{"score": 1, "description": "Technical level far above or below user\'s stated comfort level"}},\n'
            f'                {{"score": 3, "description": "Mostly appropriate technical level but some inconsistencies"}},\n'
            f'                {{"score": 5, "description": "Perfectly calibrated to user\'s stated comfort level"}}\n'
            f'            ]\n'
            f'        }}\n'
            f'    ]\n'
            f"}}\n"
        )

        result = self._call_llm(prompt)
        criteria = result.get("criteria", [])

        # Debug: Check if criteria is the expected type
        if not isinstance(criteria, list):
            print(f"ERROR: Expected criteria to be a list, got {type(criteria)}: {criteria}")
            print(f"Full result: {result}")
            return []

        return criteria

    def _generate_personal_rubric_criteria(self, personal_prefs: Dict[str, Dict], base_context: str, problem_text: str, problem_type: str, total_importance: int, persona: Dict) -> List[Dict]:
        """Generate rubric criteria specifically for personal/stylistic preferences."""

        prompt = (
            f"You are creating evaluation criteria for PERSONAL preferences - how well an AI response adapts to a user's unique learning style, communication preferences, and personal context.\n\n"
            f"{base_context}"
            f"PERSONAL PREFERENCES:\n"
        )

        for pref_name, pref_data in personal_prefs.items():
            weight = pref_data.get('local_importance', 1) / total_importance
            prompt += f"- {pref_name}: {pref_data.get('value', 'N/A')} (importance: {pref_data.get('local_importance', 1)}, weight: {weight:.3f})\n"
            prompt += f"  Why this matters: {pref_data.get('justification', 'No justification')}\n\n"

        prompt += (
            f"PERSONAL RUBRIC GUIDELINES:\n"
            f"- Focus on LEARNING STYLE ADAPTATION: Does the response match how this person prefers to learn?\n"
            f"- Evaluate COMMUNICATION STYLE: Is the tone, formality, and interaction style appropriate?\n"
            f"- Assess CONTEXTUAL RELEVANCE: Are examples, analogies, and references meaningful to this person?\n"
            f"- Check MOTIVATIONAL ELEMENTS: Does the response connect to what drives this person?\n\n"
            f"Consider this person's background: {persona.get('demographics', {}).get('occupation', 'Unknown occupation')}, "
            f"hobbies: {', '.join(persona.get('demographics', {}).get('hobbies', []))}, "
            f"and learning style: {persona.get('educational_profile', {}).get('cognitive_features', {}).get('learning_style', 'Not specified')}\n\n"
            f"CRITICAL: Create exactly ONE criterion for each personal preference listed above. Each criterion must use the EXACT preference name.\n\n"
            f"For each personal preference, create criteria that measure how well the AI response adapts to this specific person's unique learning needs and preferences for this {problem_type}.\n\n"
            f"EXAMPLE PERSONAL CRITERION:\n"
            f"{{\n"
            f'    "preference": "Engineering Examples from Work Experience",\n'
            f'    "description": "How well the response incorporates relevant engineering examples that connect to this user\'s mechanical engineering background when explaining this physics problem",\n'
            f'    "weight": 0.20,\n'
            f'    "levels": [\n'
            f'        {{"score": 1, "description": "No engineering examples provided, or examples from completely irrelevant fields that don\'t connect to user\'s mechanical engineering experience"}},\n'
            f'        {{"score": 3, "description": "Includes some engineering examples but they\'re generic or don\'t clearly connect to the physics concepts in this specific problem"}},\n'
            f'        {{"score": 5, "description": "Provides multiple relevant mechanical engineering examples that directly illuminate the physics concepts and clearly resonate with the user\'s professional experience"}}\n'
            f'    ]\n'
            f"}}\n\n"
            f"Format as JSON:\n"
            f"{{\n"
            f'    "criteria": [\n'
            f'        {{\n'
            f'            "preference": "Exact preference name",\n'
            f'            "description": "How well the response adapts to this user\'s specific personal learning preference for this problem",\n'
            f'            "weight": 0.XX,\n'
            f'            "levels": [\n'
            f'                {{"score": 1, "description": "No adaptation to personal preference or completely inappropriate approach"}},\n'
            f'                {{"score": 3, "description": "Some adaptation to personal preference but generic or inconsistent"}},\n'
            f'                {{"score": 5, "description": "Excellent adaptation that perfectly matches the user\'s personal preference"}}\n'
            f'            ]\n'
            f'        }}\n'
            f'    ]\n'
            f"}}\n"
        )

        result = self._call_llm(prompt)
        criteria = result.get("criteria", [])

        # Debug: Check if criteria is the expected type
        if not isinstance(criteria, list):
            print(f"ERROR: Expected criteria to be a list, got {type(criteria)}: {criteria}")
            print(f"Full result: {result}")
            return []

        return criteria

    def _generate_other_rubric_criteria(self, other_prefs: Dict[str, Dict], base_context: str, problem_text: str, problem_type: str, total_importance: int) -> List[Dict]:
        """Generate rubric criteria for preferences that don't fit expertise or personal categories."""

        prompt = (
            f"You are creating evaluation criteria for GENERAL preferences that don't fit into expertise or personal categories.\n\n"
            f"{base_context}"
            f"OTHER PREFERENCES:\n"
        )

        for pref_name, pref_data in other_prefs.items():
            weight = pref_data.get('local_importance', 1) / total_importance
            prompt += f"- {pref_name}: {pref_data.get('value', 'N/A')} (importance: {pref_data.get('local_importance', 1)}, weight: {weight:.3f})\n"
            prompt += f"  Why this matters: {pref_data.get('justification', 'No justification')}\n\n"

        prompt += (
            f"GENERAL RUBRIC GUIDELINES:\n"
            f"- Make criteria PROBLEM-SPECIFIC: Reference the actual problem content\n"
            f"- Make criteria OBJECTIVELY MEASURABLE: Clear, observable behaviors\n"
            f"- Focus on PERSONALIZATION QUALITY: How well the response adapts to user needs\n"
            f"- Be CONCRETE AND SPECIFIC: Avoid vague terms\n\n"
            f"CRITICAL: Create exactly ONE criterion for each preference listed above. Each criterion must use the EXACT preference name.\n\n"
            f"For each preference, create criteria that measure how well the AI response addresses this specific aspect for this {problem_type}.\n\n"
            f"Format as JSON:\n"
            f"{{\n"
            f'    "criteria": [\n'
            f'        {{\n'
            f'            "preference": "Exact preference name",\n'
            f'            "description": "What specifically to evaluate for this preference on this problem",\n'
            f'            "weight": 0.XX,\n'
            f'            "levels": [\n'
            f'                {{"score": 1, "description": "Poor adaptation to this preference"}},\n'
            f'                {{"score": 3, "description": "Adequate adaptation to this preference"}},\n'
            f'                {{"score": 5, "description": "Excellent adaptation to this preference"}}\n'
            f'            ]\n'
            f'        }}\n'
            f'    ]\n'
            f"}}\n"
        )

        result = self._call_llm(prompt)
        criteria = result.get("criteria", [])

        # Debug: Check if criteria is the expected type
        if not isinstance(criteria, list):
            print(f"ERROR: Expected criteria to be a list, got {type(criteria)}: {criteria}")
            print(f"Full result: {result}")
            return []

        return criteria


    def create_personalized_benchmark(self) -> str:
        """Generate the complete personalized benchmark."""

        # Step 1: Load or create persona library
        self.load_persona_library()

        # Step 2: Load or generate dataset-level preferences
        self.load_all_preferences(self.all_prefs_filepath)
        self.load_dataset_preferences(self.dataset_preference_filepath)

        # Load existing progress if available
        existing_problem_map = {}  # Maps problem_index -> set of persona_ids
        problem_ids = []

        if os.path.exists(self.output_file):
            print(f"Found existing output file: {self.output_file}")
            print("Loading existing progress...")

            with open(self.output_file, 'r') as f:
                self.personalized_problems = []
                valid_problems = 0
                invalid_problems = 0

                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') == 'personalized_problem':
                            problem_id = data.get('problem_id', '')

                            # Validate the problem structure
                            if self._validate_existing_problem(data):
                                self.personalized_problems.append(data)
                                problem_ids.append(problem_id)

                                # Extract problem index and persona ID from problem_id
                                problem_index, persona_id = self._parse_problem_id(problem_id)
                                if problem_index is not None and persona_id is not None:
                                    if problem_index not in existing_problem_map:
                                        existing_problem_map[problem_index] = set()
                                    existing_problem_map[problem_index].add(persona_id)
                                    valid_problems += 1
                                else:
                                    print(f"Warning: Could not parse problem_id '{problem_id}' on line {line_num}")
                                    invalid_problems += 1
                            else:
                                print(f"Warning: Invalid problem structure on line {line_num}, skipping")
                                invalid_problems += 1
                        # Skip metadata and other types silently
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        invalid_problems += 1
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}")
                        invalid_problems += 1

                print(f"Loaded {valid_problems} valid existing problems")
                if invalid_problems > 0:
                    print(f"Skipped {invalid_problems} invalid/corrupted entries")

                # Print restart summary
                total_existing = sum(len(personas) for personas in existing_problem_map.values())
                problems_with_data = len(existing_problem_map)
                print(f"Restart summary: {total_existing} persona-problem pairs across {problems_with_data} problems")

        else:
            print(f"No existing output file found. Starting fresh generation.")
            self.personalized_problems = []

            # Create new file with metadata
            with open(self.output_file, 'w') as f:
                metadata = {
                    "type": "metadata",
                    "dataset": self.dataset_name,
                    "general_preferences": self.general_preferences,
                    "total_personas_in_library": len(self.persona_library),
                    "fixed_persona_library": True,
                    "personas_per_problem": self.num_personas_per_problem,
                    "sample_size": self.sample_size,
                    "multi_model_generation": len(self.llm_clients) > 1,
                    "llm_clients": list(self.llm_clients.keys()) if len(self.llm_clients) > 1 else ["single_model"]
                }
                f.write(json.dumps(metadata) + '\n')

        # Step 3: Iterate over problems in the benchmark
        total_needed = 0
        total_existing = 0

        for i, problem in enumerate(tqdm(self.problems)):
            # Check if we should stop due to time limit
            if self._should_stop_generation():
                elapsed_time = time.time() - self.start_time
                print(f"\nStopping generation after {elapsed_time:.1f} seconds (stop_after={self.stop_after})")
                print(f"Completed {i}/{len(self.problems)} problems before stopping")
                break

            # Get existing personas for this problem
            existing_personas_for_problem = existing_problem_map.get(i, set())
            personas_needed = self.num_personas_per_problem - len(existing_personas_for_problem)
            total_existing += len(existing_personas_for_problem)
            total_needed += personas_needed

            if personas_needed <= 0:
                if self.verbose:
                    print(f"Problem {i}: Complete - already has {len(existing_personas_for_problem)} personas")
                continue

            print(f"Problem {i}: Need {personas_needed} more personas (have {len(existing_personas_for_problem)})")
            if self.verbose and existing_personas_for_problem:
                print(f"  Existing personas: {sorted(list(existing_personas_for_problem))}")

            # Step 4: Generate personas for this problem with retry logic
            successful_personas = 0
            persona_attempts = 0
            max_persona_attempts = personas_needed * 5  # Allow up to 5x attempts to find good personas

            while successful_personas < personas_needed and persona_attempts < max_persona_attempts:
                # Check if we should stop due to time limit
                if self._should_stop_generation():
                    elapsed_time = time.time() - self.start_time
                    print(f"\nStopping generation after {elapsed_time:.1f} seconds (stop_after={self.stop_after})")
                    break

                # Sample a single persona that hasn't been used for this problem
                available_personas = [p for p in self.persona_library.values()
                                    if self._generate_persona_id(p) not in existing_personas_for_problem]

                if not available_personas:
                    print(f"Error: No more available personas for problem {i}")
                    break

                persona = random.choice(available_personas)
                persona_attempts += 1

                # Ensure persona has an ID
                if 'id' not in persona:
                    persona['id'] = self._generate_persona_id(persona)

                problem_id = f"problem_{i}_{persona['id']}"

                if problem_id in problem_ids:
                    continue  # Skip if already processed

                print(f"  Attempting persona {persona['name']} ({persona['id']})")

                # Step 5: Try to generate personalized problem with retry logic
                success = self._generate_single_personalized_problem(problem, persona, i, problem_id)

                if success:
                    successful_personas += 1
                    existing_personas_for_problem.add(persona['id'])
                    problem_ids.append(problem_id)
                    print(f"  ✓ Successfully generated problem for {persona['name']}")
                else:
                    print(f"  ✗ Failed to generate problem for {persona['name']} after retries")

            if successful_personas < personas_needed:
                print(f"Warning: Only generated {successful_personas}/{personas_needed} personas for problem {i}")

        # Print final summary
        total_generated = total_needed - sum(1 for i in range(len(self.problems))
                                           if len(existing_problem_map.get(i, set())) +
                                           (self.num_personas_per_problem - len(existing_problem_map.get(i, set())))
                                           > len(existing_problem_map.get(i, set())))
        print(f"\nGeneration Summary:")
        print(f"- Total problems: {len(self.problems)}")
        print(f"- Target personas per problem: {self.num_personas_per_problem}")
        print(f"- Total target scenarios: {len(self.problems) * self.num_personas_per_problem}")
        print(f"- Pre-existing scenarios: {total_existing}")
        print(f"- Scenarios needed: {total_needed}")
        print(f"- Total scenarios after generation: {len([p for p in self.personalized_problems if p.get('type') == 'personalized_problem'])}")

    def _generate_single_personalized_problem(self, problem: Dict, persona: Dict, problem_index: int, problem_id: str, max_retries: int = 3) -> bool:
        """Generate a single personalized problem with retry logic."""
        for attempt in range(max_retries):
            try:
                # Generate personalized preferences for this problem-persona combination
                persona_preferences, success1 = self.generate_persona_preferences_for_problem(
                    problem, persona, problem_index, max_tries=3
                )

                if not success1 or not persona_preferences:
                    print(f"    Attempt {attempt + 1}: Failed to generate preferences")
                    continue

                # Generate evaluation rubric based on the persona's preferences
                rubric, success2 = self.generate_evaluation_rubric(
                    problem, persona_preferences, persona, problem_id, try_idx=0, max_tries=3
                )

                if not success2 or not rubric:
                    print(f"    Attempt {attempt + 1}: Failed to generate rubric")
                    continue

                # If we get here, everything succeeded
                # Update persona preferences tracking
                self._update_persona_preferences_tracking(persona['id'], persona, problem, persona_preferences, problem_index)

                # Process the original problem to make it JSON serializable
                processed_problem = self._process_problem_for_serialization(problem, problem_id, problem_index)

                # Create personalized problem entry
                personalized_problem = {
                    "type": "personalized_problem",
                    "problem_id": problem_id,
                    "original_problem": processed_problem,
                    "persona": {k: v for k, v in persona.items() if k != "accumulated_preferences"},
                    "persona_preferences": persona_preferences,
                    "evaluation_rubric": rubric,
                    "generation_success": True
                }

                # Save the personalized problem to file
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(personalized_problem) + '\n')

                # Also store in memory for potential use
                self.personalized_problems.append(personalized_problem)

                # Update usage count
                persona["usage_count"] = persona.get("usage_count", 0) + 1

                return True

            except Exception as e:
                print(f"    Attempt {attempt + 1}: Exception occurred: {e}")
                continue

        return False

        # Update persona library with usage counts
        self._save_persona_library_to_file()

        # Calculate total scenarios
        total_scenarios = len([p for p in self.personalized_problems if p.get('type') == 'personalized_problem'])

        print(f"Benchmark generation complete!")
        print(f"- {len(self.personalized_problems)} personalized problems saved to {self.output_file}")
        print(f"- {total_scenarios} total evaluation scenarios created")
        print(f"- {len(self.persona_library)} fixed personas in library")
        print(f"- {self.num_personas_per_problem} personas per problem")
        print(f"- Multi-model generation: {len(self.llm_clients) > 1} ({list(self.llm_clients.keys())})")
        print(f"Shared persona library (reusable across tasks): {self.persona_library_filepath}")
        print(f"Shared persona preferences tracking (builds across tasks): {self.persona_preferences_filepath}")
        return self.output_file

    def _validate_existing_problem(self, data: Dict) -> bool:
        """Validate that an existing problem entry has all required fields."""
        try:
            required_fields = ['problem_id', 'original_problem', 'persona', 'persona_preferences', 'evaluation_rubric']
            for field in required_fields:
                if field not in data:
                    return False

            # Additional validation
            if not isinstance(data.get('persona_preferences'), dict):
                return False
            if not isinstance(data.get('evaluation_rubric'), dict):
                return False
            if not data.get('problem_id'):
                return False

            return True
        except Exception:
            return False

    def _parse_problem_id(self, problem_id: str) -> Tuple[Optional[int], Optional[str]]:
        """Parse a problem_id to extract problem_index and persona_id.

        Expected format: 'problem_{index}_{persona_id}'
        Returns: (problem_index, persona_id) or (None, None) if parsing fails
        """
        try:
            if not problem_id.startswith('problem_'):
                return None, None

            # Remove 'problem_' prefix
            remainder = problem_id[8:]  # len('problem_') = 8

            # Find the first underscore after 'problem_'
            underscore_idx = remainder.find('_')
            if underscore_idx == -1:
                return None, None

            # Extract problem index and persona ID
            problem_index_str = remainder[:underscore_idx]
            persona_id = remainder[underscore_idx + 1:]

            # Validate problem index is a number
            try:
                problem_index = int(problem_index_str)
            except ValueError:
                return None, None

            # Validate persona_id is not empty
            if not persona_id:
                return None, None

            return problem_index, persona_id

        except Exception:
            return None, None

    def _load_persona_preferences_data(self) -> Dict:
        """Load existing persona preferences data or initialize empty structure."""
        if os.path.exists(self.persona_preferences_filepath):
            with open(self.persona_preferences_filepath, 'r') as f:
                return json.load(f)
        return {}

    def _save_persona_preferences_data(self):
        """Save persona preferences data to file."""
        with open(self.persona_preferences_filepath, 'w') as f:
            json.dump(self.persona_preferences_data, f, indent=2)

    def _generate_persona_id(self, persona_data: Dict) -> str:
        """Generate a unique ID for a persona based on name and description."""
        import hashlib

        # Extract last name and minimal description
        name = persona_data.get('name', 'Unknown')
        description = persona_data.get('minimal_necessary_description', '')

        # Create a hash from name and description
        combined_text = f"{name}_{description}"
        hash_value = hashlib.md5(combined_text.encode()).hexdigest()[:8]

        # Extract last name for readability
        last_name = name.split()[-1] if ' ' in name else name

        return f"{last_name.lower()}_{hash_value}"

    def _update_persona_preferences_tracking(self, persona_id: str, persona: Dict, problem: Dict, preferences: Dict, problem_index: int):
        """Update the persona preferences tracking data."""
        if persona_id not in self.persona_preferences_data:
            self.persona_preferences_data[persona_id] = {
                "persona_info": {
                    "name": persona.get('name', 'Unknown'),
                    "minimal_description": persona.get('minimal_necessary_description', ''),
                    "demographics": persona.get('demographics', {}),
                    "id": persona_id
                },
                "problems_and_preferences": []
            }

        # Extract problem identifier
        problem_text = ""
        if 'problem' in problem:
            problem_text = problem.get('problem', '')[:100] + "..." if len(problem.get('problem', '')) > 100 else problem.get('problem', '')
        elif 'question' in problem:
            problem_text = problem.get('question', '')[:100] + "..." if len(problem.get('question', '')) > 100 else problem.get('question', '')
        else:
            problem_text = str(problem)[:100] + "..." if len(str(problem)) > 100 else str(problem)

        problem_entry = {
            "problem_index": problem_index,
            "problem_text": problem_text,
            "problem_id": problem.get('id', f"problem_{problem_index}"),
            "preferences": preferences,
        }

        self.persona_preferences_data[persona_id]["problems_and_preferences"].append(problem_entry)
        self._save_persona_preferences_data()

    def _extract_options_from_field(self, problem: Dict, field_name: str) -> str:
        """Extract and format options/choices from a specific field in the problem dictionary."""
        options_text = ""

        if self.dataset_name == "allenai/social_i_qa":
            for key in ["answerA", "answerB", "answerC"]:
                if key in field_value:
                    k = key.replace("answer", "")
                    options_text += f"{k}. {field_value[key]}\n"
            return options_text.strip()

        if field_name not in problem:
            return options_text

        field_value = problem[field_name]


        if self.dataset_name == "tau/commonsense_qa":
            for key, value in zip(field_value["label"], field_value["text"]):
                options_text += f"{key}. {value}\n"
            return options_text.strip()
        # Handle dict format (e.g., {'A': 'answer1', 'B': 'answer2'})
        elif isinstance(field_value, dict):
            for key, value in field_value.items():
                options_text += f"{key}. {value}\n"
        # Handle list format (e.g., ['answer1', 'answer2', 'answer3'])
        elif isinstance(field_value, list):
            for i, option in enumerate(field_value):
                options_text += f"{chr(65+i)}. {option}\n"  # A, B, C, D...
        # Handle string format (though unusual for choices)
        elif isinstance(field_value, str):
            options_text = field_value

        return options_text.strip()

    def _extract_options_text(self, problem: Dict) -> str:
        """Legacy method: Extract and format options/choices from a problem dictionary.

        This method is kept for backward compatibility with medical dataset transformation.
        """
        # Try common choice field names
        for field_name in ['options', 'choices']:
            if field_name in problem:
                return self._extract_options_from_field(problem, field_name)
        return ""

    def _extract_problem_content(self, problem: Dict) -> Tuple[str, str, Optional[Union[Image.Image, str]]]:
        """Extract problem text, domain type, and image from a problem dictionary using configurable fields."""
        image = None

        # Extract image if configured and present
        if self.image_field and self.image_field in problem and problem[self.image_field] is not None:
            image = problem[self.image_field]

        # Check if this is a MultiMedQA dataset
        is_multimedqa = self.dataset_name in ["openlifescienceai/medqa", "medqa", "multimedqa"]

        # Extract problem text by concatenating configured fields
        problem_parts = []
        for field in self.problem_fields:
            if field in problem and problem[field]:
                problem_parts.append(str(problem[field]))

        if problem_parts:
            problem_text = "\n\n".join(problem_parts)
        else:
            # Fallback to string representation if no configured fields found
            problem_text = str(problem)

        # Determine domain based on dataset or field types
        if is_multimedqa:
            # Transform medical questions into patient communication scenarios
            problem_text = self._transform_medical_question_to_caretaker_scenario(problem, problem_text)
            domain = "medical consultation as a patient caretaker"
        elif any(field in ['problem', 'math_problem'] for field in self.problem_fields):
            domain = "math problem"
        elif any(field in ['question', 'query'] for field in self.problem_fields):
            domain = "question"
        else:
            domain = "task"

        # Add choices if configured and present (except for medical datasets which handle this internally)
        if not is_multimedqa and self.choices_field and self.choices_field in problem:
            options_text = self._extract_options_from_field(problem, self.choices_field)
            if options_text:
                problem_text += f"\n\nAnswer choices:\n{options_text}"

        return problem_text, domain, image

    def _transform_medical_question_to_caretaker_scenario(self, problem: Dict, original_question: str) -> str:
        """Transform a medical question into a patient caretaker communication scenario."""

        # Extract additional context that might be available in the medical dataset
        context_parts = []

        # Use the helper method to extract options consistently
        options_text = self._extract_options_text(problem)
        if options_text:
            context_parts.append(f"Answer choices:\n{options_text}")

        # Look for additional context fields
        for field in ['context', 'patient_info', 'case_description', 'clinical_context']:
            if field in problem and problem[field]:
                context_parts.append(f"Additional context: {problem[field]}")

        # Create the caretaker scenario prompt
        caretaker_prompt = (
            f"I am the caretaker/family member of a patient, and I need help understanding their medical situation. "
            f"Please answer my question about the patient given the provided medical information. "
            f"I want you to explain this in a way that helps me understand what's happening with my loved one's health.\n\n"
            f"My question: {original_question}"
        )

        # Add context if available
        if context_parts:
            caretaker_prompt += f"\n\nProvided medical information:\n" + "\n\n".join(context_parts)

        return caretaker_prompt

    def _save_image_to_file(self, image: Union[Image.Image, str], problem_index: int, image_dir: str) -> str:
        """Save PIL Image to file and return relative path. Only saves one copy per problem."""
        if isinstance(image, str):
            # If it's already a file path, just return it
            return image
        elif hasattr(image, 'save'):
            # Create images directory if it doesn't exist
            os.makedirs(image_dir, exist_ok=True)

            # Generate filename using only problem index
            image_filename = f"problem_{problem_index}_image.png"
            image_path = os.path.join(image_dir, image_filename)

            # Only save if the file doesn't already exist
            if not os.path.exists(image_path):
                # Convert to RGB if necessary and save
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path, format="PNG")

            # Return relative path
            return os.path.relpath(image_path, self.output_dir)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _process_problem_for_serialization(self, problem: Dict, problem_id: str, problem_index: int) -> Dict:
        """Process a problem dictionary to make it JSON serializable by saving images to files."""
        # Create a copy to avoid modifying the original
        processed_problem = problem.copy()

        # Handle image field
        if 'image' in processed_problem and processed_problem['image'] is not None:
            image_dir = os.path.join(self.output_dir, "images")
            image_path = self._save_image_to_file(processed_problem['image'], problem_index, image_dir)
            processed_problem['image'] = image_path

        return processed_problem

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required configuration sections
    required_sections = ['dataset', 'personas', 'system', 'preference_generation', 'evaluation', 'paths']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")

    # Validate required fields within sections
    required_fields = {
        'dataset': ['name', 'split', 'sample_size'],
        'personas': ['per_problem', 'initial_num_personas', 'new_persona_prob'],
        'system': ['random_seed'],
        'preference_generation': ['min_dimensions_per_problem'],
        'evaluation': ['weight_tolerance'],
        'paths': ['output_dir', 'output_file', 'persona_library_filepath']
    }

    for section, fields in required_fields.items():
        missing_fields = [field for field in fields if field not in config[section]]
        if missing_fields:
            raise ValueError(f"Missing required fields in section '{section}': {missing_fields}")

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a personalized benchmark from a dataset")
    parser.add_argument("--config", type=str, default="src/config/benchmark_generator_logiqa.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Verbose output")
    parser.add_argument("--parallel", action="store_true", default=False,
                       help="Use parallel processing for faster generation")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: from config or 4)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Override the number of source problems")
    parser.add_argument("--personas-per-problem", type=int, default=None,
                       help="Override the number of personas paired with each problem")
    parser.add_argument("--initial-personas", type=int, default=None,
                       help="Override the initial shared persona-library size")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override the output directory")
    parser.add_argument("--model", type=str, default=None,
                       help="Use one model instead of the config's multi-model setup")
    parser.add_argument("--api-account", type=str, default=None,
                       help="Override the API account for configured model clients")
    parser.add_argument("--api-info", type=str, default=None,
                       help="Override the API account YAML path")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override the generation random seed")

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    if args.sample_size is not None:
        config["dataset"]["sample_size"] = args.sample_size
    if args.personas_per_problem is not None:
        config["personas"]["per_problem"] = args.personas_per_problem
    if args.initial_personas is not None:
        config["personas"]["initial_num_personas"] = args.initial_personas
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["system"]["random_seed"] = args.seed

    if args.model:
        model_config = dict(config.get("llm_config", {}))
        model_config["model"] = args.model
        model_config["model_kwargs"] = dict(model_config.get("model_kwargs", {}))
        config["llm_configs"] = {"default": model_config}
        config.setdefault("model", {})["name"] = args.model

    for llm_config in config.get("llm_configs", {}).values():
        model_kwargs = llm_config.setdefault("model_kwargs", {})
        if args.api_account:
            model_kwargs["api_account"] = args.api_account
        if args.api_info:
            model_kwargs["api_info"] = args.api_info

    # Override workers if specified on command line
    if args.workers is not None:
        if "parallelization" not in config:
            config["parallelization"] = {}
        config["parallelization"]["num_workers"] = args.workers

    print(f"Loaded configuration from {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Sample size: {config['dataset']['sample_size']}")
    print(f"Total personas: {config['personas']['initial_num_personas']}")
    print(f"Personas per problem: {config['personas']['per_problem']}")
    print(f"Multi-model setup: {len(config.get('llm_configs', {})) > 1}")
    if len(config.get('llm_configs', {})) > 1:
        print(f"LLM clients: {list(config.get('llm_configs', {}).keys())}")
    else:
        print(f"Model: {config.get('model', {}).get('name', 'single-model')}")

    if args.parallel:
        num_workers = config.get("parallelization", {}).get("num_workers", 4)
        print(f"Parallel mode: ENABLED ({num_workers} workers)")
    else:
        print(f"Parallel mode: DISABLED (sequential)")
    print()

    # Create generator with loaded configuration
    generator = PersonalizedBenchmarkGenerator(config)

    # Set verbose mode
    generator.set_verbose(args.verbose)

    # Generate the benchmark
    if args.parallel:
        generator.create_personalized_benchmark_parallel()
    else:
        generator.create_personalized_benchmark()
