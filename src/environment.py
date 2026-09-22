import os
import json
import random
import argparse
import base64
import io
import fcntl
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from PIL import Image
import yaml
from llm.api_client import LLMClient
from benchmark_data import normalize_problem
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.utils import *

# Hardcoded mapping of user prompt keys to file paths
USER_PROMPT_MAPPING = {
    "passive_nostop": "src/prompts/passive_user_nostop.txt",
    "passive": "src/prompts/passive_user.txt",
    "collaborative": "src/prompts/collaborative_user.txt"
}

class UserSimulator:
    def __init__(self, persona: Dict, persona_profile: str, config: Dict, user_prompt_template: str, user_type: str = "default"):
        """
        Initialize the user simulator.

        Args:
            persona: Persona data with preferences.
            persona_profile: A string containing the detailed persona profile.
            config: The main configuration dictionary, used to get evaluator LLM settings.
            user_prompt_template: The string template for the user prompt.
        """

        evaluator_llm_config = config.get("evaluator_llm_config", {})
        if evaluator_llm_config.get("model", "") == "" and os.path.exists(evaluator_llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml")):
            with open(evaluator_llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml"), "r") as f:
                api_info = yaml.safe_load(f)
            api_account = evaluator_llm_config.get("model_kwargs", {}).get("api_account", "openai")
            evaluator_llm_config["model"] = api_info.get(api_account, {"model_name": api_account}).get("model_name", "gpt-4o")

        # print(f"UserSimulator using evaluator config: {evaluator_llm_config}")
        self.client = LLMClient(config=evaluator_llm_config)
        self.model_name = evaluator_llm_config["model"]

        self.persona = persona
        self.persona_profile = persona_profile
        self.preference_expressed = set()
        self.interaction_history = []
        self.user_prompt_template = user_prompt_template
        self.user_type = user_type


    def _is_requesting_final_answer(self, response: str) -> bool:
        """Determines if the assistant is trying to provide a final answer instead of asking a question.

        For the new structured format, this checks if the ACTION is 'answer_question'.
        Falls back to heuristic parsing for responses that don't follow the structured format.
        """
        # First, try to parse the structured format
        if "###ACTION###" in response:
            # Extract the action from the structured response
            try:
                action_start = response.find("###ACTION###:")
                if action_start != -1:
                    # Find the end of the action line
                    action_line = response[action_start:].split('\n')[0]
                    # Extract the action value (handle both quoted and unquoted formats)
                    action_content = action_line.split("###ACTION###:")[-1].strip()
                    # Remove quotes and brackets if present
                    action_content = action_content.strip("[]'\"")

                    # If action is 'answer_question', then it's providing a final answer
                    return action_content.lower() == 'final_answer'
            except (IndexError, AttributeError):
                # If parsing fails, fall back to heuristic method
                pass

        # Fallback to original heuristic method for unstructured responses
        response_lower = response.lower()
        # Keywords that suggest a final answer is being given
        final_answer_keywords = ["here is the solution", "the answer is", "finally, we get", "in conclusion"]
        # Phrases that suggest a question is being asked
        question_keywords = ["what do you think", "how do you", "would you like", "are you familiar"]

        if any(keyword in response_lower for keyword in final_answer_keywords):
            # If it sounds like a final answer, we check if it ALSO sounds like a question.
            # This handles cases like "Here is the solution, would you like me to explain it further?"
            if not any(q_keyword in response_lower for q_keyword in question_keywords) and not response.strip().endswith("?"):
                 return True
        return False

    def respond_to_assistant(self, assistant_response: str) -> Dict:
        """Generate a response to the assistant's question using the internal evaluator client."""
        # Track interaction
        self.interaction_history.append({"role": "assistant", "content": assistant_response})
        # Note: Verbose output removed - args.verbose was not defined
        # if args.verbose:
        #     print("ASSISTANT: ", assistant_response)

        # Check if the assistant has stopped asking questions and is giving a final answer.
        if self._is_requesting_final_answer(assistant_response):
            # The assistant jumped to a final answer prematurely.
            # We can treat this as the end of the conversational phase.
            return {"response": "###STOP###", "terminate": True}

        # Focus on the question and get the user to respond
        prompt = self.user_prompt_template.format(
            persona_profile=self.persona_profile,
            chat_history=parse_messages(self.interaction_history, strip_sys_prompt=True),
            terminal_signal="###STOP###" if "nostop" not in self.user_type else "",
        )

        msg = [{"role": "user", "content": prompt}]
        # Call LLM to generate response - let exceptions bubble up
        response = self.client.chat(
            messages=msg,
            temperature=0.4,
            response_format="json",
            regenerate_if_unfinished=True
        )

        full_response = response["response_text"]
        if isinstance(full_response, str):
            full_response = extract_json(full_response)

        if isinstance(full_response, dict):
            user_response = str(full_response.get('response', "I don't understand."))
        else:
            user_response = str(full_response)

        # Note: Verbose output removed - args.verbose was not defined
        # if args.verbose:
        #     print("USER:", user_response)

        # Track which preferences this likely expressed
        self._update_expressed_preferences(assistant_response, user_response)

        # user_response += "\n\nDecide whether to ask the user for more information or to proceed with the answering the question. Respond in the same format:\n"
        # user_response += "###ACTION###: ['ask_question' or 'final_answer']\n"
        # user_response += "###RESPONSE###: [clarifying question or full answer]\n"
        # Add to history
        self.interaction_history.append({"role": "user", "content": user_response})

        return {
                "response": user_response,
                "terminate": self._is_user_done(user_response) if "nostop" not in self.user_type else False
            }


    def _is_user_done(self, user_message: str) -> bool:
        """
        Returns True if the user message indicates they want to end the conversation,
        i.e., it contains '###STOP###' anywhere. Otherwise, returns False.
        """
        # Ensure user_message is a string to avoid TypeError
        if not isinstance(user_message, str):
            user_message = str(user_message)
        return "###STOP###" in user_message


    def _update_expressed_preferences(self, question: str, response: str) -> None:
        """Update which preferences have been expressed in the conversation."""
        prompt = f"""
        Analyze which preferences were likely detected by the assistant in this interaction:

        Assistant: "{question}"
        User: "{response}"

        Available preferences:
        {json.dumps(list(self.persona.get('preferences', {}).keys()), indent=2)}

        Determine which preferences were expressed, either:
        1. EXPLICITLY: The user directly stated a preference
        2. IMPLICITLY: The user's phrasing, focus, or examples revealed a preference
        3. INDIRECTLY: The user responded in a way that suggests a preference

        For each identified preference, provide:
        1. The preference dimension name
        2. Whether it was expressed explicitly, implicitly, or indirectly
        3. The specific evidence from the user's response (quote the exact text)
        4. Confidence level (high, medium, low)

        Only include preferences that were actually expressed in this specific exchange - do NOT include preferences that might be inferred but aren't evidenced in the text.

        Format your response as JSON:
        {{
            "expressed_preferences": [
                {{
                    "preference": "Visual Aid Preference",
                    "expression_type": "implicit",
                    "evidence": "I'm having trouble picturing how this works",
                    "confidence": "high"
                }},
                ...
            ]
        }}
        """

        # Let exceptions bubble up - will be caught by main evaluation loop
        result = self.client.chat(
            messages=[
                {"role": "system", "content": "You analyze conversations to detect expressed preferences. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format="json",
            temperature=0.1,
            regenerate_if_unfinished=True
        )

        expressed = result.get("expressed_preferences", [])
        for pref_data in expressed:
            self.preference_expressed.add(pref_data.get("preference", ""))

    def evaluate_final_response(self, final_response: str, rubric: Dict, persona_preferences: Dict, problem: Dict) -> Dict:
        """Evaluate the final response against each rubric criterion individually."""

        criteria = rubric.get("evaluation_criteria", [])
        all_scores = {}
        if not criteria:
            return {
                "criteria_scores": {}, "weighted_score": 0, "interaction_turns": len(self.interaction_history) // 2,
                "preferences_expressed": list(self.preference_expressed)
            }

        for criterion in criteria:
            prompt = f"""
            You are an expert evaluation specialist using a standardized rubric to assess personalized responses. Provide a precise, evidence-based assessment of how well this response meets the specified criterion.

            CRITERION TO EVALUATE:
            {criterion['preference']}
            Description: {criterion['description']}

            PERFORMANCE LEVELS:
            {json.dumps(criterion.get('levels', []), indent=2)}

            RESPONSE TO EVALUATE:
            "{final_response}"

            USER CONTEXT:
            Preference value: {persona_preferences.get(criterion['preference'])['value']}
            Context: {persona_preferences.get(criterion['preference'])['justification']}

            EVALUATION INSTRUCTIONS:
            1. Carefully compare the response against each performance level description.
            2. Identify specific evidence in the response that matches level descriptions.
            3. Determine the exact score (1-5) that best represents the response quality for this criterion.
            4. Provide a detailed justification referencing specific elements of the response.
            5. Be objective and consistent - apply the same standards across all evaluations.
            6. Consider the user's stated preferences.
            7. Provide PRECISE REASONING that another evaluator could follow to reach the same conclusion.

            Format your evaluation as a JSON object:
            {{
                "score": X (integer between 1 and 5),
                "justification": "The response includes [specific example 1] and [specific example 2], which aligns with the user's preference for [quote performance level] to the extent of X. However, it lacks [specific missing element] that would be needed for a score of 5."
            }}
            """

            # Let exceptions bubble up - will be caught by main evaluation loop
            result = self.client.chat(
                messages=[
                    {"role": "system", "content": "You evaluate responses against specific criteria. Output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json",
                temperature=0.1,
                regenerate_if_unfinished=True
            )

            evaluation = result.get("response_text", {})
            if isinstance(evaluation, str):
               evaluation = extract_json(evaluation)

            all_scores[criterion['preference']] = {
                "score": evaluation.get("score", 1),
                "justification": evaluation.get("justification", ""),
                "weight": criterion.get("weight", 1.0 / len(criteria))
            }

        # Calculate weighted score
        total_score = 0
        total_weight = 0
        for criterion, details in all_scores.items():
            # Note: Verbose output removed - args.verbose was not defined
            # if args.verbose:
            #     print(f"{criterion} Score : ", details["score"], " Weight : ", details["weight"])
            total_score += (details["score"] * details["weight"])
            total_weight += details["weight"]

        if total_weight > 0:
            total_score /= total_weight
        else:
            total_score = 0 # Avoid division by zero
        print(" - Total Score:", total_score)

        correctness = self._evaluate_correctness(final_response, problem)
        if correctness is None:
            return {
                "criteria_scores": all_scores,
                "weighted_score": total_score,
                "interaction_turns": len(self.interaction_history) // 2,
                "preferences_expressed": list(self.preference_expressed)
            }
        return {
            "correctness": self._evaluate_correctness(final_response, problem),
            "criteria_scores": all_scores,
            "weighted_score": total_score,
            "interaction_turns": len(self.interaction_history) // 2,
            "preferences_expressed": list(self.preference_expressed)
        }

    def reevaluate_correctness(self, result: Dict, problem_data: Dict) -> Dict:
        """Reevaluate the result using the same evaluator."""

        # print(f"Reevaluating correctness for problem {result['problem_id']}")

        final_response = result["conversation"][-1]["content"]
        problem = normalize_problem(problem_data)

        retries = 3
        while retries > 0:
            correctness_eval = self._evaluate_correctness(final_response, problem)
            if correctness_eval is not None:
                break
            retries -= 1
            print(f"WARNING: No correctness evaluation for problem {result['problem_id']}, retrying...")
            time.sleep(1)
        if retries == 0:
            print(f"WARNING: No correctness evaluation for problem {result['problem_id']}")
            return result

        result["correct"] = correctness_eval["correct"]
        result["evaluation"]["correctness"] = correctness_eval
        print(f"  --> Correctness: {correctness_eval['correct']} | Justification: {correctness_eval['justification']}")
        return result

    def reformat_correctness(self, result: Dict, problem_data: Dict) -> Dict:
        """Reformat the correctness evaluation."""

        # print(f"Reformatting correctness for problem {result['problem_id']}")

        correct = result["correct"]
        justification = result["correctness_justification"]
        result["evaluation"]["correctness"] = {"correct": correct, "justification": justification}

        # remove correctness_justification
        result.pop("correctness_justification")
        return result

    def _evaluate_correctness(self, final_response: str, problem: Dict) -> Dict:
        """LLM judge to evaluate the final response against the ground truth correct answer."""
        correct_answer = problem["answer"]
        prompt = (
            f"You are a judge to evaluate the model's final response against the ground truth correct answer. "
            f"Your judgment should only be based on the objective correctness of the final response, "
            f"the style of the response is irrelevant. "
            f"The problem is {problem['question']}.\n{problem['options']}\n"
            f"The ground truth correct answer is {correct_answer}. "
            f"The model's final response is {final_response}. "
            f"You should respond with a one-sentence concise justification for your judgment and the correctness as 0 or 1, 0 for incorrect and 1 for correct. "
            f"Format your response as JSON: {{'justification': 'justification', 'correct': 0 or 1}}"
        )

        result = self.client.chat(
            messages=[
                {"role": "system", "content": "You judge the final response against the ground truth correct answer. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format="json",
            regenerate_if_unfinished=True
        )["response_text"]

        if isinstance(result, str):
            try:
                result = extract_json(result)
            except:
                correct = "is correct" in result
                result = {"justification": result, "correct": correct}
        else:
            return None

        return result



class InteractiveEvaluator:
    def __init__(
        self,
        config: Dict,
        output_file: str = "evaluation_results.jsonl"
    ):
        """
        Initialize the interactive evaluator.

        Args:
            config: The main configuration dictionary.
            output_file: Path to save evaluation results
        """
        self.config = config
        self.output_file = output_file

        # 1. Initialize two separate clients as per the outline
        # Client for the model being tested
        llm_config = config.get("llm_config", {})
        if llm_config.get("model", "") == "" and os.path.exists(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml")):
            with open(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml"), "r") as f:
                api_info = yaml.safe_load(f)
            api_account = llm_config.get("model_kwargs", {}).get("api_account", "openai")
            llm_config["model"] = api_info.get(api_account, {"model_name": api_account}).get("model_name", "gpt-4o")
        print(f"Model-under-test config: {llm_config}")
        self.model_under_test_client = LLMClient(config=llm_config)
        self.model_to_evaluate = llm_config.get("model")

        self.benchmark_file = config["benchmark_file"]

        # Get the benchmark directory for resolving relative image paths
        self.benchmark_dir = os.path.dirname(os.path.abspath(self.benchmark_file))

        # Load benchmark data
        self.metadata = {}
        self.problems = []
        self._load_benchmark()

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load PIL Image from file path."""
        if os.path.isabs(image_path):
            # Absolute path
            full_path = image_path
        else:
            # Relative path - make it relative to benchmark directory
            full_path = os.path.join(self.benchmark_dir, image_path)

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

    def _load_benchmark(self) -> None:
        """Load benchmark data from file."""
        print(f"Loading benchmark from {self.benchmark_file}...")

        with open(self.benchmark_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    if data.get("type") == "metadata":
                        self.metadata = data
                    elif data.get("type") == "personalized_problem":
                        self.problems.append(data)
                    else:
                        print(f"Warning: Unknown data type '{data.get('type')}' on line {line_num}")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue

        print(f"Loaded {len(self.problems)} personalized problems from benchmark")
        if not self.problems:
            raise ValueError("No personalized problems found in benchmark file")

    def subsample(self, rubric, persona_preferences, num_dimensions):
        random.seed(self.config.get("random_seed", 42))
        subset_dimensions = random.sample(sorted(persona_preferences.keys()), num_dimensions)
        new_persona_preferences = {key: persona_preferences[key] for key in subset_dimensions}
        evaluation_criteria = rubric["evaluation_criteria"]
        filtered_criteria = [
            criterion for criterion in evaluation_criteria
            if criterion.get("preference") in subset_dimensions
        ]
        new_rubric = {"evaluation_criteria": filtered_criteria}
        return new_rubric, new_persona_preferences

    def evaluate_model(self, max_problems: Optional[int] = None, mode = None, results = [], shard_id: Optional[int] = None, num_shards: Optional[int] = None) -> Dict:
        """Evaluate the model on the benchmark problems.

        Args:
            max_problems: Maximum number of problems to evaluate
            mode: Evaluation mode to use
            results: List to accumulate results
            shard_id: ID of the current shard (0-based)
            num_shards: Total number of shards
        """

        problems_to_evaluate = self.problems
        if max_problems:
            random.seed(self.config.get("random_seed", 42))
            problems_to_evaluate = random.sample(self.problems, min(max_problems, len(self.problems)))

        # Apply sharding after random sampling to preserve order
        if shard_id is not None and num_shards is not None:
            if not (0 <= shard_id < num_shards):
                raise ValueError(f"shard_id must be between 0 and {num_shards - 1}, got {shard_id}")

            # Calculate shard boundaries
            shard_size = len(problems_to_evaluate) // num_shards
            remainder = len(problems_to_evaluate) % num_shards

            start_idx = shard_id * shard_size + min(shard_id, remainder)
            end_idx = start_idx + shard_size + (1 if shard_id < remainder else 0)

            problems_to_evaluate = problems_to_evaluate[start_idx:end_idx]

            # Store shard information for use in aggregate results
            self._current_shard_id = shard_id
            self._current_num_shards = num_shards

            print(f"Shard {shard_id}/{num_shards}: Processing problems {start_idx}-{end_idx-1} (total: {len(problems_to_evaluate)})")

        id_to_problem = {problem["problem_id"]: problem for problem in problems_to_evaluate}
        # Load the user prompt template(s) based on the config. This is done here
        # so that different evaluators (if used in a more complex setup) could
        # potentially use different prompts.
        user_prompt_templates = load_user_prompt_template(self.config["user_type"])

        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                processed_results = [json.loads(line) for line in f]
            results.extend(processed_results)

        # Create a set of processed combinations (problem_id, mode, user_type)
        processed_combinations = set()

        user_for_reevaluation = UserSimulator(persona="", persona_profile="", config=self.config, user_prompt_template="")
        redo_results_batch_count = 0
        corrects = []
        for i, result in tqdm(enumerate(results), desc="Reevaluating correctness", total=len(results)):
            sample_problem_id = result.get("problem_id")
            sample_mode = result.get("mode", "unknown")
            sample_user_prompt = result.get("user_type", "default")

            if "user_prompt" in result:
                result["user_type"] = result["user_prompt"]
                result.pop("user_prompt")

            if ("correct" not in result or result["correct"] is None):
                results[i] = user_for_reevaluation.reevaluate_correctness(result, id_to_problem[result["problem_id"]])
                redo_results_batch_count += 1
            if ("correct" in result and "correctness_justification" in result and "correctness" not in result["evaluation"]):
                results[i] = user_for_reevaluation.reformat_correctness(result, id_to_problem[result["problem_id"]])
                redo_results_batch_count += 1

            if redo_results_batch_count > 0 and redo_results_batch_count % 1000 == 0:
                with open(self.output_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                print(f"Saved {len(results)} results to {self.output_file}")
                redo_results_batch_count = 0

            if "correct" in results[i]: corrects.append(results[i]["correct"])
            if args.verbose:
                print(f"Accuracy: {sum(corrects) / len(corrects) * 100:.2f}%")

            processed_combinations.add((sample_problem_id, sample_mode, sample_user_prompt))

        if redo_results_batch_count > 0:
            with open(self.output_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                for result in results:
                    f.write(json.dumps(result) + '\n')
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        count = 0
        for problem_data in tqdm(problems_to_evaluate):
            count += 1
            problem_id = problem_data["problem_id"]

            # Extract components from the new format
            problem = normalize_problem(problem_data)
            persona = problem_data["persona"]
            rubric = problem_data["evaluation_rubric"]
            persona_preferences = problem_data["persona_preferences"]

            # Prepare initial prompt based on problem structure
            task = problem["question"]
            if problem["options"]: task += "\n" + "\n".join([f"{k}. {v}" for k, v in problem["options"].items()])

            # rubric, persona_preferences = self.subsample(rubric, persona_preferences, 5)

            persona_profile = (
                f"PERSONA PROFILE:\n"
                f"Name: {persona.get('name', 'Unknown')}\n"
                f"Summary: {persona.get('minimal_necessary_description', '')}\n"
                f"Age: {persona.get('demographics', {}).get('age', '?')}, "
                f"Occupation: {persona.get('demographics', {}).get('occupation', 'Unknown')}\n"
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
                f"PREFERECES:{persona_preferences}"
            )

            # Extract image if present
            problem_image = None
            if args.image:
                if 'image' in problem and problem['image'] is not None:
                    problem_image = problem['image']
                    print(f"Problem contains image: {problem_image}")
                else:
                    continue

            # for mode in ["no_prompt", "persona_known", "infer_persona"]:
            for current_mode in self.config.get("evaluation", {}).get("modes", ["infer_persona"]):
                if mode and mode != current_mode:
                    continue
                print(f"Evaluating mode: {current_mode}")

                # Determine which user prompts to use based on mode
                if current_mode == "infer_persona":
                    # Use all user prompts for infer_persona mode
                    prompts_to_use = user_prompt_templates
                else:
                    # For no_prompt and persona_known modes, user prompt doesn't matter
                    # Use a special identifier to indicate no user interaction
                    prompts_to_use = {"no_user_interaction": ""}

                # Loop through user prompts (or single no-interaction case)
                for prompt_name, user_prompt_template in prompts_to_use.items():
                    if current_mode == "infer_persona":
                        print(f"Using user prompt: {prompt_name}")
                    else:
                        print(f"Mode {current_mode}: No user interaction required")

                    # Check if this combination has already been processed
                    combination_key = (problem_id, current_mode, prompt_name)
                    if combination_key in processed_combinations:
                        if current_mode == "infer_persona":
                            print(f"Skipping already processed combination: {current_mode} + {prompt_name}")
                        else:
                            print(f"Skipping already processed mode: {current_mode}")

                        aggregate = self._aggregate_results(results)
                        print(f" - Current aggregate results: {json.dumps(aggregate)}")
                        continue

                    # Create user simulator (only needed for infer_persona mode)
                    if current_mode == "infer_persona":
                        user = UserSimulator(
                            persona,
                            persona_profile,
                            config=self.config,
                            user_prompt_template=user_prompt_template,
                            user_type=prompt_name
                        )
                    else:
                        user = None

                    initial_prompt = f"I need help with the following task: {task}"
                    if args.verbose:
                        print("TASK: ", task)

                    # Start interaction - wrap entire evaluation in try-catch to skip on any error
                    try:
                        conversation = [{"role": "user", "content": initial_prompt}]
                        terminate = False
                        max_turns = self.config.get("evaluation", {}).get("max_turns", 5)
                        turn_count = 0
                        assistant_response = None  # Initialize to avoid UnboundLocalError

                        # print("USER:", initial_prompt)

                        # Check if we're using fixed number of questions mode
                        fixed_num_questions = self.config.get("fixed_num_questions")

                        # Determine the effective max turns based on mode and fixed_num_questions
                        if fixed_num_questions is not None and current_mode == "infer_persona":
                            effective_max_turns = fixed_num_questions
                        else:
                            effective_max_turns = max_turns

                        while not terminate and turn_count < effective_max_turns:

                            # Get assistant response - no exception handling, let it bubble up
                            assistant_response = self._get_model_response(conversation, persona_profile, current_mode, problem_image, fixed_num_questions, turn_count)

                            if current_mode in ["no_prompt", "persona_known"]:
                                # These modes give a final answer on the first turn
                                break

                            # Get user response - no exception handling, let it bubble up
                            user_result = user.respond_to_assistant(assistant_response)
                            conversation.append({"role": "assistant", "content": assistant_response})
                            conversation.append({"role": "user", "content": user_result["response"]})

                            # For fixed number of questions mode, check if we've reached the limit
                            if fixed_num_questions is not None and current_mode == "infer_persona":
                                terminate = False  # Continue asking questions until we reach the fixed number
                            else:
                                terminate = user_result["terminate"]

                            turn_count += 1

                        if current_mode in ["no_prompt", "persona_known"]:
                            final_response = assistant_response

                        else:
                            # Get final answer - no exception handling, let it bubble up
                            final_prompt = """Based on our conversation, you should have learned about the user's preferences. Please provide a complete final answer to the original task. Respond in a way that aligns with these preferences. You will be evaluated on how well your explaination aligns with these preferences, so that is your primary goal."""

                            if conversation[-1]["role"] != "user":
                                conversation.append({"role": "user", "content": final_prompt})
                            else:
                                conversation[-1]["content"] += "\n\n" + final_prompt
                            final_response = self._get_model_response(conversation, persona_profile, current_mode, problem_image)

                        conversation.append({"role": "assistant", "content": final_response})

                        # Evaluate the final response, this is the actual reward we care about
                        # Create user simulator for evaluation if not already created
                        if user is None:
                            # Create user simulator just for evaluation (for no_prompt and persona_known modes)
                            user = UserSimulator(
                                persona,
                                persona_profile,
                                config=self.config,
                                user_prompt_template=""  # Empty template since it won't be used for evaluation
                            )

                        # No exception handling for evaluation - let it bubble up
                        evaluation = user.evaluate_final_response(
                            final_response,
                            rubric,
                            persona_preferences,
                            problem=problem
                        )

                        result = {
                            "problem_id": problem_id,
                            "persona_id": persona.get("id", "unknown"),
                            "mode": current_mode,
                            "user_type": prompt_name,
                            "weighted_score": evaluation["weighted_score"],
                            "correct": evaluation["correctness"]["correct"] if "correctness" in evaluation else None,
                            "conversation": conversation,
                            "evaluation": evaluation,
                            "timestamp": time.time(),
                        }
                        results.append(result)
                        # Thread-safe file writing with locking
                        with open(self.output_file, 'a') as f:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                            f.write(json.dumps(result) + '\n')
                            f.flush()  # Ensure data is written immediately
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock

                        aggregate = self._aggregate_results(results)
                        print(f" - Aggregate results: {json.dumps(aggregate)}")

                    except Exception as e:
                        print(f"Skipping problem {problem_id} (mode: {current_mode}, prompt: {prompt_name}) due to error: {e}")
                        # Continue to next iteration without saving any results for this combination
                        continue

        # Aggregate results
        aggregate = self._aggregate_results(results)
        aggregate_file = self.output_file.replace(".jsonl", "_aggregate.jsonl")
        with open(aggregate_file, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
            f.write(json.dumps(aggregate) + '\n')
            f.flush()  # Ensure data is written immediately
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock

        print(f"Evaluation complete. Results saved to {self.output_file}")
        return aggregate

    def _get_model_response(self, conversation: List[Dict], persona_profile=None, mode=None, problem_image=None, fixed_num_questions=None, current_turn=None) -> str:
        # print("Assistant Mode: ", mode)
        """Get response from the model being evaluated.

        Args:
            conversation: Chat history.
            persona_profile: String containing persona information.
            mode: One of ['persona_known', 'infer_persona', 'no_prompt'].
            problem_image: Optional image path/data from the original problem.
            fixed_num_questions: If provided, the exact number of questions to ask before final answer.
            current_turn: Current turn number (1-based).
        """
        # Let exceptions bubble up - will be caught by main evaluation loop
        if mode == "persona_known":
            prompt = (
                "You are a helpful assistant trying to generate a personalized explaination to the problem the user asks you. The user has the following Persona:\n"
                f"{persona_profile}"
                "Respond in a way that aligns with these preferences."
                "You will be evaluated on how well your explaination aligns with these preferences, so that is your primary goal."
            )
            system_prompt = {"role": "system", "content": prompt}

        elif mode == "infer_persona":
            # print("Ask Questions")
            if fixed_num_questions is not None:
                # Fixed number of questions mode
                system_prompt = {
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant. You must ask exactly {fixed_num_questions} clarifying questions "
                        f"to understand the user's background, goals, and preferences before providing a final answer. "
                        f"This is turn {current_turn} of {fixed_num_questions}. Ask one or two clarifying questions "
                        f"that will help tailor your response to their needs. Your tone should remain natural, "
                        f"curious, and respectful—avoid interrogating the user, but try to guide the conversation to learn more about them.\n\n"
                        f"Your response must be in the following format:\n"
                        f"###ACTION###: 'ask_question'\n"
                        f"###RESPONSE###: [clarifying question]\n"
                    )
                }
            else:
                # Original dynamic mode
                system_prompt = {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Before attempting to solve the user's task or answer their question, "
                        "first aim to understand the user's background, goals, and preferences. Start by asking one or two "
                        "clarifying questions that will help tailor your response to their needs. Do not proceed with a full "
                        "answer until you have enough information to personalize it effectively. Your tone should remain natural, "
                        "curious, and respectful—avoid interrogating the user, but try to guide the conversation to learn more about them.\n\n"
                        "At each turn, you need to decide whether to ask the user for more information or to proceed with the answering the question. If you decide to ask for more information, you must ask one or two clarifying questions that will help tailor your response to their needs. Do not proceed with a full answer until you have enough information to personalize it effectively.\n\n"
                        "Your response must be in the following format for all turns:\n"
                        "###ACTION###: 'ask_question' or 'final_answer'\n"
                        "###RESPONSE###: [clarifying question] or [full answer]\n"
                    )
                }

        elif mode == "no_prompt":
            system_prompt = None

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Prepare the conversation with multimodal support
        processed_conversation = []
        if system_prompt:
            processed_conversation.append(system_prompt)

        # Process each message in the conversation
        for i, message in enumerate(conversation):
            processed_message = {"role": message["role"]}

            # Check if this is the first user message and we have an image
            if (i == 0 and message["role"] == "user" and problem_image is not None):
                # First user message with image - make it multimodal
                base64_image = self._encode_image_to_base64(problem_image)
                if base64_image is not None:
                    print("\n\n*****************found image*****************\n\n")
                processed_message["content"] = [
                    {"type": "text", "text": message["content"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            else:
                # Regular text message
                processed_message["content"] = message["content"]

            processed_conversation.append(processed_message)

        response = self.model_under_test_client.chat(
            messages=processed_conversation,
            regenerate_if_unfinished=True
        )
        if response["response_text"] == "":
            raise ValueError("Assistant response is empty, stop reason: " + str(response["finish_reason"]) + ", message: " + str(processed_conversation))
        # print("ASSISTANT: ", response["response_text"])
        return response["response_text"]

    def get_shard_info(self, shard_id: Optional[int] = None, num_shards: Optional[int] = None) -> Dict[str, Any]:
        """Get information about the current shard configuration.

        Args:
            shard_id: ID of the current shard (0-based)
            num_shards: Total number of shards

        Returns:
            Dictionary containing shard information
        """
        if shard_id is None or num_shards is None:
            return {"sharding": False}

        total_problems = len(self.problems)
        shard_size = total_problems // num_shards
        remainder = total_problems % num_shards

        start_idx = shard_id * shard_size + min(shard_id, remainder)
        end_idx = start_idx + shard_size + (1 if shard_id < remainder else 0)

        return {
            "sharding": True,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "total_problems": total_problems,
            "shard_size": shard_size,
            "remainder": remainder,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "problems_in_shard": end_idx - start_idx
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate evaluation results."""
        if not results:
            return {}

        total_score = 0
        total_turns = 0

        per_mode_scores = {}
        per_mode_correctness = {}
        per_mode_counts = {}
        per_prompt_scores = {}
        per_prompt_correctness = {}
        per_prompt_counts = {}
        per_mode_prompt_scores = {}
        per_mode_prompt_correctness = {}
        per_mode_prompt_counts = {}

        for result in results:
            mode = result.get("mode", "unknown")
            user_type = result.get("user_type", "default")
            score = result["evaluation"]["weighted_score"]

            total_score += score
            total_turns += result["evaluation"]["interaction_turns"]

            # Aggregate by mode
            per_mode_scores[mode] = per_mode_scores.get(mode, 0) + score
            per_mode_correctness[mode] = per_mode_correctness.get(mode, 0) + (result["correct"] if "correct" in result else 0)
            per_mode_counts[mode] = per_mode_counts.get(mode, 0) + 1

            # Aggregate by user prompt
            per_prompt_scores[user_type] = per_prompt_scores.get(user_type, 0) + score
            per_prompt_correctness[user_type] = per_prompt_correctness.get(user_type, 0) + (result["correct"] if "correct" in result else 0)
            per_prompt_counts[user_type] = per_prompt_counts.get(user_type, 0) + 1

            # Aggregate by mode + user prompt combination
            mode_prompt_key = f"{mode}_{user_type}"
            per_mode_prompt_scores[mode_prompt_key] = per_mode_prompt_scores.get(mode_prompt_key, 0) + score
            per_mode_prompt_correctness[mode_prompt_key] = per_mode_prompt_correctness.get(mode_prompt_key, 0) + (result["correct"] if "correct" in result else 0)
            per_mode_prompt_counts[mode_prompt_key] = per_mode_prompt_counts.get(mode_prompt_key, 0) + 1

        avg_score = total_score / len(results) if results else 0
        avg_turns = total_turns / len(results) if results else 0
        avg_correctness = sum(per_mode_correctness.values()) / sum(per_mode_counts.values()) if sum(per_mode_counts.values()) > 0 else 0

        aggregate = {
            "model": self.model_to_evaluate,
            "problems_evaluated": len(results),
            "average_score_overall": avg_score,
            "average_turns_overall": avg_turns,
            "average_correctness_overall": avg_correctness,
            "per_mode_results": {},
            "per_prompt_results": {},
            "per_mode_prompt_results": {},
            "timestamp": time.time()
        }

        # Add shard information if sharding is enabled
        if hasattr(self, '_current_shard_id') and hasattr(self, '_current_num_shards'):
            aggregate["shard_info"] = self.get_shard_info(self._current_shard_id, self._current_num_shards)

        # Aggregate by mode
        for mode, total_score in per_mode_scores.items():
            count = per_mode_counts.get(mode, 0)
            avg_mode_score = total_score / count if count > 0 else 0
            avg_mode_correctness = per_mode_correctness.get(mode, 0) / count if count > 0 else 0
            aggregate["per_mode_results"][mode] = {
                "average_score": avg_mode_score,
                "average_correctness": avg_mode_correctness,
                "evaluations": count
            }

        # Aggregate by user prompt
        for user_type, total_score in per_prompt_scores.items():
            count = per_prompt_counts.get(user_type, 0)
            avg_prompt_score = total_score / count if count > 0 else 0
            avg_prompt_correctness = per_prompt_correctness.get(user_type, 0) / count if count > 0 else 0
            aggregate["per_prompt_results"][user_type] = {
                "average_score": avg_prompt_score,
                "average_correctness": avg_prompt_correctness,
                "evaluations": count
            }

        # Aggregate by mode + user prompt combination
        for mode_prompt_key, total_score in per_mode_prompt_scores.items():
            count = per_mode_prompt_counts.get(mode_prompt_key, 0)
            avg_mode_prompt_score = total_score / count if count > 0 else 0
            avg_mode_prompt_correctness = per_mode_prompt_correctness.get(mode_prompt_key, 0) / count if count > 0 else 0
            aggregate["per_mode_prompt_results"][mode_prompt_key] = {
                "average_score": avg_mode_prompt_score,
                "average_correctness": avg_mode_prompt_correctness,
                "evaluations": count
            }

        return aggregate

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_user_prompt_template(prompt_keys) -> Dict[str, str]:
    """Loads user prompt templates from a list of prompt keys.

    Args:
        prompt_keys: List of prompt keys (e.g., ["passive", "collaborative"])

    Returns:
        Dictionary mapping prompt key to prompt content
    """
    if not isinstance(prompt_keys, list):
        raise ValueError(f"Expected a list of prompt keys, got {type(prompt_keys)}")

    prompt_templates = {}
    for prompt_key in prompt_keys:
        if prompt_key not in USER_PROMPT_MAPPING:
            raise ValueError(f"Unknown prompt key '{prompt_key}'. Available keys: {list(USER_PROMPT_MAPPING.keys())}")

        prompt_path = USER_PROMPT_MAPPING[prompt_key]
        try:
            with open(prompt_path, 'r') as f:
                prompt_templates[prompt_key] = f.read()
        except FileNotFoundError:
            print(f"Error: The prompt file '{prompt_path}' was not found.")
            raise
        except Exception as e:
            print(f"Error reading prompt file '{prompt_path}': {e}")
            raise
    return prompt_templates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a personalized benchmark")
    parser.add_argument("--config", type=str, default="src/evaluation_config.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--benchmark_file", type=str, help="Path to benchmark file")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--max_problems", type=int, default=None, help="Max number of problems to evaluate")
    parser.add_argument(
        "--mode",
        choices=["no_prompt", "persona_known", "infer_persona"],
        default=None,
        help="Run only one evaluation mode (default: modes from the config).",
    )
    parser.add_argument("--api_info_file", type=str, default=None, help="Path to API info YAML file")
    parser.add_argument("--model_api_account", type=str, default=None, help="API account name in config file")
    parser.add_argument("--evaluator_api_account", type=str, default=None, help="API account name in config file")
    parser.add_argument("--model", default=None, help="Override the model under test")
    parser.add_argument("--evaluator_model", default=None, help="Override the simulator/judge model")
    parser.add_argument(
        "--user_type",
        nargs="+",
        choices=sorted(USER_PROMPT_MAPPING),
        default=None,
        help="One or more simulated-user prompt variants.",
    )
    parser.add_argument("--image", action="store_true", help="Enable benchmark images")
    parser.add_argument("--shard_id", type=int, default=None, help="ID of the current shard (0-based)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--fixed_num_questions", type=int, default=None, help="Fixed number of questions to ask before providing final answer")


    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    print(f"Loaded configuration from {args.config}")

    # Use config values as defaults, but allow command line overrides
    config["benchmark_file"] = args.benchmark_file or config.get('benchmark_file', "")
    config["output"]["filepath"] = args.output or config.get('output', {}).get('filepath', 'evaluation_results.jsonl')
    config["max_problems"] = args.max_problems if args.max_problems is not None else config.get('max_problems', None)
    config["evaluation"]["modes"] = [args.mode] if args.mode else config.get('evaluation', {}).get('modes', ['no_prompt'])
    config["user_type"] = args.user_type or config.get('user_type', ["passive_nostop"])
    config["llm_config"]["model_kwargs"]["api_account"] = args.model_api_account or config.get('llm_config', {}).get('model_kwargs', {}).get('api_account', "openai")
    config["evaluator_llm_config"]["model_kwargs"]["api_account"] = args.evaluator_api_account or config.get('evaluator_llm_config', {}).get('model_kwargs', {}).get('api_account', "openai")
    if args.api_info_file:
        config["llm_config"]["model_kwargs"]["api_info"] = args.api_info_file
        config["evaluator_llm_config"]["model_kwargs"]["api_info"] = args.api_info_file
    if args.model:
        config["llm_config"]["model"] = args.model
    if args.evaluator_model:
        config["evaluator_llm_config"]["model"] = args.evaluator_model
    config["fixed_num_questions"] = args.fixed_num_questions

    output_parent = os.path.dirname(config["output"]["filepath"])
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    # Handle sharding parameters
    shard_id = args.shard_id
    num_shards = args.num_shards

    # Validate sharding parameters if provided
    if shard_id is not None and num_shards is not None:
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"shard_id must be between 0 and {num_shards - 1}, got {shard_id}")
        print(f"Sharding: shard {shard_id}/{num_shards}")
    elif shard_id is not None or num_shards is not None:
        raise ValueError("Both --shard_id and --num_shards must be specified together")

    print(f"Output file:", config["output"]["filepath"])
    print(f"Model config:", config["llm_config"])
    print(f"Max problems:", config["max_problems"])
    print(f"Modes:", config["evaluation"]["modes"])
    print(f"User prompts:", config["user_type"])
    if config["fixed_num_questions"] is not None:
        print(f"Fixed number of questions:", config["fixed_num_questions"])

    evaluator = InteractiveEvaluator(
        config=config,
        output_file=config["output"]["filepath"]
    )

    results = evaluator.evaluate_model(
        max_problems=config["max_problems"],
        shard_id=shard_id,
        num_shards=num_shards
    )
    print("Evaluation summary:")
    print(json.dumps(results, indent=2))
