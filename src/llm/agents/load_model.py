# from .vllm import VllmAgent, VllmBaseAgent

def load_model(model_name, num_gpus=2, **kwargs):
    if model_name.startswith("gpt") and "oss" not in model_name:
        # in ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-0125-preview", "gpt-4-0613", "gpt-3.5-turbo-0125", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4.1-standard"]:
        from .gpt import AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent
        # model = AsyncConversationalGPTBaseAgent({'model': model_name, **kwargs})
        model = ConversationalGPTBaseAgent({'model': model_name, **kwargs})
    elif model_name.startswith("o3") or model_name.startswith("o4") or model_name.startswith("o1"):
        from .gpt import AsyncConversationalGPTReasoningAgent, ConversationalGPTReasoningAgent
        model = ConversationalGPTReasoningAgent({'model': model_name, **kwargs})
    elif model_name.startswith("gemini-"):
        from .gemini import AsyncGeminiAgent
        model = AsyncGeminiAgent({'model': model_name, **kwargs})
    elif model_name.startswith("anthropic") or model_name.startswith("claude"):
        from .claude import AsyncClaudeAgent
        model = AsyncClaudeAgent({'model': model_name, **kwargs})
    elif "oss" in model_name:
        from .huggingface import OssAgent
        model = OssAgent({'model': model_name, **kwargs})
    else:
        from .huggingface import HuggingFaceAgent
        model = HuggingFaceAgent({'model': model_name, **kwargs})
    return model
