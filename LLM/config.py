import os


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


class LLMConfig:
    API_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1500)

    RULE_TXT_PATH = os.getenv("RULE_TXT_PATH", "rule.txt")
    BASE_MODEL_NAME = os.getenv("EMBEDDING_BASE_MODEL", "BAAI/bge-m3")
    LORA_MODEL_PATH = os.getenv("EMBEDDING_LORA_PATH", "model/bge")

    TOP_K_RAG = max(1, _env_int("TOP_K_RAG", 5))
    MAX_RULES = _env_int("MAX_RULES", 0) or None
