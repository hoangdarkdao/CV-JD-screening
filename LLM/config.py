import os


def load_dotenv(path=None, override=False):
    env_path = path or os.getenv("ENV_FILE", ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value


load_dotenv()


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
