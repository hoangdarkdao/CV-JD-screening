import os

class LLMConfig:
    # 1. CẤU HÌNH NHÀ CUNG CẤP API (Có thể chuyển đổi giữa OpenAI, Gemini, hoặc local vLLM/Ollama)
    API_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai / gemini / local
    API_KEY = os.getenv("LLM_API_KEY", "YOUR_API_KEY_HERE")
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    
    # 2. CẤU HÌNH MÔ HÌNH (MODEL CONFIG)
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")  # Hoặc gemini-1.5-flash / qwen2.5
    TEMPERATURE = 0.2  # Đặt thấp để đảm bảo tính logic và nhất quán của kết quả chấm điểm
    MAX_TOKENS = 1500
    
    # 3. ĐƯỜNG DẪN HỆ THỐNG (SYSTEM PATHS)
    RULE_TXT_PATH = os.getenv("RULE_TXT_PATH", "rule.txt")
    BASE_MODEL_NAME = "BAAI/bge-m3"
    LORA_MODEL_PATH = "model/bge"
    
    # 4. THAM SỐ PHỄU LỌC (FILTER HYPERPARAMETERS)
    TOP_K_RAG = 5  # Số lượng CV tối đa chuyển qua tầng LLM chấm điểm sâu