import os
import torch
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from peft import PeftModel

class Pix2StructCVParser:
    def __init__(self, base_model_id="google/pix2struct-base", lora_weights_path="model/pix2struct"):
        """
        Khởi tạo và nạp mô hình.
        """
        print(f"[INIT] Đang nạp Processor từ {base_model_id}...")
        self.processor = Pix2StructProcessor.from_pretrained(base_model_id)
        
        print(f"[INIT] Đang nạp Base Model ({base_model_id})...")
        base_model = Pix2StructForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" # Tự động phân bổ lên GPU nếu có
        )
        
        print(f"[INIT] Đang áp dụng LoRA weights từ {lora_weights_path}...")
        self.model = PeftModel.from_pretrained(base_model, lora_weights_path)
        self.model.eval() # Chuyển sang chế độ suy luận
        
        self.device = self.model.device

    def parse_output_to_dict(self, generated_text):
        """
        Dịch ngược chuỗi kết quả dạng "[key] value" (từ train2.py) thành Dictionary.
        """
        result_dict = {}
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and ']' in line:
                # Tách key và value dựa trên dấu ngoặc vuông
                end_bracket_idx = line.find(']')
                key = line[1:end_bracket_idx].strip()
                value = line[end_bracket_idx+1:].strip()
                if key:
                    result_dict[key] = value
        return result_dict

    def extract_cv_data(self, image_path):
        """
        Hàm chính để đọc ảnh CV và trả về dữ liệu có cấu trúc.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        # 1. Đọc và tiền xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 2. Sinh văn bản (Inference)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024, # Tương đương max_target_length trong lúc train
                use_cache=True
            )

        # 3. Giải mã kết quả
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # 4. Parse về dictionary để main.py có thể dùng được
        structured_data = self.parse_output_to_dict(generated_text)
        
        return structured_data

# Khởi tạo instance toàn cục để dùng chung (tránh việc load lại model mỗi lần gọi ảnh)
# Bạn hãy đảm bảo đường dẫn "model/pix2struct" là chính xác với nơi bạn lưu LoRA
cv_parser_instance = None

def ocr_pix2struct_cv(image_path):
    global cv_parser_instance
    if cv_parser_instance is None:
        cv_parser_instance = Pix2StructCVParser(lora_weights_path="model/pix2struct")
    return cv_parser_instance.extract_cv_data(image_path)