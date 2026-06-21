import os
import torch
from PIL import Image
import numpy as np

# Hỗ trợ tự động chuyển đổi file PDF thành hình ảnh tương thích với VLM
try:
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

# Đảm bảo thư viện Transformers phiên bản mới (hỗ trợ trực tiếp Qwen2-VL/Qwen2.5-VL)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Biến toàn cục đóng vai trò bộ nhớ đệm (Cache) tối ưu hóa tài nguyên phần cứng
_QWEN_MODEL_CACHE = None
_QWEN_PROCESSOR_CACHE = None

# Định nghĩa tên mô hình mặc định (Có thể đổi sang "Qwen/Qwen2.5-VL-7B-Instruct" hoặc bản 2B để chạy nhẹ hơn)
DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"


def get_qwen_vlm_model(model_name=DEFAULT_MODEL_NAME):
    """
    Khởi tạo cấu trúc Single-instance cho Qwen VLM.
    Nạp Mô hình và Bộ xử lý (Processor) vào VRAM duy nhất một lần.
    """
    global _QWEN_MODEL_CACHE, _QWEN_PROCESSOR_CACHE
    
    if _QWEN_MODEL_CACHE is not None and _QWEN_PROCESSOR_CACHE is not None:
        return _QWEN_MODEL_CACHE, _QWEN_PROCESSOR_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[QWEN OCR] Đang nạp mô hình Vision-Language: {model_name} trên thiết bị: {device.upper()}...")
    
    # Thiết lập kiểu dữ liệu tối ưu dựa trên phần cứng (bfloat16 cho Ampere+ hoặc float16 cho T4/P100)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Khởi tạo mô hình với cơ chế tự động phân bổ tầng bộ nhớ (device_map)
    _QWEN_MODEL_CACHE = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    # Khởi tạo bộ xử lý hình ảnh và văn bản tương ứng
    _QWEN_PROCESSOR_CACHE = AutoProcessor.from_pretrained(model_name)
    
    print("[+] Nạp thành công hệ thống mô hình Qwen VLM OCR.")
    return _QWEN_MODEL_CACHE, _QWEN_PROCESSOR_CACHE


def parse_page_with_qwen(model, processor, pil_image):
    """
    Truyền một trang ảnh CV vào mô hình Qwen VLM kèm theo Prompt định hướng cấu trúc,
    yêu cầu mô hình giữ nguyên định dạng phân cấp trường dữ liệu.
    """
    # Xây dựng Prompt ép kiểu định dạng đầu ra mô phỏng chính xác cấu trúc Heuristic
    prompt_instruction = (
        "You are an advanced Document Parsing Engine. Extract all text from this resume/CV image.\n"
        "Strictly obey the following formatting rules:\n"
        "1. Identify main section headers (e.g., EDUCATION, EXPERIENCE, SKILLS, PROJECTS) and write them in UPPERCASE.\n"
        "2. For list items or details under skills/experience/projects, format them with a leading bullet point '- '.\n"
        "3. Do NOT add any conversational intro, outro, markdown block fences (like ```text or ```json), or explanations.\n"
        "4. Output the raw structured text directly."
    )

    # Cấu trúc tin nhắn đầu vào theo chuẩn quy định Chat Template mới của HuggingFace Transformers
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt_instruction}
            ]
        }
    ]

    # Chuẩn bị dữ liệu đầu vào qua bộ xử lý Processor chuyên dụng
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.image_processor(images=pil_image, videos=None), None
    
    inputs = processor(
        text=[text_prompt],
        images=pil_image,
        padding=True,
        return_tensors="pt"
    )
    
    # Di chuyển các tensor đầu vào lên cùng thiết bị phần cứng của mô hình
    inputs = inputs.to(model.device)

    # Thực hiện suy luận không tính toán đạo hàm (Inference mode)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1500)
        
        # Loại bỏ các token thuộc prompt gốc ở đầu chuỗi kết quả
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Giải mã chuỗi token thu được thành văn bản thuần
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
    return output_text.strip()


def ocr_qwen_vlm(cv_path, model_name=DEFAULT_MODEL_NAME):
    """
    HÀM GIAO DIỆN CHÍNH (MAIN INTERFACE):
    Chấp nhận cả định dạng ảnh (.png, .jpg) lẫn tài liệu .pdf.
    Trả về chính xác cấu trúc dữ liệu mục tiêu: {"cv_content": cv_content_string}
    """
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"Không tìm thấy file CV tại đường dẫn: {cv_path}")
        
    ext = os.path.splitext(cv_path)[1].lower()
    pil_images = []
    
    # 1. Cơ chế bóc tách chuyển đổi tệp tin sang đối tượng Ảnh PIL chuẩn hóa
    if ext == '.pdf':
        if not HAS_PDF2IMAGE:
            raise ImportError(
                "Để phân tích file PDF bằng Qwen VLM, vui lòng cài đặt thư viện 'pdf2image' "
                "bằng lệnh: pip install pdf2image"
            )
        # Chuyển đổi các trang tài liệu PDF sang dạng ảnh PIL
        pil_images = pdf2image.convert_from_path(cv_path)
    else:
        # Mở file hình ảnh thông thường bằng thư viện PIL
        try:
            img = Image.open(cv_path).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            raise ValueError(f"Không thể mở hoặc đọc nội dung file ảnh qua PIL: {cv_path}. Lỗi: {e}")

    # 2. Gọi hàm lấy mô hình từ bộ nhớ đệm Cache chống lãng phí VRAM
    model, processor = get_qwen_vlm_model(model_name)
    
    page_outputs = []
    print(f"[QWEN OCR] Bắt đầu quét thông tin cấu trúc cho {len(pil_images)} trang của tài liệu...")
    
    # 3. Tiến hành xử lý tuần tự từng trang tài liệu bằng mô hình thị giác lớn
    for page_idx, pil_img in enumerate(pil_images, start=1):
        print(f" -> Đang xử lý trang {page_idx}/{len(pil_images)}...")
        try:
            page_text = parse_page_with_qwen(model, processor, pil_img)
            if page_text:
                page_outputs.append(page_text)
        except Exception as e:
            print(f"[-] Gặp lỗi khi phân tích OCR trang {page_idx} bằng Qwen: {e}")
            continue

    # 4. Ghép nối kết quả các trang bằng hai dấu xuống dòng liên tiếp (\n\n) để đồng bộ hóa
    combined_cv_content = "\n\n".join(page_outputs)
    
    # 5. Đóng gói đầu ra trả về mục tiêu khớp hoàn toàn với kiến trúc phễu lọc đa tầng của bạn
    return {"cv_content": combined_cv_content}