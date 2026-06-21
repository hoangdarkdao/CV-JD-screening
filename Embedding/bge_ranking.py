import os
import torch
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

# Biến toàn cục đóng vai trò bộ nhớ đệm (Cache) tránh nạp đi nạp lại mô hình lớn
_MODEL_CACHE = None

BASE_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_LORA_PATH = "model/bge"  # Đường dẫn thư mục lưu trữ model finetune theo yêu cầu của bạn

def get_bge_lora_model(lora_path=DEFAULT_LORA_PATH):
    """
    Hàm khởi tạo Single-instance: Tải mô hình gốc bge-m3 và cấu trúc tầng LoRA (PEFT).
    Nếu mô hình đã được tải trước đó, hàm trả về ngay thực thể cũ trong bộ nhớ đệm.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Khởi tạo không gian tính toán Embedding trên thiết bị: {device.upper()}")
    print(f"[INFO] Đang nạp mô hình nền tảng: {BASE_MODEL_NAME}...")
    
    # 1. Khởi tạo thực thể thực thể SentenceTransformer gốc
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 2. Nạp lớp bọc LoRA từ checkpoint lưu trong thư mục model/bge ứng dụng PEFT
    if os.path.exists(lora_path):
        print(f"[INFO] Tìm thấy thư mục trọng số LoRA tại '{lora_path}'. Tiến hành ghép màng (PEFT)...")
        try:
            # Áp dụng cơ chế map chính xác phím auto_model giống như trong cấu trúc thử nghiệm Notebook
            model[0].auto_model = PeftModel.from_pretrained(
                model[0].auto_model, 
                lora_path
            )
            print("[+] Tích hợp thành công lớp LoRA Fine-tuned mượt mà.")
        except Exception as e:
            print(f"[WARNING] Xảy ra lỗi khi nạp trọng số LoRA từ '{lora_path}': {e}")
            print("[WARNING] Hệ thống tự động fallback sử dụng mô hình nền tảng gốc (Base Model).")
    else:
        print(f"[WARNING] Không tìm thấy thư mục LoRA tại '{lora_path}'. Mặc định sử dụng mô hình gốc (Base Model).")

    # Chuyển mô hình lên GPU (nếu có) hoặc CPU để tối ưu hóa hiệu năng suy luận
    model = model.to(device)
    _MODEL_CACHE = model
    return _MODEL_CACHE


def bge_semantic_ranking(passed_filter_cvs, jd_text, top_k=5, lora_path=DEFAULT_LORA_PATH):
    """
    Thực hiện mã hóa ngữ nghĩa văn bản của CV và JD sang không gian Vector,
    tính điểm tương đồng Cosine và xếp hạng để trích xuất Top-K ứng viên phù hợp nhất.
    
    Parameters:
    - passed_filter_cvs (list): Danh sách CV vượt qua vòng lọc thô ở Tầng 2.
    - jd_text (str): Nội dung văn bản của bản Mô tả công việc (Job Description).
    - top_k (int): Số lượng ứng viên xuất sắc nhất cần giữ lại từ tầng Embedding.
    - lora_path (str): Đường dẫn cục bộ chứa trọng số LoRA đã được tinh chỉnh.
    
    Returns:
    - list: Danh sách Top-K phần tử CV đã được sắp xếp giảm dần kèm theo khóa 'embedding_score'.
    """
    if not passed_filter_cvs:
        return []

    # Gọi hàm trích xuất mô hình từ Cache bảo vệ tài nguyên phần cứng
    model = get_bge_lora_model(lora_path)

    # 1. Thu thập chuỗi văn bản thuần của các CV từ cấu trúc dữ liệu JSON sinh từ tầng OCR
    cv_contents = []
    for item in passed_filter_cvs:
        # main.py lưu dữ liệu tại: item["structured_data"]["cv_content"]
        content = item.get("structured_data", {}).get("cv_content", "")
        cv_contents.append(content)

    # 2. Số hóa văn bản sang không gian vector (Mã hóa Embeddings)
    # Kích hoạt 'normalize_embeddings=True' theo đúng tiêu chuẩn thiết kế của BAAI/bge-m3
    print(f"[INFO] Tiến hành trích xuất đặc trưng ngữ nghĩa cho {len(cv_contents)} CV ứng viên...")
    
    with torch.no_grad():
        jd_embedding = model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
        cv_embeddings = model.encode(cv_contents, convert_to_tensor=True, normalize_embeddings=True)
        
        # 3. Tính toán độ tương quan Cosine qua Dot Product (Do vector đã được chuẩn hóa)
        cosine_sims = util.cos_sim(jd_embedding, cv_embeddings)[0].cpu().numpy()

    # 4. Gán điểm số Cosine Similarity tương ứng vào từng đối tượng ứng viên
    for idx, item in enumerate(passed_filter_cvs):
        item["embedding_score"] = float(cosine_sims[idx])

    # 5. Sắp xếp danh sách ứng viên giảm dần theo điểm số ngữ nghĩa sâu sâu sắc
    sorted_cvs = sorted(passed_filter_cvs, key=lambda x: x["embedding_score"], reverse=True)

    # 6. Trích lọc lấy chính xác số lượng Top-K ứng viên xuất sắc nhất chuyển giao cho tầng LLM
    return sorted_cvs[:top_k]