import os
import torch
from sentence_transformers import SentenceTransformer, util

_MODEL_CACHE = None

BASE_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_LORA_PATH = "model/nomic"

def get_nomic_model(lora_path=DEFAULT_LORA_PATH):
    """
    Tải mô hình Nomic với cấu trúc tùy chỉnh. 
    Lưu ý: Nomic v1.5 yêu cầu trust_remote_code=True.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Khởi tạo không gian tính toán Nomic trên thiết bị: {device.upper()}")
    
    # 1. Nạp mô hình trực tiếp từ thư mục chứa model đã merge hoặc path finetuned
    # Nếu thư mục 'model/nomic' chứa model đã merge (chứa adapter), nó sẽ load trực tiếp
    try:
        model = SentenceTransformer(lora_path, trust_remote_code=True).to(device)
        print(f"[+] Tải thành công mô hình Nomic từ: {lora_path}")
    except Exception as e:
        print(f"[WARNING] Lỗi khi nạp từ {lora_path}: {e}. Fallback về base model...")
        model = SentenceTransformer(BASE_MODEL_NAME, trust_remote_code=True).to(device)

    _MODEL_CACHE = model
    return _MODEL_CACHE


def nomic_semantic_ranking(passed_filter_cvs, jd_text, top_k=5, lora_path=DEFAULT_LORA_PATH):
    """
    Xếp hạng ứng viên sử dụng Nomic Embedding với tiền tố bất đối xứng (asymmetric).
    
    Parameters:
    - passed_filter_cvs: Danh sách CV đã qua lọc.
    - jd_text: Nội dung Job Description.
    """
    if not passed_filter_cvs:
        return []

    model = get_nomic_model(lora_path)

    # 1. Chuẩn bị dữ liệu với tiền tố Nomic
    # Nomic yêu cầu 'search_query:' cho JD và 'search_document:' cho nội dung CV
    jd_query = "search_query: " + jd_text
    cv_contents = ["search_document: " + item.get("structured_data", {}).get("cv_content", "") 
                   for item in passed_filter_cvs]

    # 2. Mã hóa Embeddings
    print(f"[INFO] Trích xuất đặc trưng Nomic cho {len(cv_contents)} ứng viên...")
    
    with torch.no_grad():
        jd_embedding = model.encode(jd_query, convert_to_tensor=True, normalize_embeddings=True)
        cv_embeddings = model.encode(cv_contents, convert_to_tensor=True, normalize_embeddings=True)
        
        # 3. Tính Cosine Similarity
        cosine_sims = util.cos_sim(jd_embedding, cv_embeddings)[0].cpu().numpy()

    # 4. Gán điểm số và sắp xếp
    for idx, item in enumerate(passed_filter_cvs):
        item["embedding_score"] = float(cosine_sims[idx])

    sorted_cvs = sorted(passed_filter_cvs, key=lambda x: x["embedding_score"], reverse=True)

    return sorted_cvs[:top_k]