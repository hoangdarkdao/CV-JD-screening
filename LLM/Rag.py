import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel
from LLM.config import LLMConfig
from Embedding.hard_filter import execute_hard_filtering_stage

_BGE_CACHE = None

def get_bge_model():
    """Tải và lưu trữ mô hình BGE-M3 + LoRA vào bộ nhớ cache"""
    global _BGE_CACHE
    if _BGE_CACHE is not None:
        return _BGE_CACHE
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(LLMConfig.BASE_MODEL_NAME)
    
    if os.path.exists(LLMConfig.LORA_MODEL_PATH):
        try:
            model[0].auto_model = PeftModel.from_pretrained(
                model[0].auto_model, 
                LLMConfig.LORA_MODEL_PATH
            )
            print("[RAG INFO] Đã tích hợp thành công lớp LoRA vào BGE-M3.")
        except Exception as e:
            print(f"[RAG WARNING] Không thể nạp LoRA: {e}. Sử dụng Base model.")
            
    model = model.to(device)
    _BGE_CACHE = model
    return _BGE_CACHE

def hard_keyword_filter(cv_content, keywords):
    if not keywords:
        return True
    
    cv_lower = cv_content.lower()
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, cv_lower):
            return True
    return False

def heuristic_split_cv(text):
    """Hàm Helper: Bóc tách CV thành các trường phục vụ matching riêng biệt."""
    sections = {"Skill": "", "Exp": "", "Edu": ""}
    current_section = "Other" # Các phần rác sẽ bị bỏ qua khi tính embedding
    
    skill_pattern = re.compile(r'(?i)\b(skills?|kỹ năng|công nghệ|technologies)\b')
    exp_pattern = re.compile(r'(?i)\b(experience|kinh nghiệm|work history|lịch sử|dự án|projects?)\b')
    edu_pattern = re.compile(r'(?i)\b(education|học vấn|trình độ|bằng cấp|certifications?)\b')
    
    for line in text.split('\n'):
        line_strip = line.strip()
        if not line_strip:
            continue
            
        if len(line_strip) < 60:
            if skill_pattern.search(line_strip):
                current_section = "Skill"
                continue
            elif exp_pattern.search(line_strip):
                current_section = "Exp"
                continue
            elif edu_pattern.search(line_strip):
                current_section = "Edu"
                continue
                
        if current_section in sections:
            sections[current_section] += line_strip + "\n"
            
    return sections

def retrieve_and_build_context(
    raw_cv_list,
    jd_text,
    mandatory_keywords=None,
    blacklist_keywords=None,
    top_k=None,
):
    """
    Pipeline Hybrid RAG cập nhật:
    Tách CV thành các trường Skill/Exp/Edu thông qua heuristic và tính toán similarity 
    độc lập cho từng trường để có điểm tổng hợp hợp lý, chính xác hơn.
    """
    filtered_cvs = execute_hard_filtering_stage(raw_cv_list, mandatory_keywords, blacklist_keywords)
    
    if not filtered_cvs:
        return []
        
    model = get_bge_model()
    
    # 1. Bóc tách toàn bộ CV qua Heuristic
    parsed_cvs = []
    for item in filtered_cvs:
        cv_text = item.get("structured_data", {}).get("cv_content", "")
        parsed = heuristic_split_cv(cv_text)
        # Nếu một trường trống, ta gán 1 khoảng trắng để tránh lỗi tensor
        parsed_cvs.append({
            "Skill": parsed["Skill"] if parsed["Skill"] else " ",
            "Exp": parsed["Exp"] if parsed["Exp"] else " ",
            "Edu": parsed["Edu"] if parsed["Edu"] else " "
        })
    
    with torch.no_grad():
        # Encode JD (Có thể dùng toàn bộ JD để đối chiếu với từng phần của CV)
        jd_emb = model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
        
        # Encode từng trường độc lập
        cv_skills = [p["Skill"] for p in parsed_cvs]
        cv_exps = [p["Exp"] for p in parsed_cvs]
        cv_edus = [p["Edu"] for p in parsed_cvs]
        
        skill_embs = model.encode(cv_skills, convert_to_tensor=True, normalize_embeddings=True)
        exp_embs = model.encode(cv_exps, convert_to_tensor=True, normalize_embeddings=True)
        edu_embs = model.encode(cv_edus, convert_to_tensor=True, normalize_embeddings=True)
        
        # Tính Cosine Similarity riêng biệt cho từng mảng
        skill_scores = util.cos_sim(jd_emb, skill_embs)[0].cpu().numpy()
        exp_scores = util.cos_sim(jd_emb, exp_embs)[0].cpu().numpy()
        edu_scores = util.cos_sim(jd_emb, edu_embs)[0].cpu().numpy()
        
    for idx, item in enumerate(filtered_cvs):
        # Weighted Scoring (Điểm số hợp lý hơn nhờ focus vào các trọng số thực tế)
        # VD: Kỹ năng chiếm 50%, Kinh nghiệm chiếm 35%, Học vấn chiếm 15%
        weighted_score = (skill_scores[idx] * 0.50) + (exp_scores[idx] * 0.35) + (edu_scores[idx] * 0.15)
        item["semantic_score"] = float(weighted_score)
        
        # Lưu lại điểm thành phần để debug hoặc hiển thị
        item["component_scores"] = {
            "skill_score": float(skill_scores[idx]),
            "exp_score": float(exp_scores[idx]),
            "edu_score": float(edu_scores[idx])
        }
        
    sorted_cvs = sorted(filtered_cvs, key=lambda x: x["semantic_score"], reverse=True)
    top_k = top_k or LLMConfig.TOP_K_RAG
    top_candidates = sorted_cvs[:top_k]
    
    for rank, cv in enumerate(top_candidates, start=1):
        cv["rag_context"] = (
            f"--- ỨNG VIÊN TOP {rank} (Mã số file: {cv.get('file_name', 'N/A')}) ---\n"
            f"Điểm tương đồng ngữ nghĩa tổng hợp: {cv['semantic_score']:.4f}\n"
            f"(Kỹ năng: {cv['component_scores']['skill_score']:.2f} | Kinh nghiệm: {cv['component_scores']['exp_score']:.2f})\n"
            f"Nội dung hồ sơ:\n{cv.get('structured_data', {}).get('cv_content', '')}\n"
        )
        
    return top_candidates
