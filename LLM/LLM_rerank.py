import os
import re
import json
from openai import OpenAI
from LLM.config import LLMConfig

def load_screening_rules(rule_path=LLMConfig.RULE_TXT_PATH):
    """Đọc tệp tin rule.txt để nạp các quy tắc chấm điểm và sàng lọc từ hệ thống."""
    if not os.path.exists(rule_path):
        print(f"[LLM WARNING] Không tìm thấy file quy tắc tại '{rule_path}'. Sử dụng bộ quy tắc mặc định.")
        return (
            "1. Thang điểm từ 0 đến 100 dựa trên độ khớp kỹ năng công nghệ.\n"
            "2. Trừ 20 điểm nếu ứng viên có thời gian làm việc dưới 6 tháng tại các vị trí gần nhất.\n"
            "3. Ưu tiên các ứng viên có chứng chỉ chuyên môn quốc tế hoặc giải thưởng học thuật."
        )
    with open(rule_path, "r", encoding="utf-8") as f:
        return f.read()

def heuristic_split_cv(text):
    """Sử dụng heuristic (từ khóa và độ dài dòng) để bóc tách CV thành các trường cụ thể."""
    sections = {"Skill": "", "Exp": "", "Edu": "", "Other": ""}
    current_section = "Other"
    
    skill_pattern = re.compile(r'(?i)\b(skills?|kỹ năng|công nghệ|technologies|tech stack)\b')
    exp_pattern = re.compile(r'(?i)\b(experience|kinh nghiệm|work history|lịch sử làm việc|employment|dự án|projects?)\b')
    edu_pattern = re.compile(r'(?i)\b(education|học vấn|trình độ|bằng cấp|certifications?|đại học|university)\b')
    
    for line in text.split('\n'):
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Heuristic: Các tiêu đề (header) thường có độ dài ngắn
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
                
        sections[current_section] += line_strip + "\n"
        
    return sections

def llm_rerank_candidate(candidate_cv, jd_text):
    """
    Sử dụng LLM chấm điểm chi tiết. CV đã được tách trường bằng Heuristic 
    để LLM dễ dàng phân tích và đối chiếu luật hơn.
    """
    client = OpenAI(
        api_key=LLMConfig.API_KEY,
        base_url=LLMConfig.BASE_URL
    )
    
    rules = load_screening_rules()
    cv_content = candidate_cv.get("structured_data", {}).get("cv_content", "")
    file_name = candidate_cv.get("file_name", "Unknown_CV")
    
    # Bóc tách CV bằng Heuristic trước khi đưa cho LLM
    parsed_cv = heuristic_split_cv(cv_content)
    structured_cv_text = (
        f"[KỸ NĂNG - SKILLS]\n{parsed_cv['Skill']}\n\n"
        f"[KINH NGHIỆM - EXPERIENCE]\n{parsed_cv['Exp']}\n\n"
        f"[HỌC VẤN - EDUCATION]\n{parsed_cv['Edu']}\n\n"
        f"[THÔNG TIN KHÁC]\n{parsed_cv['Other']}"
    )
    
    system_prompt = (
        "Bạn là một chuyên gia tuyển dụng cấp cao và kỹ sư phân tích nhân sự AI hệ thống.\n"
        "Nhiệm vụ của bạn là đánh giá chuyên sâu một hồ sơ ứng viên (CV) dựa trên bản Mô tả công việc (JD) "
        "và Tuân thủ nghiêm ngặt tập Quy tắc Sàng lọc dưới đây.\n\n"
        f"=== TẬP QUY TẮC SÀNG LỌC ===\n{rules}\n\n"
        "Bạn PHẢI trả về kết quả dưới định dạng JSON thuần túy, không bao gồm khối markdown ```json, "
        "với cấu trúc chính xác như sau:\n"
        "{\n"
        '  "final_score": <int từ 0 đến 100>,\n'
        '  "strengths": [<danh sách các điểm mạnh>],\n'
        '  "weaknesses": [<danh sách các điểm thiếu sót hoặc vi phạm quy tắc>],\n'
        '  "detailed_justification": "Đoạn văn phân tích giải trình logic minh bạch."\n'
        "}"
    )
    
    user_prompt = (
        f"--- TÊN FILE HỒ SƠ: {file_name} ---\n\n"
        f"=== BẢN MÔ TẢ CÔNG VIỆC (JD) ===\n{jd_text}\n\n"
        f"=== NỘI DUNG HỒ SƠ ỨNG VIÊN (ĐÃ BÓC TÁCH) ===\n{structured_cv_text}\n"
    )
    
    try:
        response = client.chat.completions.create(
            model=LLMConfig.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        
        result_json = json.loads(response.choices[0].message.content)
        
        candidate_cv["llm_evaluation"] = result_json
        candidate_cv["final_score"] = result_json.get("final_score", 0)
        candidate_cv["decision"] = result_json.get("decision", "Cần xem xét lại")
        candidate_cv["parsed_sections"] = parsed_cv  # Lưu lại để track log nếu cần
        
        return candidate_cv
        
    except Exception as e:
        print(f"[-] Xảy ra lỗi xử lý LLM cho file {file_name}: {e}")
        candidate_cv["llm_evaluation"] = {
            "final_score": int(candidate_cv.get("semantic_score", 0) * 100),
            "decision": "Lỗi xử lý hệ thống",
            "detailed_justification": f"Không thể phân tích sâu bằng LLM do lỗi kỹ thuật: {str(e)}"
        }
        candidate_cv["final_score"] = int(candidate_cv.get("semantic_score", 0) * 100)
        candidate_cv["decision"] = "Lỗi xử lý hệ thống"
        return candidate_cv

def run_full_llm_reranking_pipeline(rag_top_candidates, jd_text):
    final_ranked_results = []
    print(f"[LLM INFO] Bắt đầu gọi LLM Reranking phân tích sâu cho {len(rag_top_candidates)} ứng viên...")
    
    for cv in rag_top_candidates:
        evaluated_cv = llm_rerank_candidate(cv, jd_text)
        final_ranked_results.append(evaluated_cv)
        
    final_ranked_results = sorted(final_ranked_results, key=lambda x: x["final_score"], reverse=True)
    return final_ranked_results