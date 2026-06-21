import os
import json
from OCR.qwen_ocr import ocr_qwen_vlm
from LLM.Rag import retrieve_and_build_context
from LLM.LLM_rerank import run_full_llm_reranking_pipeline
from LLM.config import LLMConfig
from OCR.pix2struct import ocr_pix2struct_cv
from OCR.tesseract_heuristic import ocr_tesseract_heuristic
from PIL import Image

def load_jd_and_keywords(jd_path="jd.txt"):
    """
    Đọc nội dung Job Description từ tệp tin jd.txt bên ngoài.
    Nếu không tìm thấy file, hệ thống sẽ tự động tạo một file mẫu để tránh lỗi.
    """
    default_jd = (
        "Vị trí tuyển dụng: Kỹ sư Python / Backend Developer (Middle)\n"
        "Yêu cầu công việc:\n"
        "- Có từ 2 năm kinh nghiệm làm việc với ngôn ngữ lập trình Python và Framework Django/FastAPI.\n"
        "- Thành thạo cơ sở dữ liệu quan hệ PostgreSQL hoặc MySQL.\n"
        "- Có tư duy tốt về cấu trúc dữ liệu, thuật toán và thiết kế hệ thống Microservices.\n"
        "- Khả năng giao tiếp tiếng Anh tốt (đọc hiểu tài liệu kỹ thuật chuyên sâu)."
    )
    
    # Kiểm tra xem file jd.txt có tồn tại không
    if os.path.exists(jd_path):
        print(f"[STAGE 0] Đang nạp dữ liệu mô tả công việc từ tệp: '{jd_path}'")
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()
    else:
        print(f"[WARNING] Không tìm thấy file '{jd_path}'. Hệ thống tự động tạo file mẫu.")
        with open(jd_path, "w", encoding="utf-8") as f:
            f.write(default_jd)
        jd_text = default_jd
        
    # Bạn vẫn giữ bộ từ khóa lọc cứng ở đây hoặc có thể tùy biến đọc từ dòng đầu của file nếu muốn
    must_have_keywords = ["Python", "Postgres"]  # Bắt buộc phải xuất hiện (Phép toán AND)
    must_not_have_keywords = ["PHP", "WordPress"] # Từ khóa cấm (Blacklist loại ngay lập tức)
    
    return jd_text, must_have_keywords, must_not_have_keywords


def main():
    print("="*70)
    print("     KHỞI CHẠY HỆ THỐNG SÀNG LỌC & XẾP HẠNG CV THÔNG MINH (AI-HR)     ")
    print("="*70)

    # 0. Nạp dữ liệu từ file jd.txt bên ngoài thay vì hardcode
    cv_folder = "CV"  
    jd_text, must_have_keywords, must_not_have_keywords = load_jd_and_keywords("jd.txt")
    
    if not os.path.exists(cv_folder):
        print(f"[INFO] Tạo mới thư mục '{cv_folder}/'. Vui lòng bỏ các file CV vào đây để quét.")
        os.makedirs(cv_folder)
        return

    # Lấy danh sách các tệp tin CV hỗ trợ
    supported_extensions = ('.pdf', '.png', '.jpg', '.jpeg')
    cv_files = [f for f in os.listdir(cv_folder) if f.lower().endswith(supported_extensions)]
    
    if not cv_files:
        print(f"[WARNING] Thư mục '{cv_folder}/' đang trống. Hãy thêm các file .pdf hoặc .png vào.")
        return

    print(f"[STAGE 0] Tìm thấy {len(cv_files)} tệp tin hồ sơ ứng viên chuẩn bị đưa vào xử lý.")

    # ----------------------------------------------------------------------------------
    # TẦNG 1: OCR & DOCUMENT PARSING (Sử dụng Qwen VLM nâng cao)
    # ----------------------------------------------------------------------------------
    print("\n" + "="*50)
    print("[STAGE 1] BẮT ĐẦU QUÁ TRÌNH OCR & PHÂN TÁCH CẤU TRÚC BẰNG QWEN VLM")
    print("="*50)
    
    raw_cv_list = []
    for filename in cv_files:
        file_path = os.path.join(cv_folder, filename)
        print(f"\n[OCR] Đang xử lý tài liệu: {filename}")
        try:
            ocr_result = ocr_tesseract_heuristic(file_path)
            cv_item = {
                "file_name": filename,
                "file_path": file_path,
                "structured_data": ocr_result  
            }
            raw_cv_list.append(cv_item)
        except Exception as e:
            print(f"[-] Bỏ qua file '{filename}' do gặp lỗi OCR hệ thống: {e}")

    if not raw_cv_list:
        print("[-] Không có dữ liệu CV nào được trích xuất thành công. Dừng luồng Pipeline.")
        return

    # ----------------------------------------------------------------------------------
    # TẦNG 2 & 3: HYBRID RAG RETRIEVAL (Hard Keyword Filter + BGE-M3 LoRA Semantic Scoring)
    # ----------------------------------------------------------------------------------
    print("\n" + "="*50)
    print("[STAGE 2 & 3] BẮT ĐẦU PHỄU LỌC HYBRID RAG (TỪ KHÓA CỨNG + NGỮ NGHĨA VECTOR)")
    print("="*50)
    
    rag_top_candidates = retrieve_and_build_context(
        raw_cv_list=raw_cv_list,
        jd_text=jd_text,
        mandatory_keywords=must_have_keywords,
        blacklist_keywords=must_not_have_keywords
    )

    if not rag_top_candidates:
        print("[WARNING] Không có ứng viên nào vượt qua phễu lọc thô RAG.")
        return

    # ----------------------------------------------------------------------------------
    # TẦNG 4: DEEP LLM RERANKING & DECISION (Phân tích chuyên sâu kết hợp tuân thủ rule.txt)
    # ----------------------------------------------------------------------------------
    print("\n" + "="*50)
    print("[STAGE 4] BẮT ĐẦU GIAI ĐOẠN ĐÁNH GIÁ CHUYÊN SÂU BẰNG LLM RERANKING")
    print("="*50)
    
    final_ranked_results = run_full_llm_reranking_pipeline(rag_top_candidates, jd_text)

    # ----------------------------------------------------------------------------------
    # IN KẾT QUẢ CUỐI CÙNG (OUTPUT REPORT GENERATION)
    # ----------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("                BẢNG XẾP HẠNG ỨNG VIÊN CUỐI CÙNG (FINAL RANKING)               ")
    print("="*70)
    
    print(f"{'Hạng':<5} | {'Tên File CV':<25} | {'Điểm LLM':<10} | {'Điểm BGE':<10} | {'Quyết định':<15}")
    print("-" * 75)
    
    for idx, cv in enumerate(final_ranked_results, start=1):
        file_name = cv.get("file_name", "Unknown")
        llm_score = cv.get("final_score", 0)
        bge_score = cv.get("semantic_score", 0.0)
        decision = cv.get("decision", "N/A")
        
        if len(file_name) > 23:
            file_name = file_name[:20] + "..."
            
        print(f"{idx:<5} | {file_name:<25} | {llm_score:<10} | {bge_score:<10.4f} | {decision:<15}")

    print("\n" + "="*70)
    print("            BÁO CÁO PHÂN TÍCH CHI TIẾT ỨNG VIÊN XUẤT SẮC NHẤT (TOP 1)          ")
    print("="*70)
    
    top_1_cv = final_ranked_results[0]
    evaluation = top_1_cv.get("llm_evaluation", {})
    
    print(f"[*] Tên tệp tin: {top_1_cv.get('file_name')}")
    print(f"[*] Điểm số toàn diện: {top_1_cv.get('final_score')}/100 điểm.")
    print(f"[*] Trạng thái đề xuất: {top_1_cv.get('decision')}")
    
    print("\n[+] Điểm mạnh thấu khớp kỹ năng:")
    for strength in evaluation.get("strengths", []):
        print(f"   - {strength}")
        
    print("\n[-] Điểm hạn chế / Rủi ro đối chiếu rule.txt:")
    for weakness in evaluation.get("weaknesses", []):
        print(f"   - {weakness}")
        
    print(f"\n[=>] Lập luận giải trình của AI:")
    print(evaluation.get("detailed_justification", "Không có giải trình cụ thể."))
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Pipeline gặp lỗi nghiêm trọng: {e}")
        exit(1)