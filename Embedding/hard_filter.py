import re

def match_keyword_exact(text, keyword):
    """
    Sử dụng cơ chế Lookaround (?<!...) và (?!...) trong Regex để so khớp chính xác từ khóa IT.
    Giải quyết triệt để lỗi của biên từ ranh giới '\\b' truyền thống khi gặp ký tự (+, #, .).
    
    Ví dụ: 
    - Không bắt nhầm 'Java' trong 'JavaScript'.
    - Không bắt nhầm 'C' trong 'C++' hoặc 'C#'.
    - Nhận diện đúng '.NET', 'Node.js'.
    """
    cv_lower = text.lower()
    kw_lower = keyword.lower()
    
    # Ép ký tự đặc biệt về chuỗi an toàn cho Regex
    escaped_kw = re.escape(kw_lower)
    
    # Định nghĩa ranh giới: Trước và sau từ khóa không được là ký tự chữ cái hoặc chữ số
    # Điều này cho phép các ký tự như dấu cộng (+), dấu thăng (#), dấu chấm (.) đứng độc lập hoặc sát lề
    pattern = r'(?<![a-zA-Z0-9])' + escaped_kw + r'(?![a-zA-Z0-9])'
    
    return bool(re.search(pattern, cv_lower))


def evaluate_hard_rules(cv_content, must_have_keywords=None, must_not_have_keywords=None):
    """
    Hàm kiểm tra điều kiện cứng (Hard Gatekeeper) cho một văn bản CV cụ thể:
    - must_have_keywords: Danh sách các từ khóa BẮT BUỘC phải xuất hiện đồng thời (Phép toán AND).
    - must_not_have_keywords: Danh sách các từ khóa ĐEN (Blacklist), nếu xuất hiện sẽ loại ngay lập tức.
    
    Returns:
        bool: True nếu CV đủ điều kiện đi tiếp vào tầng RAG, False nếu bị loại bỏ.
    """
    if not cv_content:
        return False
        
    # 1. KIỂM TRA BỘ LỌC ĐEN (MUST-NOT-HAVE)
    # Nếu CV chứa bất kỳ từ khóa cấm nào (Ví dụ: tuyển Senior nhưng CV chứa 'Intern'/'Fresher' quá dày) -> Loại luôn
    if must_not_have_keywords:
        for black_kw in must_not_have_keywords:
            if black_kw.strip() and match_keyword_exact(cv_content, black_kw.strip()):
                print(f"[HARD FILTER] LOẠI vì chứa từ khóa cấm: '{black_kw}'")
                return False

    # 2. KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT (MUST-HAVE)
    # Ứng viên phải đáp ứng đầy đủ TẤT CẢ các kỹ năng bắt buộc được liệt kê
    if must_have_keywords:
        for req_kw in must_have_keywords:
            if req_kw.strip() and not match_keyword_exact(cv_content, req_kw.strip()):
                # Chỉ cần thiếu 1 công nghệ cốt lõi -> Không vượt qua vòng lọc hồ sơ sơ bộ
                return False
                
    return True


def execute_hard_filtering_stage(raw_cv_list, must_have_keywords=None, must_not_have_keywords=None):
    """
    HÀM GIAO DIỆN CHÍNH (MAIN INTERFACE):
    Duyệt qua danh sách CV thô ban đầu để lọc ra những ứng viên đạt tiêu chuẩn cứng.
    
    Parameters:
    - raw_cv_list (list): Danh sách các đối tượng CV (Mỗi đối tượng chứa key 'structured_data')
    
    Returns:
    - list: Danh sách các đối tượng CV vượt qua bộ lọc khắt khe.
    """
    passed_candidates = []
    
    print(f"[HARD FILTER] Bắt đầu quét điều kiện cứng cho {len(raw_cv_list)} ứng viên...")
    print(f" -> Từ khóa bắt buộc (AND): {must_have_keywords}")
    print(f" -> Từ khóa loại trừ (Blacklist): {must_not_have_keywords}")
    
    for cv_item in raw_cv_list:
        # Bóc tách nội dung văn bản đã được xử lý qua tầng OCR trước đó
        cv_text = cv_item.get("structured_data", {}).get("cv_content", "")
        file_name = cv_item.get("file_name", "Unknown_File")
        
        if evaluate_hard_rules(cv_text, must_have_keywords, must_not_have_keywords):
            passed_candidates.append(cv_item)
        else:
            print(f" [-] File '{file_name}' không vượt qua bộ lọc từ khóa cứng.")
            
    print(f"[HARD FILTER] Hoàn thành Tầng 2: Giữ lại {len(passed_candidates)}/{len(raw_cv_list)} ứng viên đạt chuẩn.")
    return passed_candidates