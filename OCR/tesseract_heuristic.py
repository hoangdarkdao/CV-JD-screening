import os
import re
import cv2
import json
import numpy as np
import pandas as pd
import pytesseract

# Hỗ trợ chuyển đổi file PDF thành hình ảnh nếu đầu vào là định dạng PDF
try:
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

# DANH SÁCH TỪ KHÓA MỤC LỚN TIÊU CHUẨN ĐƯỢC PHỤC HỒI TỪ NOTEBOOK
MAJOR_KEYWORDS = [
    "SUMMARY",
    "EDUCATION",
    "SKILLS",
    "PROJECT",
    "PROJECTS",
    "EXPERIENCE",
    "HONORS",
    "AWARDS",
    "LANGUAGES",
    "CERTIFICATIONS",
    "CORE COMPETENCIES",
    "BACKGROUND",
    "LEADERSHIP",
    "OBJECTIVE",
]

def check_cv_layout(img, threshold=30):
    """
    Bước 1: Kiểm tra cấu trúc CV là 1 cột hay 2 cột dựa trên khoảng cách màu 
    của hai pixel vùng đáy sát biên trái và biên phải.
    """
    h, w, _ = img.shape
    
    # Lấy tọa độ pixel góc dưới bên trái và góc dưới bên phải (tránh lỗi tràn biên)
    y_coord = min(int(h * 0.99), h - 1)
    x_left = min(int(w * 0.09), w - 1)
    x_right = min(int(w * 0.99), w - 1)
    
    pixel_left = img[y_coord, x_left]
    pixel_right = img[y_coord, x_right]
    
    # Tính khoảng cách màu sắc L2 (Euclidean distance) giữa 2 pixel
    color_distance = np.linalg.norm(pixel_left - pixel_right)
    is_two_column = color_distance > threshold
    
    print(f"[INFO] Khoảng cách màu sắc góc dưới: {color_distance:.2f} -> Layout: {'2 CỘT' if is_two_column else '1 CỘT'}")
    return is_two_column


def split_cv_columns(img, is_two_column):
    """
    Bước 2: Sử dụng thuật toán quét đáy (vùng không gian 95%-98%) tìm điểm chuyển màu
    đột ngột nhất để phân tách chính xác cột trái và cột phải của CV.
    """
    h, w, _ = img.shape
    
    if not is_two_column:
        return [img]
    
    # Lấy vùng không gian từ 95% đến 98% chiều cao ảnh để triệt tiêu nhiễu footer
    bottom_zone = img[int(h * 0.95):int(h * 0.98), :, :]
    if bottom_zone.size == 0:
        return [img]
        
    # Tính trung vị theo chiều dọc của vùng này
    median_profile = np.median(bottom_zone, axis=0).astype(np.uint8)
    
    # Chuyển sang ảnh xám để tìm điểm chuyển màu rõ rệt nhất
    gray_profile = cv2.cvtColor(median_profile.reshape(1, w, 3), cv2.COLOR_BGR2GRAY).flatten()
    
    # Tính độ chênh lệch màu giữa các pixel liên tiếp
    color_diff = np.abs(np.diff(gray_profile.astype(float)))
    
    # Giới hạn vùng tìm kiếm ở nửa biên bên trái CV (từ 15% đến 50% chiều rộng)
    start_x = int(w * 0.15)
    end_x = int(w * 0.5)
    
    if start_x >= end_x or end_x > len(color_diff):
        return [img]
        
    # Tìm vị trí màu thay đổi đột ngột nhất (biên phân tách màu lề kem/trắng)
    split_x = start_x + np.argmax(color_diff[start_x:end_x])
    split_x += 2  # Cộng thêm khoảng đệm nhỏ chống dính lề
    
    print(f"[INFO] Phân tách thành công bằng phương pháp quét đáy tại trục X = {split_x}")
    
    # Tiến hành cắt ảnh thành 2 thực thể cột độc lập
    left_col = img[:, :split_x]
    right_col = img[:, split_x:]
    
    return [left_col, right_col]


def ocr_extract_cv_fields(column_images):
    """
    Bước 3: Thực hiện nhận dạng ký tự bằng Tesseract theo trục dọc của từng cột,
    tự động phân loại khối tiêu đề lớn dựa trên kích thước chữ và từ khóa tiêu chuẩn.
    """
    final_result = {}

    for idx, col_img in enumerate(column_images):
        gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
        try:
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"[WARNING] Lỗi khi gọi Pytesseract: {e}")
            continue
            
        df = pd.DataFrame(ocr_data)
        if df.empty or "text" not in df.columns:
            continue

        # Loại bỏ các khoảng trắng vô nghĩa
        df = df[df["text"].str.strip() != ""]
        if df.empty:
            continue

        # Khôi phục dòng chữ (line_id) từ block, paragraph và line number
        df["line_id"] = (
            df["block_num"].astype(str)
            + "_"
            + df["par_num"].astype(str)
            + "_"
            + df["line_num"].astype(str)
        )

        lines = []
        for line_id, group in df.groupby("line_id"):
            sorted_words = group.sort_values(by="left")
            line_text = " ".join(sorted_words["text"].values)
            avg_height = sorted_words["height"].mean()
            min_top = sorted_words["top"].min()

            lines.append(
                {"text": line_text, "height": avg_height, "top": min_top}
            )

        # Sắp xếp các dòng tuần tự từ trên xuống dưới theo tọa độ Y (top)
        lines = sorted(lines, key=lambda x: x["top"])

        all_heights = [l["height"] for l in lines]
        median_height = np.median(all_heights) if all_heights else 12
        header_height_threshold = median_height * 1.2

        current_field = "GENERAL"

        for line in lines:
            text = line["text"].strip()

            # Chuẩn hóa loại bỏ các ký tự dấu chấm đầu dòng nguyên bản
            if text.startswith(("•", "-", "*", "+")):
                text = text[1:].strip()

            contains_date = bool(re.search(r"\\d{2}/\\d{4}|\\d{4}", text))
            text_upper = text.upper()
            is_major_header = False

            # Điều kiện nhận diện một Mục Lớn (Header): Chữ in hoa HOẶC kích thước vượt ngưỡng và không chứa ngày tháng
            if (text.isupper() or line["height"] > header_height_threshold) and not contains_date:
                if any(kw in text_upper for kw in MAJOR_KEYWORDS):
                    is_major_header = True

            if is_major_header:
                current_field = text
                if current_field not in final_result:
                    final_result[current_field] = []
            else:
                if current_field not in final_result:
                    final_result[current_field] = []
                if text:
                    final_result[current_field].append(text)

    return final_result


def format_cv_to_json(final_result):
    """
    Bước 4: Định dạng hậu kỳ cấu trúc văn bản thô thành một chuỗi văn bản hoàn chỉnh 
    và đóng gói vào một đối tượng Dictionary chứa khóa 'cv_content'.
    """
    cv_body_lines = []

    for field, values in final_result.items():
        if not values:
            continue

        field_header = field.strip().upper()
        cv_body_lines.append(field_header)

        section_lines = []
        for text_line in values:
            text_line = text_line.strip()
            if not text_line:
                continue

            # Tự động nhận diện các phần kỹ năng, kinh nghiệm hoặc dự án để sinh dấu gạch đầu dòng (-)
            is_bullet_section = any(
                kw in field_header
                for kw in ["SKILL", "EXPERIENCE", "PROJECT", "CONTRIBUTION"]
            )

            if is_bullet_section:
                # Không thêm dấu gạch đầu dòng nếu dòng đó chứa mốc thời gian hoặc thông tin phân cách phụ
                contains_date = bool(
                    re.search(r"\\d{4}|Present|June|January|December|May", text_line)
                )
                is_profile_header = "|" in text_line or text_line.isupper()

                if contains_date or is_profile_header:
                    section_lines.append(text_line)
                else:
                    section_lines.append(f"- {text_line}")
            else:
                section_lines.append(text_line)

        cv_body_lines.append("\n".join(section_lines))

    # Nối các phân vùng dữ liệu bằng 2 dấu xuống dòng liên tiếp
    cv_content_string = "\n\n".join(cv_body_lines)
    
    # Trả về dưới dạng dictionary để module main.py có thể xử lý mượt mà
    return {"cv_content": cv_content_string}


def ocr_tesseract_heuristic(cv_path):
    """
    HÀM ĐIỀU PHỐI CHÍNH (MAIN INTERFACE):
    Hỗ trợ xử lý tự động cả file ảnh đơn thông thường (.jpg, .png) lẫn tài liệu PDF nhiều trang.
    """
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"Không tìm thấy file CV tại đường dẫn: {cv_path}")
        
    ext = os.path.splitext(cv_path)[1].lower()
    images_to_process = []
    
    # 1. Cơ chế đọc và chuyển đổi tài liệu đầu vào thành ma trận ảnh xử lý
    if ext == '.pdf':
        if not HAS_PDF2IMAGE:
            raise ImportError(
                "Để phân tích CV dạng file PDF, vui lòng cài đặt thư viện 'pdf2image' "
                "bằng lệnh: pip install pdf2image (Yêu cầu hệ thống đã cài đặt Poppler)."
            )
        # Chuyển đổi các trang PDF thành danh sách ảnh PIL
        pil_images = pdf2image.convert_from_path(cv_path)
        for pil_img in pil_images:
            # Chuyển đổi định dạng từ PIL Image sang OpenCV BGR
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images_to_process.append(cv_img)
    else:
        # Xử lý các định dạng ảnh tiêu chuẩn (PNG, JPG, JPEG,...)
        img = cv2.imread(cv_path)
        if img is None:
            raise ValueError(f"Không thể mở hoặc đọc nội dung file ảnh: {cv_path}")
        images_to_process.append(img)
        
    all_cv_fields = {}
    
    # 2. Vòng lặp duyệt qua từng trang/hình ảnh của hồ sơ ứng viên
    for img in images_to_process:
        # Kiểm tra Layout (1 cột hay 2 cột)
        is_two_column = check_cv_layout(img)
        
        # Tiến hành tách cột dựa trên giải thuật quét đáy
        columns = split_cv_columns(img, is_two_column)
        
        # Thực hiện trích xuất OCR cấu trúc thô trên các vùng ảnh đã xử lý
        page_fields = ocr_extract_cv_fields(columns)
        
        # Gộp dữ liệu từ các trang vào một bộ nhớ trường dữ liệu duy nhất
        for field, lines in page_fields.items():
            if field not in all_cv_fields:
                all_cv_fields[field] = []
            all_cv_fields[field].extend(lines)
            
    # 3. Định dạng cấu trúc phân cấp đầu ra chuẩn
    structured_output = format_cv_to_json(all_cv_fields)
    return structured_output