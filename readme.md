# CV-JD Screening Pipeline

Pipeline này sàng lọc và xếp hạng CV theo Job Description qua 3 tầng chính:

1. OCR: đọc CV trong thư mục `CV/` và chuẩn hóa thành `{"cv_content": "..."}`
2. Embedding retrieval: lọc keyword cứng, chấm điểm semantic và lấy top-k CV phù hợp nhất
3. LLM rerank: đánh giá sâu bằng LLM dựa trên `jd.txt` và `rule.txt`

## Cấu Trúc Thư Mục

```text
.
|-- main.py
|-- jd.txt
|-- rule.txt
|-- CV/
|-- OCR/
|-- Embedding/
|-- LLM/
`-- model/
```

## Cài Đặt

Tạo môi trường Python rồi cài các package cần thiết:

```bash
pip install openai torch pillow opencv-python numpy pandas pytesseract pdf2image transformers sentence-transformers peft accelerate
```

Nếu dùng OCR Tesseract, cần cài thêm Tesseract OCR trên hệ điều hành và đảm bảo lệnh `tesseract` có trong `PATH`.

Nếu đọc file PDF bằng Tesseract hoặc Qwen, cần cài thêm Poppler vì project dùng `pdf2image`.

## Chuẩn Bị Dữ Liệu

- Đặt CV vào thư mục `CV/`
- Định dạng hỗ trợ: `.pdf`, `.png`, `.jpg`, `.jpeg`
- Viết nội dung Job Description trong `jd.txt`
- Viết rule chấm điểm trong `rule.txt`
- Nếu muốn dùng `--max-rules`, nên để mỗi rule cách nhau bằng một dòng trống để hệ thống cắt rule chính xác

## Cấu Hình LLM

Trên PowerShell:

```powershell
$env:LLM_API_KEY="your_api_key"
```

Hoặc:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Các biến môi trường thường dùng:

```powershell
$env:LLM_MODEL_NAME="gpt-4o-mini"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:TOP_K_RAG="5"
$env:MAX_RULES="20"
```

Trên macOS/Linux:

```bash
export LLM_API_KEY="your_api_key"
export LLM_MODEL_NAME="gpt-4o-mini"
export LLM_BASE_URL="https://api.openai.com/v1"
```

## Chạy Mặc Định

```bash
python main.py
```

Mặc định:

- OCR: `tesseract`
- Embedding: `hybrid-bge`
- top-k: `5`
- max-rules: dùng toàn bộ rule trong `rule.txt`
- must-have keywords: `Python,Postgres`
- blacklist keywords: `PHP,WordPress`

## Tùy Chỉnh Khi Chạy

Chọn số CV đi tiếp sau tầng embedding và số rule tối đa cho LLM:

```bash
python main.py --top-k 3 --max-rules 20
```

Chọn OCR engine:

```bash
python main.py --ocr tesseract
python main.py --ocr qwen --qwen-model Qwen/Qwen2-VL-7B-Instruct
python main.py --ocr pix2struct
```

Chọn embedding backend:

```bash
python main.py --embedding hybrid-bge
python main.py --embedding bge
python main.py --embedding nomic
```

Tùy chỉnh keyword filter:

```bash
python main.py --must-have "Python,FastAPI,Postgres" --must-not-have "PHP,WordPress"
```

Tắt keyword filter:

```bash
python main.py --must-have "" --must-not-have ""
```

Dùng đường dẫn khác:

```bash
python main.py --cv-folder CV --jd-path jd.txt --rule-path rule.txt
```

## Flow Hiện Tại

`main.py` đã kết nối các tầng theo luồng sau:

```text
main.py
  -> OCR engine
      tesseract | qwen | pix2struct
  -> embedding retrieval
      hybrid-bge | bge | nomic
  -> LLM rerank
      đọc rule.txt, giới hạn bằng --max-rules, trả JSON final_score/decision
  -> in bảng xếp hạng cuối cùng
```

## Ghi Chú

- Nếu thiếu API key LLM, pipeline không bị crash ở bước rerank. Từng CV sẽ được gắn trạng thái `System error` và điểm fallback theo embedding score.
- `--top-k` quyết định số lượng CV từ tầng embedding được đưa sang LLM.
- `--max-rules 0` hoặc không truyền `--max-rules` nghĩa là dùng toàn bộ rule.
- Các model LoRA mặc định nằm trong thư mục `model/`.
