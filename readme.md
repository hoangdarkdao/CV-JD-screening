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
|-- .env.example
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

## Cấu Hình Bằng `.env`

Tạo file `.env` từ file mẫu:

```bash
copy .env.example .env
```

Trên macOS/Linux:

```bash
cp .env.example .env
```

Sau đó sửa các giá trị trong `.env`:

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

CV_FOLDER=CV
JD_PATH=jd.txt
RULE_TXT_PATH=rule.txt

OCR_ENGINE=tesseract
EMBEDDING_ENGINE=hybrid-bge

TOP_K_RAG=5
MAX_RULES=0
MUST_HAVE_KEYWORDS=Python,Postgres
MUST_NOT_HAVE_KEYWORDS=PHP,WordPress
```

Ghi chú:

- `.env` được tự động load khi chạy `main.py`.
- Biến môi trường thật của hệ điều hành có ưu tiên cao hơn giá trị trong `.env`.
- `.env` đã được đưa vào `.gitignore` để tránh commit API key.
- Có thể dùng `ENV_FILE=duong_dan_file_env` nếu muốn load file env khác.

## Chuẩn Bị Dữ Liệu

- Đặt CV vào thư mục `CV/`
- Định dạng hỗ trợ: `.pdf`, `.png`, `.jpg`, `.jpeg`
- Viết nội dung Job Description trong `jd.txt`
- Viết rule chấm điểm trong `rule.txt`
- Nếu muốn dùng `MAX_RULES` hoặc `--max-rules`, nên để mỗi rule cách nhau bằng một dòng trống để hệ thống cắt rule chính xác

## Chạy Mặc Định

```bash
python main.py
```

Mặc định lấy cấu hình từ `.env`. Nếu một biến không có trong `.env`, hệ thống dùng giá trị mặc định trong code.

## Tùy Chỉnh Khi Chạy

CLI argument có thể ghi đè cấu hình trong `.env`.

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

## `hybrid-bge` Là Gì?

`hybrid-bge` là backend retrieval mặc định trong `LLM/Rag.py`. Nó là "hybrid" vì kết hợp nhiều bước:

1. Hard filter: loại CV theo `MUST_HAVE_KEYWORDS` và `MUST_NOT_HAVE_KEYWORDS`
2. Heuristic split: tách nội dung CV thành các phần `Skill`, `Exp`, `Edu`
3. BGE embedding: dùng `BAAI/bge-m3` và LoRA trong `model/bge` nếu có
4. Weighted scoring: tính điểm tổng hợp theo trọng số `Skill 50%`, `Exp 35%`, `Edu 15%`
5. Top-k retrieval: lấy `TOP_K_RAG` CV tốt nhất đưa sang LLM rerank

Khác biệt với `bge` là `bge` trong `Embedding/bge_ranking.py` encode toàn bộ CV thành một đoạn text duy nhất. `hybrid-bge` tách CV theo section trước nên điểm semantic thường sát hơn khi CV có cấu trúc rõ.

## Giải Thích Từng Module

### `main.py`

Đây là entrypoint điều phối toàn bộ pipeline. File này chịu trách nhiệm:

- Đọc cấu hình từ `.env` và CLI argument
- Đọc `jd.txt`
- Tìm các file CV trong thư mục `CV/`
- Chọn OCR engine theo `OCR_ENGINE` hoặc `--ocr`
- Chọn embedding backend theo `EMBEDDING_ENGINE` hoặc `--embedding`
- Truyền `TOP_K_RAG` / `--top-k` sang tầng embedding
- Truyền `MAX_RULES` / `--max-rules` sang tầng LLM rerank
- In bảng xếp hạng cuối cùng

### `LLM/config.py`

Module cấu hình trung tâm. File này tự động load `.env`, sau đó tạo class `LLMConfig` để các module khác dùng chung.

Các nhóm cấu hình chính:

- LLM: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
- Đường dẫn: `CV_FOLDER`, `JD_PATH`, `RULE_TXT_PATH`
- Embedding: `EMBEDDING_BASE_MODEL`, `EMBEDDING_LORA_PATH`
- Pipeline: `TOP_K_RAG`, `MAX_RULES`

Biến môi trường thật của hệ điều hành có ưu tiên cao hơn `.env`.

### Nhóm OCR: `OCR/`

Các module OCR đều có nhiệm vụ đọc file CV và trả về dữ liệu chuẩn:

```python
{"cv_content": "..."}
```

| Engine | File | Cách hoạt động | Khi nào nên dùng |
| --- | --- | --- | --- |
| `tesseract` | `OCR/tesseract_heuristic.py` | Dùng Tesseract OCR, có heuristic nhận diện layout 1 cột/2 cột và gom section | Mặc định, nhẹ nhất, phù hợp chạy local |
| `qwen` | `OCR/qwen_ocr.py` | Dùng Qwen Vision-Language Model để đọc ảnh/PDF và sinh text có cấu trúc | Khi có GPU tốt và muốn OCR hiểu layout/phần nội dung tốt hơn |
| `pix2struct` | `OCR/pix2struct.py` | Dùng Pix2Struct + LoRA trong `model/pix2struct` | Khi muốn dùng model đã fine-tune riêng cho CV dạng ảnh |

### Nhóm Embedding: `Embedding/` Và `LLM/Rag.py`

Tầng embedding có 2 việc:

1. Lọc cứng CV bằng keyword bắt buộc và keyword loại trừ
2. Chấm điểm semantic giữa CV và JD để lấy top-k CV đưa sang LLM

| Backend | File chính | Cách tính điểm | Ưu điểm | Hạn chế |
| --- | --- | --- | --- | --- |
| `hybrid-bge` | `LLM/Rag.py` | Lọc keyword, tách CV thành `Skill`, `Exp`, `Edu`, encode từng phần bằng BGE-M3, tính điểm có trọng số | Sát nghiệp vụ hơn khi CV có cấu trúc rõ; mặc định nên dùng | Phụ thuộc heuristic tách section, nếu OCR quá lộn xộn thì điểm có thể nhiễu |
| `bge` | `Embedding/bge_ranking.py` | Encode toàn bộ CV thành một đoạn text và so cosine với JD bằng BGE-M3 | Đơn giản, ổn định, ít phụ thuộc section heading | Không phân biệt kỹ năng/kinh nghiệm/học vấn nên điểm có thể kém chi tiết hơn |
| `nomic` | `Embedding/nomic_ranking.py` | Dùng Nomic embedding với prefix `search_query:` cho JD và `search_document:` cho CV | Phù hợp thử nghiệm model embedding khác BGE; có asymmetric retrieval | Cần model Nomic hoặc fallback base model; kết quả phụ thuộc chất lượng model/local weights |

Chi tiết từng module:

- `Embedding/hard_filter.py`: lọc CV theo `MUST_HAVE_KEYWORDS` và `MUST_NOT_HAVE_KEYWORDS`.
- `Embedding/bge_ranking.py`: ranking semantic bằng BGE-M3 trên toàn bộ text CV.
- `Embedding/nomic_ranking.py`: ranking semantic bằng Nomic embedding.
- `LLM/Rag.py`: backend `hybrid-bge`, gồm hard filter, heuristic split, BGE scoring theo trọng số và tạo `rag_context`.

### Nhóm LLM: `LLM/`

| Module | Vai trò |
| --- | --- |
| `LLM/LLM_rerank.py` | Gọi LLM để chấm điểm top-k CV, đọc `rule.txt`, giới hạn rule bằng `MAX_RULES`, yêu cầu output JSON |
| `LLM/Rag.py` | Retrieval backend `hybrid-bge`, chuẩn bị ứng viên trước khi đưa sang LLM |
| `LLM/config.py` | Load `.env` và tập trung toàn bộ cấu hình |

`LLM_rerank.py` yêu cầu LLM trả về JSON dạng:

```json
{
  "final_score": 85,
  "decision": "Strong fit",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "detailed_justification": "..."
}
```

Nếu gọi LLM lỗi hoặc thiếu API key, pipeline không dừng hẳn. Candidate sẽ được gắn `System error` và dùng điểm fallback từ embedding.

### Thư Mục `model/`

Thư mục này chứa local LoRA/model weights:

| Thư mục | Dùng bởi | Vai trò |
| --- | --- | --- |
| `model/bge` | `LLM/Rag.py`, `Embedding/bge_ranking.py` | LoRA weights cho BGE-M3 |
| `model/nomic` | `Embedding/nomic_ranking.py` | Local/fine-tuned Nomic embedding |
| `model/pix2struct` | `OCR/pix2struct.py` | LoRA weights cho Pix2Struct OCR |

Nếu không tìm thấy LoRA tương ứng, một số module sẽ fallback về base model.

## Nên Chọn Backend Nào?

Khuyến nghị thực tế:

| Mục tiêu | OCR nên dùng | Embedding nên dùng |
| --- | --- | --- |
| Chạy nhanh, ít phụ thuộc GPU | `tesseract` | `hybrid-bge` hoặc `bge` |
| CV scan/ảnh phức tạp, cần hiểu layout tốt hơn | `qwen` | `hybrid-bge` |
| Muốn baseline đơn giản để debug | `tesseract` | `bge` |
| Muốn thử embedding model khác BGE | `tesseract` hoặc `qwen` | `nomic` |
| CV có section rõ ràng như Skills/Experience/Education | bất kỳ | `hybrid-bge` |
| OCR bị lộn section nhiều | bất kỳ | `bge` có thể ổn định hơn |

## Flow Hiện Tại

```text
main.py
  -> load .env
  -> OCR engine
      tesseract | qwen | pix2struct
  -> embedding retrieval
      hybrid-bge | bge | nomic
  -> LLM rerank
      đọc rule.txt, giới hạn bằng MAX_RULES/--max-rules, trả JSON final_score/decision
  -> in bảng xếp hạng cuối cùng
```

## Ghi Chú

- Nếu thiếu API key LLM, pipeline không bị crash ở bước rerank. Từng CV sẽ được gắn trạng thái `System error` và điểm fallback theo embedding score.
- `TOP_K_RAG` hoặc `--top-k` quyết định số lượng CV từ tầng embedding được đưa sang LLM.
- `MAX_RULES=0` hoặc `--max-rules 0` nghĩa là dùng toàn bộ rule.
- Các model LoRA mặc định nằm trong thư mục `model/`.
