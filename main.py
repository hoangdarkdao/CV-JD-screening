import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from LLM.config import LLMConfig


SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")
DEFAULT_JD_TEXT = (
    "Position: Python / Backend Developer (Middle)\n"
    "Requirements:\n"
    "- At least 2 years of experience with Python and Django/FastAPI.\n"
    "- Strong knowledge of PostgreSQL or MySQL.\n"
    "- Good understanding of data structures, algorithms, and microservices.\n"
    "- Good English reading skills for technical documentation."
)


@dataclass
class PipelineSettings:
    cv_folder: str
    jd_path: str
    rule_path: str
    ocr_engine: str
    embedding_engine: str
    top_k: int
    max_rules: Optional[int]
    must_have_keywords: List[str]
    must_not_have_keywords: List[str]
    qwen_model: Optional[str] = None


def parse_keywords(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def optional_positive_int(raw_value: str) -> Optional[int]:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    value = int(raw_value)
    if value <= 0:
        return None
    return value


def load_jd(jd_path: str) -> str:
    if os.path.exists(jd_path):
        print(f"[STAGE 0] Loading JD from: {jd_path}")
        with open(jd_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    print(f"[WARNING] JD file not found at '{jd_path}'. Creating a sample file.")
    with open(jd_path, "w", encoding="utf-8") as file:
        file.write(DEFAULT_JD_TEXT)
    return DEFAULT_JD_TEXT


def discover_cv_files(cv_folder: str) -> List[str]:
    if not os.path.exists(cv_folder):
        print(f"[INFO] Creating '{cv_folder}/'. Add CV files there and run again.")
        os.makedirs(cv_folder)
        return []

    return sorted(
        file_name
        for file_name in os.listdir(cv_folder)
        if file_name.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def resolve_ocr_runner(settings: PipelineSettings) -> Callable[[str], Dict[str, str]]:
    if settings.ocr_engine == "tesseract":
        from OCR.tesseract_heuristic import ocr_tesseract_heuristic

        return ocr_tesseract_heuristic

    if settings.ocr_engine == "qwen":
        from OCR.qwen_ocr import ocr_qwen_vlm

        if settings.qwen_model:
            return lambda file_path: ocr_qwen_vlm(file_path, model_name=settings.qwen_model)
        return ocr_qwen_vlm

    if settings.ocr_engine == "pix2struct":
        from OCR.pix2struct import ocr_pix2struct_cv

        return ocr_pix2struct_cv

    raise ValueError(f"Unsupported OCR engine: {settings.ocr_engine}")


def normalize_candidate_scores(candidates: List[Dict]) -> List[Dict]:
    for candidate in candidates:
        if "semantic_score" not in candidate and "embedding_score" in candidate:
            candidate["semantic_score"] = float(candidate["embedding_score"])
    return candidates


def retrieve_candidates(
    raw_cv_list: List[Dict],
    jd_text: str,
    settings: PipelineSettings,
) -> List[Dict]:
    if settings.embedding_engine == "hybrid-bge":
        from LLM.Rag import retrieve_and_build_context

        candidates = retrieve_and_build_context(
            raw_cv_list=raw_cv_list,
            jd_text=jd_text,
            mandatory_keywords=settings.must_have_keywords,
            blacklist_keywords=settings.must_not_have_keywords,
            top_k=settings.top_k,
        )
        return normalize_candidate_scores(candidates)

    from Embedding.hard_filter import execute_hard_filtering_stage

    filtered_cvs = execute_hard_filtering_stage(
        raw_cv_list,
        settings.must_have_keywords,
        settings.must_not_have_keywords,
    )
    if not filtered_cvs:
        return []

    if settings.embedding_engine == "bge":
        from Embedding.bge_ranking import bge_semantic_ranking

        candidates = bge_semantic_ranking(filtered_cvs, jd_text, top_k=settings.top_k)
        return normalize_candidate_scores(candidates)

    if settings.embedding_engine == "nomic":
        from Embedding.nomic_ranking import nomic_semantic_ranking

        candidates = nomic_semantic_ranking(filtered_cvs, jd_text, top_k=settings.top_k)
        return normalize_candidate_scores(candidates)

    raise ValueError(f"Unsupported embedding engine: {settings.embedding_engine}")


def run_ocr_stage(cv_files: List[str], settings: PipelineSettings) -> List[Dict]:
    ocr_runner = resolve_ocr_runner(settings)
    raw_cv_list = []

    for file_name in cv_files:
        file_path = os.path.join(settings.cv_folder, file_name)
        print(f"\n[OCR] Processing: {file_name}")
        try:
            ocr_result = ocr_runner(file_path)
            if not isinstance(ocr_result, dict):
                ocr_result = {"cv_content": str(ocr_result or "")}
            elif "cv_content" not in ocr_result:
                ocr_result = {"cv_content": str(ocr_result)}

            raw_cv_list.append(
                {
                    "file_name": file_name,
                    "file_path": file_path,
                    "structured_data": ocr_result,
                }
            )
        except Exception as exc:
            print(f"[-] Skipping '{file_name}' because OCR failed: {exc}")

    return raw_cv_list


def print_final_report(final_ranked_results: List[Dict]) -> None:
    if not final_ranked_results:
        print("[WARNING] No final ranked results to display.")
        return

    print("\n" + "=" * 70)
    print("FINAL CANDIDATE RANKING")
    print("=" * 70)
    print(f"{'Rank':<5} | {'CV file':<25} | {'LLM':<10} | {'Embedding':<10} | {'Decision':<18}")
    print("-" * 82)

    for idx, cv in enumerate(final_ranked_results, start=1):
        file_name = cv.get("file_name", "Unknown")
        display_name = file_name[:20] + "..." if len(file_name) > 23 else file_name
        llm_score = cv.get("final_score", 0)
        embedding_score = cv.get("semantic_score", cv.get("embedding_score", 0.0))
        decision = cv.get("decision", "Review")

        print(
            f"{idx:<5} | {display_name:<25} | {llm_score:<10} | "
            f"{float(embedding_score):<10.4f} | {decision:<18}"
        )

    top_cv = final_ranked_results[0]
    evaluation = top_cv.get("llm_evaluation", {})

    print("\n" + "=" * 70)
    print("TOP 1 DETAILED ANALYSIS")
    print("=" * 70)
    print(f"[*] File: {top_cv.get('file_name')}")
    print(f"[*] Final score: {top_cv.get('final_score')}/100")
    print(f"[*] Decision: {top_cv.get('decision')}")

    print("\n[+] Strengths:")
    for strength in evaluation.get("strengths", []):
        print(f"   - {strength}")

    print("\n[-] Weaknesses / rule risks:")
    for weakness in evaluation.get("weaknesses", []):
        print(f"   - {weakness}")

    print("\n[=>] LLM justification:")
    print(evaluation.get("detailed_justification", "No detailed justification."))
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CV screening pipeline: OCR -> embedding retrieval -> LLM rerank."
    )
    parser.add_argument("--cv-folder", default=os.getenv("CV_FOLDER", "CV"))
    parser.add_argument("--jd-path", default=os.getenv("JD_PATH", "jd.txt"))
    parser.add_argument("--rule-path", default=LLMConfig.RULE_TXT_PATH)
    parser.add_argument(
        "--ocr",
        choices=["tesseract", "qwen", "pix2struct"],
        default=os.getenv("OCR_ENGINE", "tesseract"),
        help="OCR engine to use.",
    )
    parser.add_argument(
        "--embedding",
        choices=["hybrid-bge", "bge", "nomic"],
        default=os.getenv("EMBEDDING_ENGINE", "hybrid-bge"),
        help="Embedding/ranking backend. hybrid-bge uses LLM/Rag.py weighted scoring.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=LLMConfig.TOP_K_RAG,
        help="Number of embedding candidates passed to LLM reranking.",
    )
    parser.add_argument(
        "--max-rules",
        type=optional_positive_int,
        default=LLMConfig.MAX_RULES,
        help="Maximum rules loaded from rule.txt for LLM reranking. 0 means all.",
    )
    parser.add_argument(
        "--must-have",
        default=os.getenv("MUST_HAVE_KEYWORDS", "Python,Postgres"),
        help="Comma-separated mandatory keywords. Empty string disables it.",
    )
    parser.add_argument(
        "--must-not-have",
        default=os.getenv("MUST_NOT_HAVE_KEYWORDS", "PHP,WordPress"),
        help="Comma-separated blacklist keywords. Empty string disables it.",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL_NAME"),
        help="Optional Hugging Face model id for Qwen OCR.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> PipelineSettings:
    return PipelineSettings(
        cv_folder=args.cv_folder,
        jd_path=args.jd_path,
        rule_path=args.rule_path,
        ocr_engine=args.ocr,
        embedding_engine=args.embedding,
        top_k=args.top_k,
        max_rules=args.max_rules,
        must_have_keywords=parse_keywords(args.must_have),
        must_not_have_keywords=parse_keywords(args.must_not_have),
        qwen_model=args.qwen_model,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = build_settings(args)

    print("=" * 70)
    print("AI-HR CV SCREENING PIPELINE")
    print("=" * 70)
    print(
        "[CONFIG] "
        f"OCR={settings.ocr_engine} | "
        f"Embedding={settings.embedding_engine} | "
        f"top_k={settings.top_k} | "
        f"max_rules={settings.max_rules if settings.max_rules else 'all'}"
    )

    jd_text = load_jd(settings.jd_path)
    cv_files = discover_cv_files(settings.cv_folder)
    if not cv_files:
        print(f"[WARNING] No supported CV files found in '{settings.cv_folder}/'.")
        return

    print(f"[STAGE 0] Found {len(cv_files)} CV file(s).")

    print("\n" + "=" * 50)
    print(f"[STAGE 1] OCR with engine: {settings.ocr_engine}")
    print("=" * 50)
    raw_cv_list = run_ocr_stage(cv_files, settings)
    if not raw_cv_list:
        print("[-] No CV data was extracted successfully. Stopping pipeline.")
        return

    print("\n" + "=" * 50)
    print(f"[STAGE 2] Embedding retrieval with engine: {settings.embedding_engine}")
    print("=" * 50)
    rag_top_candidates = retrieve_candidates(raw_cv_list, jd_text, settings)
    if not rag_top_candidates:
        print("[WARNING] No candidate passed hard filtering and embedding retrieval.")
        return

    print("\n" + "=" * 50)
    print("[STAGE 3] LLM reranking")
    print("=" * 50)
    from LLM.LLM_rerank import run_full_llm_reranking_pipeline

    final_ranked_results = run_full_llm_reranking_pipeline(
        rag_top_candidates,
        jd_text,
        rule_path=settings.rule_path,
        max_rules=settings.max_rules,
    )
    print_final_report(final_ranked_results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}")
        raise
