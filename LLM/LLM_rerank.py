import json
import os
import re

from LLM.config import LLMConfig


DEFAULT_RULES = (
    "1. Score from 0 to 100 based on technical skill match.\n"
    "2. Penalize candidates with unclear recent work history or very short tenure.\n"
    "3. Prefer candidates with relevant production experience, projects, or certifications."
)


def limit_rules(rules_text, max_rules=None):
    if not max_rules or max_rules <= 0:
        return rules_text

    rules = [rule.strip() for rule in re.split(r"\n\s*\n", rules_text) if rule.strip()]
    if len(rules) <= 1:
        rules = [line.strip() for line in rules_text.splitlines() if line.strip()]

    return "\n\n".join(rules[:max_rules])


def load_screening_rules(rule_path=LLMConfig.RULE_TXT_PATH, max_rules=None):
    """Load screening rules and optionally keep only the first max_rules rules."""
    if not os.path.exists(rule_path):
        print(f"[LLM WARNING] Rule file not found at '{rule_path}'. Using default rules.")
        return limit_rules(DEFAULT_RULES, max_rules=max_rules)

    with open(rule_path, "r", encoding="utf-8") as file:
        return limit_rules(file.read(), max_rules=max_rules)


def heuristic_split_cv(text):
    """Split a CV into coarse sections so the LLM sees a cleaner input."""
    sections = {"Skill": "", "Exp": "", "Edu": "", "Other": ""}
    current_section = "Other"

    skill_pattern = re.compile(
        r"(?i)\b(skills?|ky nang|k\u1ef9 n\u0103ng|cong nghe|"
        r"c\u00f4ng ngh\u1ec7|technologies|tech stack)\b"
    )
    exp_pattern = re.compile(
        r"(?i)\b(experience|kinh nghiem|kinh nghi\u1ec7m|work history|"
        r"l\u1ecbch s\u1eed l\u00e0m vi\u1ec7c|employment|du an|"
        r"d\u1ef1 \u00e1n|projects?)\b"
    )
    edu_pattern = re.compile(
        r"(?i)\b(education|hoc van|h\u1ecdc v\u1ea5n|trinh do|"
        r"tr\u00ecnh \u0111\u1ed9|bang cap|b\u1eb1ng c\u1ea5p|"
        r"certifications?|dai hoc|\u0111\u1ea1i h\u1ecdc|university)\b"
    )

    for line in text.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue

        if len(line_strip) < 60:
            if skill_pattern.search(line_strip):
                current_section = "Skill"
                continue
            if exp_pattern.search(line_strip):
                current_section = "Exp"
                continue
            if edu_pattern.search(line_strip):
                current_section = "Edu"
                continue

        sections[current_section] += line_strip + "\n"

    return sections


def _build_openai_client():
    if not LLMConfig.API_KEY or LLMConfig.API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY.")

    from openai import OpenAI

    return OpenAI(
        api_key=LLMConfig.API_KEY,
        base_url=LLMConfig.BASE_URL,
    )


def llm_rerank_candidate(
    candidate_cv,
    jd_text,
    rule_path=LLMConfig.RULE_TXT_PATH,
    max_rules=None,
):
    """Use an LLM to score one candidate after embedding retrieval."""
    rules = load_screening_rules(rule_path, max_rules=max_rules)
    cv_content = candidate_cv.get("structured_data", {}).get("cv_content", "")
    file_name = candidate_cv.get("file_name", "Unknown_CV")

    parsed_cv = heuristic_split_cv(cv_content)
    structured_cv_text = (
        f"[SKILLS]\n{parsed_cv['Skill']}\n\n"
        f"[EXPERIENCE]\n{parsed_cv['Exp']}\n\n"
        f"[EDUCATION]\n{parsed_cv['Edu']}\n\n"
        f"[OTHER]\n{parsed_cv['Other']}"
    )

    system_prompt = (
        "You are a senior technical recruiter and AI hiring analyst.\n"
        "Evaluate one candidate CV against the job description and the screening rules below.\n\n"
        f"=== SCREENING RULES ===\n{rules}\n\n"
        "Return pure JSON only, without markdown fences. Use exactly this schema:\n"
        "{\n"
        '  "final_score": <integer from 0 to 100>,\n'
        '  "decision": "Strong fit | Review | Reject",\n'
        '  "strengths": [<candidate strengths>],\n'
        '  "weaknesses": [<candidate gaps or rule risks>],\n'
        '  "detailed_justification": "<clear explanation of the score and decision>"\n'
        "}"
    )

    user_prompt = (
        f"--- CV FILE: {file_name} ---\n\n"
        f"=== JOB DESCRIPTION ===\n{jd_text}\n\n"
        f"=== PARSED CANDIDATE CV ===\n{structured_cv_text}\n"
    )

    try:
        client = _build_openai_client()
        response = client.chat.completions.create(
            model=LLMConfig.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        result_json = json.loads(response.choices[0].message.content)

        candidate_cv["llm_evaluation"] = result_json
        candidate_cv["final_score"] = result_json.get("final_score", 0)
        candidate_cv["decision"] = result_json.get("decision", "Review")
        candidate_cv["parsed_sections"] = parsed_cv

        return candidate_cv

    except Exception as exc:
        print(f"[-] LLM rerank failed for {file_name}: {exc}")
        candidate_cv["llm_evaluation"] = {
            "final_score": int(candidate_cv.get("semantic_score", 0) * 100),
            "decision": "System error",
            "strengths": [],
            "weaknesses": ["LLM reranking failed."],
            "detailed_justification": f"LLM reranking failed: {str(exc)}",
        }
        candidate_cv["final_score"] = int(candidate_cv.get("semantic_score", 0) * 100)
        candidate_cv["decision"] = "System error"
        return candidate_cv


def run_full_llm_reranking_pipeline(
    rag_top_candidates,
    jd_text,
    rule_path=LLMConfig.RULE_TXT_PATH,
    max_rules=None,
):
    final_ranked_results = []
    print(
        "[LLM INFO] Running deep rerank for "
        f"{len(rag_top_candidates)} candidate(s) with "
        f"max_rules={max_rules if max_rules else 'all'}..."
    )

    for cv in rag_top_candidates:
        evaluated_cv = llm_rerank_candidate(
            cv,
            jd_text,
            rule_path=rule_path,
            max_rules=max_rules,
        )
        final_ranked_results.append(evaluated_cv)

    return sorted(final_ranked_results, key=lambda item: item["final_score"], reverse=True)
