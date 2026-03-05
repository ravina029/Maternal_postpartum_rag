# src/trustworthy_maternal_postpartum_rag/pipeline/intent_classifier.py

from trustworthy_maternal_postpartum_rag.utils import call_ollama

INTENT_PROMPT = """
Classify the user query into exactly one of:
A) Emergency / urgent symptoms / time-critical warning signs
B) Medical decision / diagnosis / treatment (non-emergency)
C) General informational guidance / education

Rules:
- Return ONLY the single letter: A, B, or C.
- If the query describes severe symptoms, rapid deterioration, heavy bleeding, chest pain, trouble breathing, seizures, fainting, or suicidal thoughts → A.
- If the query asks what to do medically (medications, dose, diagnosis, treatment choice) but not clearly urgent → B.
- Otherwise → C.
"""

def classify_intent(query: str, llm_call=call_ollama) -> str:
    try:
        out = llm_call(INTENT_PROMPT + f"\nQUERY:\n{query}\n").strip().upper()

        # Defensive normalization (handles "A)", "A.", "Answer: A", etc.)
        if out:
            out = out.replace("ANSWER:", "").strip()
            out = out[0]  # first character is usually A/B/C

        if out in {"A", "B", "C"}:
            return out
    except Exception:
        pass

    # Fail-open to informational guidance
    return "C"
