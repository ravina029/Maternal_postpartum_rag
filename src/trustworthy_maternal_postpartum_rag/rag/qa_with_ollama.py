# qa_with_ollama.py

import logging

from trustworthy_maternal_postpartum_rag.utils import call_ollama
from trustworthy_maternal_postpartum_rag.app.final_answer_generation import answer_question_final

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def ollama_llm(prompt: str) -> str:
    return call_ollama(prompt)

if __name__ == "__main__":
    print("Local maternal / postpartum / child-care RAG QA (Ollama)")
    print("Evidence-based • Lifecycle-aware • Topic-aware • Transparent\n")

    while True:
        q = input("Question> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        result = answer_question_final(q, llm_fn=ollama_llm)

        print(f"\n--- STATUS: {str(result.get('status', '')).upper()} ---")
        print(result.get("answer", ""))

        print("\n--- AUDIT ---")
        print(result.get("audit", {}))
        print()
