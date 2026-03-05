High‑level timeline (2 weeks)
Assume today is night of 2 Jan and you have family + travel load, so plan ~3 focused hours/day, plus 1–2 longer blocks on weekends.
​

By 4 Jan – Stabilize ingestion and retrieval
Goal: all ingestion scripts run without errors and retrieval returns sensible chunks.
​

Re‑run preprocessing → chunking → Chroma indexing once, cleanly, on your 7–8 PDFs.
​

Fix any remaining errors like the infer_lifecycle(section_title=...) mismatch so pipeline runs end‑to‑end.
​

Use your interactive retriever script to test 10–15 typical queries (pregnancy food, postpartum bleeding, baby fever, etc.).
​

If retrieval looks “good enough” (top 1–3 chunks are on topic), stop tweaking weights.
​

5–7 Jan – Solidify generation (QA script)
Goal: one robust qa.py (or qa_with_ollama.py) that:
​

Takes a question, calls retriever, builds prompt with k chunks, calls Ollama, prints answer + sources JSON.
​

Handles basic failures: no chunks, model timeout, etc., without crashing.
​

Uses a single prompt pattern for safety (e.g., “only answer from context; if missing, say you don’t know”).
​

Test this with 10–15 questions across pregnancy, postpartum, newborn, and general topics and keep the best examples.
​

8–10 Jan – Trustworthiness mini‑evaluation
Goal: a small but serious evaluation you can show in your PhD applications.
​

Build a 20–30‑question evaluation set (CSV/JSON) covering: danger signs, diet, medications, common baby issues.
​

For each question, run your QA script, log: query, answer, retrieved chunks, and sources.
​

Manually score each answer on 3 axes (0–2 scale): relevance, factual faithfulness to sources, and safety.
​

At the end, compute simple percentages: what fraction of answers are fully correct and safe.
​

11–13 Jan – Documentation and polish
Goal: turn this into something “submission‑ready”.
​

Write a concise README with: motivation (pregnancy/postpartum assistant), data description, architecture, how to run, limitations.
​

Add one diagram (even ASCII/Markdown) showing: user → retriever → LLM with context → answer + citations.
​

Collect 4–5 strong example interactions and 2–3 borderline ones to illustrate failure modes and how you handle them.
​

If you have time, wrap the QA endpoint in a tiny FastAPI or Streamlit UI; if not, keep it CLI‑only.
​

14–15 Jan – Buffer and freeze
Goal: stop coding and only stabilize.
​

Full dry‑run: clone repo to a fresh folder/venv on your Mac, install deps, run end‑to‑end following your own README.
​

Fix any install/run issues; do not add new features.
​

Back up: push to GitHub, keep a local zipped copy, and (if possible) a private cloud copy so travel won’t risk losing work.
​

Daily execution pattern (with baby + travel prep)
Use strict time‑boxing so you get progress even on tired days.
​

Block 2 × 60–75‑minute deep‑work slots per day (e.g., baby’s nap + late evening).
​

For each slot, have exactly one target: “fix chunking error”, “finish QA script”, “score 5 evaluation questions”.
​

Keep a simple TODO for the next day at the top of your repo (e.g., NEXT_TASKS.md) so you can resume quickly after interruptions.
​
