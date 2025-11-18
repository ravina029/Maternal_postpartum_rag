import os
from pathlib import Path

# ----------------------------------------
# Root project directory
# ----------------------------------------
ROOT = Path(__file__).resolve().parent

# ----------------------------------------
# Folder structure (matches README + RAG architecture)
# ----------------------------------------
FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",

    "src/retriever",
    "src/generator",
    "src/safety",
    "src/explainability",
    "src/evaluation",
    "src/ui",

    "models/local_llm",
    "models/sentence_transformer",

    "scripts",
]

# ----------------------------------------
# Initial files (only created if missing)
# ----------------------------------------
FILES = {
    "data/sample_corpus.txt": "",
    "src/retriever/embedder.py": "",
    "src/retriever/retriever.py": "",
    "src/retriever/indexing.py": "",

    "src/generator/rag_model.py": "",

    "src/safety/hallucination_check.py": "",
    "src/safety/safety_classifier.py": "",
    "src/safety/rule_based_filters.py": "",

    "src/explainability/lime_explainer.py": "",
    "src/explainability/shap_explainer.py": "",

    "src/evaluation/faithfulness_eval.py": "",
    "src/evaluation/safety_eval.py": "",

    "src/ui/streamlit_app.py": "",

    "scripts/prepare_corpus.py": "",
    "scripts/build_index.py": "",
    "scripts/run_inference.py": "",
}

# ----------------------------------------
# Safety check: never overwrite existing files
# ----------------------------------------
def create_folders_and_files():
    print("\n🚀 Setting up project structure...\n")

    # Create folders
    for folder in FOLDERS:
        folder_path = ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Folder ready: {folder_path}")

    # Create files only if they don't exist
    for file_path, content in FILES.items():
        path = ROOT / file_path
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 File created: {path}")
        else:
            print(f"⚠️ File already exists, skipped: {path}")

    print("\n✨ Project structure is fully ready.\n")

# ----------------------------------------
if __name__ == "__main__":
    create_folders_and_files()
