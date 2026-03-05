# scripts/init_project_structure.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PACKAGE_PATH = ROOT / "src" / "trustworthy_maternal_postpartum_rag"

FOLDERS = [
    # Main package
    PACKAGE_PATH,
    PACKAGE_PATH / "ingestion",
    PACKAGE_PATH / "preprocessing",
    PACKAGE_PATH / "retrieval",
    PACKAGE_PATH / "pipeline",
    PACKAGE_PATH / "rag_app",
    # Existing module folders kept
    ROOT / "src" / "retriever",
    ROOT / "src" / "generator",
    ROOT / "src" / "safety",
    ROOT / "src" / "explainability",
    ROOT / "src" / "evaluation",
    ROOT / "src" / "ui",
    # Data folders
    ROOT / "data" / "raw" / "pdfs",
    ROOT / "data" / "processed",
    # Logs folders
    ROOT / "logs" / "data_logs",
    ROOT / "logs" / "pipeline_logs",
    # Other project folders
    ROOT / "notebooks",
    ROOT / "models" / "local_llm",
    ROOT / "models" / "sentence_transformer",
    ROOT / "scripts",
]

FILES = {
    PACKAGE_PATH / "__init__.py": "",
    PACKAGE_PATH / "utils.py": "# utility functions will go here\n",
    (PACKAGE_PATH / "ingestion" / "__init__.py"): "",
    (PACKAGE_PATH / "preprocessing" / "__init__.py"): "",
    (PACKAGE_PATH / "retrieval" / "__init__.py"): "",
    (PACKAGE_PATH / "pipeline" / "__init__.py"): "",
    (PACKAGE_PATH / "rag_app" / "__init__.py"): "",
}

def main():
    print("\nUpdating project structure (manual script)...\n")

    for folder in FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder ready: {folder}")

    for path, content in FILES.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(f"File created: {path}")
        else:
            print(f"File exists, skipped: {path}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
