from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent
readme_path = ROOT / "Readme.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="trustworthy_maternal_postpartum_rag",
    version="0.1.0",
    description="A locally-runnable Trustworthy & Explainable RAG system designed for maternal postpartum knowledge retrieval.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ravina Verma",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["trustworthy_maternal_postpartum_rag*"]),
    include_package_data=True,
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "langchain",
        "faiss-cpu",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "nltk",
        "pyyaml",
        "streamlit",
        "shap",
        "lime",
    ],
)
