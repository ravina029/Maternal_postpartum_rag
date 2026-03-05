import subprocess

def call_ollama(prompt: str) -> str:
    """
    Calls Ollama locally and returns raw text output.
    Assumes Ollama + llama3 are installed.
    """
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout
