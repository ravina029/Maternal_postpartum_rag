EMERGENCY_KEYWORDS = [
    "unconscious", "loss of consciousness", "not responding",
    "seizure", "convulsions",
    "difficulty breathing", "shortness of breath",
    "blue lips", "chest pain",
    "severe bleeding", "heavy bleeding",
    "vomiting blood", "blood in stool",
    "no fetal movement",
    "newborn not breathing",
    "baby seizure",
    "child seizure",
    "choking",
    "poisoning",
]

def detect_emergency(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in EMERGENCY_KEYWORDS)
