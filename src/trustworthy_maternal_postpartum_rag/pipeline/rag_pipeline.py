# src/trustworthy_maternal_postpartum_rag/pipeline/rag_pipeline.py

from typing import Optional, Dict, Any

from trustworthy_maternal_postpartum_rag.pipeline.intent_classifier import classify_intent
from trustworthy_maternal_postpartum_rag.safety.emergency import detect_emergency
from trustworthy_maternal_postpartum_rag.pipeline.logger import log_reasoning


def rag_pipeline(
    query,
    docs,
    generator_fn,
    llm_call,
    audit_ctx: Optional[Dict[str, Any]] = None
):
    """
    Orchestrates safety → intent → generation.

    audit_ctx is optional and should be populated by the caller (typically local_qa),
    because it has access to retrieval stats and selected chunks.
    """
    audit_ctx = audit_ctx or {}

    # ----------------------------
    # Emergency hard stop (rule-based)
    # ----------------------------
    if detect_emergency(query):
        decision = {"intent": "A", "action": "emergency"}
        status = "emergency"

        log_reasoning(
            query,
            decision,
            status=status,
            lifecycle=audit_ctx.get("lifecycle"),
            topic=audit_ctx.get("topic"),
            retrieved_chunks=audit_ctx.get("retrieved_chunks"),
            used_chunks=audit_ctx.get("used_chunks"),
            publisher_counts=audit_ctx.get("publisher_counts"),
            run_id=audit_ctx.get("run_id"),
        )

        return {
            "query": query,
            "decision": decision,
            "answer": (
                "This may be a medical emergency.\n"
                "Please seek immediate medical care or contact emergency services."
            ),
            "status": status,
        }

    # ----------------------------
    # Intent routing (LLM)
    # ----------------------------
    intent = classify_intent(query, llm_call)

    if intent == "B":
        decision = {"intent": "B", "action": "medical_guidance"}
    else:
        decision = {"intent": intent, "action": "answer"}

    # ----------------------------
    # Generation
    # ----------------------------
    error = None
    try:
        answer = generator_fn(query, docs, decision)
        status = "answered"
    except Exception as e:
        error = str(e)
        decision["error"] = error
        answer = (
            "I could not generate a response due to an internal error. "
            "Please try again, or rephrase your question."
        )
        status = "error"

    # ----------------------------
    # Audit log (always)
    # ----------------------------
    log_reasoning(
        query,
        decision,
        status=status,
        lifecycle=audit_ctx.get("lifecycle"),
        topic=audit_ctx.get("topic"),
        retrieved_chunks=audit_ctx.get("retrieved_chunks"),
        used_chunks=audit_ctx.get("used_chunks"),
        publisher_counts=audit_ctx.get("publisher_counts"),
        error=error,
        run_id=audit_ctx.get("run_id"),
    )

    return {
        "query": query,
        "decision": decision,
        "answer": answer,
        "status": status,
    }
