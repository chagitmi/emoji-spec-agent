from typing import TypedDict, Optional, Literal


class AgentState(TypedDict, total=False):
    # --- קלט מהמשתמש ---
    user_input: str
    target_platforms: list[str]

    # --- נוד 0: Guardrail ---
    is_in_scope: bool
    guardrail_reason: Optional[str]

    # --- נודים 1-3: הבנה ---
    emotion: Optional[str]
    gesture: Optional[str]
    features: Optional[dict]

    # --- נוד 4: בדיקת דמיון קיים ---
    similar_emoji_found: Optional[bool]
    similar_emoji_details: Optional[dict]

    # --- נוד 5: Gate ---
    needs_new_emoji: Optional[bool]

    # --- נוד 6-7: RAG + מפרט + הסבר ---
    style_guide_context: Optional[list[str]]
    spec: Optional[dict]
    explanation: Optional[str]

    # --- Evaluator ---
    eval_score: Optional[float]
    eval_feedback: Optional[str]
    retry_count: int

    # --- מעקב עלות/שימוש ---
    total_tokens_used: int
    total_cost_usd: float

    # --- פלט סופי ---
    final_output: Optional[str]


from langgraph.graph import StateGraph, END
from agent.nodes import (
    guardrail_node,
    emotion_node,
    gesture_node,
    features_node,
    similarity_check_node,
    needs_new_emoji_gate,
    rag_and_spec_node,
    explanation_node,
    evaluator_node,
    evaluator_gate,
    output_node,
)


def guardrail_gate(state: AgentState) -> str:
    """Gate - האם הבקשה בתחום? אם לא, מדלגים ישר ל-Output עם הודעת סירוב."""
    if state.get("is_in_scope"):
        return "continue"
    return "blocked"


def build_graph():
    graph = StateGraph(AgentState)

    # רישום כל הנודים
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("emotion", emotion_node)
    graph.add_node("gesture", gesture_node)
    graph.add_node("features", features_node)
    graph.add_node("similarity_check", similarity_check_node)
    graph.add_node("rag_and_spec", rag_and_spec_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("output", output_node)

    # נקודת הכניסה
    graph.set_entry_point("guardrail")

    # Gate מותנה אחרי ה-Guardrail
    graph.add_conditional_edges(
        "guardrail",
        guardrail_gate,
        {
            "continue": "emotion",
            "blocked": "output",
        },
    )

    # זרימה ליניארית
    graph.add_edge("emotion", "gesture")
    graph.add_edge("gesture", "features")
    graph.add_edge("features", "similarity_check")

    # Gate מותנה אחרי נוד 4
    graph.add_conditional_edges(
        "similarity_check",
        needs_new_emoji_gate,
        {
            "continue_to_rag": "rag_and_spec",
            "skip_to_output": "output",
        },
    )

    # המשך הזרימה
    graph.add_edge("rag_and_spec", "explanation")
    graph.add_edge("explanation", "evaluator")

    # Gate מותנה של ה-Evaluator - לולאת חזרה
    graph.add_conditional_edges(
        "evaluator",
        evaluator_gate,
        {
            "retry": "rag_and_spec",
            "output": "output",
        },
    )

    graph.add_edge("output", END)

    return graph.compile()