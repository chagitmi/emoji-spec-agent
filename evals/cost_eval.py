import os
import sys
import csv
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agent.tools import estimate_cost

TEST_INPUT = "פרצוף שמח עם עיניים נוצצות ועיגול ורוד על הלחיים"

MODELS_TO_TEST = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# מחיר "תיאורטי" למודל חינמי - כאילו היה בתשלום, לפי מודל בגודל דומה (70B) של ספק אחר.
# OpenRouter לא גובה כסף על מודלים חינמיים, אז אין להם מחיר אמיתי - זו הערכה לצורך השוואה.
THEORETICAL_FREE_PRICING = {"input": 0.10, "output": 0.30}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "cost_eval.csv")


def run_node_with_cost(node_func, state: dict, model_name: str, node_display_name: str, max_retries: int = 2) -> tuple:
    """מריצה נוד בודד, עוטפת אותו ב-callback נפרד, ומחזירה (state_updates, cost_row).
    מנסה שוב אם השגיאה היא rate-limit זמני (429)."""
    import time as _time
    for attempt in range(max_retries + 1):
        try:
            with get_usage_metadata_callback() as usage_cb:
                updates = node_func(state)
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                print(f"      (rate limit, ממתינה 15 שניות... ניסיון {attempt + 2}/{max_retries + 1})")
                _time.sleep(15)
                continue
            raise
    input_tokens = output_tokens = 0
    for model, usage in usage_cb.usage_metadata.items():
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    is_free = model_name.endswith(":free")
    if is_free:
        pricing = THEORETICAL_FREE_PRICING
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
        cost_type = "תיאורטי (מודל חינמי)"
    else:
        cost = estimate_cost(model_name, input_tokens, output_tokens)
        cost_type = "אמיתי"

    row = {
        "model": model_name,
        "node": node_display_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "cost_type": cost_type,
    }
    return updates, row


def run_full_pipeline_cost(model_name: str) -> list:
    """מריצה את כל שלבי הגרף ידנית (לא דרך app.stream), נוד-נוד, כדי לפרק עלות מדויקת לכל שלב."""
    os.environ["MODEL_UNDERSTANDING"] = model_name
    os.environ["MODEL_DRAFTING"] = model_name

  # חשוב: קודם agent.graph (מגדיר AgentState), ורק אז agent.nodes - למניעת circular import
    import agent.graph  # noqa: F401 - נדרש כדי ש-agent.nodes ייטען נכון
    from agent import nodes as N
    rows = []
    state = {"user_input": TEST_INPUT, "retry_count": 0}

    pipeline = [
        (N.guardrail_node, "Guardrail"),
        (N.emotion_node, "הבנת רגש"),
        (N.gesture_node, "זיהוי מחווה"),
        (N.features_node, "פירוק למאפיינים"),
        (N.similarity_check_node, "בדיקת דמיון קיים"),
        (N.rag_and_spec_node, "RAG + מפרט"),
        (N.explanation_node, "הסבר החלטות"),
        (N.evaluator_node, "Evaluator"),
    ]

    for node_func, display_name in pipeline:
        # מכבדים את ה-Gates: אם חסום או נמצא דומה, לא ממשיכים להריץ את שאר הגרף
        if display_name == "הבנת רגש" and not state.get("is_in_scope", True):
            break
        if display_name == "RAG + מפרט" and state.get("similar_emoji_found"):
            break

        updates, row = run_node_with_cost(node_func, state, model_name, display_name)
        state.update(updates)
        rows.append(row)
        print(f"    {display_name}: {row['input_tokens']}+{row['output_tokens']} טוקנים | ${row['cost_usd']:.6f} ({row['cost_type']})")

    return rows


def main():
    all_rows = []

    for model in MODELS_TO_TEST:
        print(f"\n=== מודל: {model} ===")
        try:
            rows = run_full_pipeline_cost(model)
            all_rows.extend(rows)
            total_cost = sum(r["cost_usd"] for r in rows)
            print(f"  --> סה\"כ לריצה מלאה: ${total_cost:.6f}")
        except Exception as e:
            print(f"  ✗ שגיאה: {str(e)[:200]}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "node", "input_tokens", "output_tokens", "cost_usd", "cost_type"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nתוצאות נשמרו ל-{RESULTS_PATH}")

    print("\n--- סיכום עלות כוללת לפי מודל ---")
    for model in MODELS_TO_TEST:
        model_rows = [r for r in all_rows if r["model"] == model]
        if not model_rows:
            print(f"{model}: נכשל, אין נתונים")
            continue
        total = sum(r["cost_usd"] for r in model_rows)
        cost_type = model_rows[0]["cost_type"]
        print(f"{model}: ${total:.6f} ({cost_type})")


if __name__ == "__main__":
    main()