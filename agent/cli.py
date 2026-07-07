import os
import sys
from langchain_core.callbacks import get_usage_metadata_callback
from agent.graph import build_graph
from agent.tools import estimate_cost


HELP_TEXT = """
פקודות זמינות:
  help / עזרה   - מציג הודעה זו
  exit / quit   - יציאה מהתוכנית

איך להשתמש:
  אפשרות 1: הקלידי תיאור חופשי של אימוג'י, למשל:
    "פרצוף שמח עם עיניים נוצצות"

  אפשרות 2: הקלידי נתיב לקובץ טקסט (.txt) שמכיל את התיאור, למשל:
    C:\\Users\\me\\Desktop\\my_emoji_idea.txt

  האייג'נט יזהה את הרגש והמחווה, יבדוק אם כבר קיים אימוג'י דומה,
  ואם לא - יפיק מפרט עיצובי מותאם לכל פלטפורמת יעד (Apple/Google/Samsung).
"""

# שמות "ידידותיים" לכל נוד, להצגה בזמן ה-streaming
NODE_DISPLAY_NAMES = {
    "guardrail": "בדיקת תחום (Guardrail)",
    "emotion": "הבנת רגש",
    "gesture": "זיהוי מחווה",
    "features": "פירוק למאפיינים",
    "similarity_check": "בדיקת דמיון קיים",
    "rag_and_spec": "RAG + התאמה לפלטפורמה",
    "explanation": "הסבר החלטות",
    "evaluator": "Evaluator",
    "output": "Output",
}


def resolve_input(raw_input: str) -> str:
    """
    אם הקלט הוא נתיב לקובץ טקסט קיים - קוראת ממנו את התוכן.
    אחרת, מחזירה את הקלט כמו שהוא (תיאור טקסטואלי ישיר).
    """
    possible_path = raw_input.strip('"')  # לפעמים משתמשים מדביקים נתיב עם מרכאות
    if os.path.isfile(possible_path):
        try:
            with open(possible_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"[מידע] נקרא תוכן מהקובץ: {possible_path}")
            if not content:
                raise ValueError("הקובץ ריק")
            return content
        except UnicodeDecodeError:
            raise ValueError(f"לא ניתן לקרוא את הקובץ {possible_path} - ודאי שהוא שמור כ-UTF-8")
    return raw_input


def print_spec(spec: dict):
    """מציגה את המפרט בצורה קריאה בטרמינל."""
    if not isinstance(spec, dict):
        print(spec)
        return

    for platform, details in spec.items():
        print(f"\n=== {platform} ===")
        if not isinstance(details, dict):
            print(details)
            continue
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")


def print_usage_and_cost(usage_metadata: dict):
    """מציגה טבלת שימוש ועלות משוערת, לפי המודלים שהיו מעורבים בריצה."""
    if not usage_metadata:
        return

    print("\n--- שימוש ועלות (משוער) ---")
    total_cost = 0.0
    total_tokens = 0

    for model, usage in usage_metadata.items():
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = estimate_cost(model, input_tokens, output_tokens)
        total_cost += cost
        total_tokens += input_tokens + output_tokens
        print(f"  {model}: {input_tokens} קלט + {output_tokens} פלט טוקנים | ~${cost:.5f}")

    print(f"  סה\"כ: {total_tokens} טוקנים | ~${total_cost:.5f}")


def run_graph_with_streaming(app, initial_state: dict) -> dict:
    """
    מריצה את הגרף תוך הצגת כל שלב שמתבצע (streaming), ומחזירה את ה-state הסופי.
    """
    final_state = dict(initial_state)

    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            if node_output:
                final_state.update(node_output)
    return final_state


def run_session():
    print("=" * 60)
    print("אייג'נט מפרטי אימוג'י — הקלידי תיאור, או 'help'/'exit'")
    print("=" * 60)

    app = build_graph()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nלהתראות!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "יציאה"):
            print("להתראות!")
            break
        if user_input.lower() in ("help", "עזרה"):
            print(HELP_TEXT)
            continue

        try:
            resolved_input = resolve_input(user_input)
        except ValueError as e:
            print(f"[שגיאה] {e}")
            continue

        try:
            with get_usage_metadata_callback() as usage_cb:
                result = run_graph_with_streaming(app, {"user_input": resolved_input, "retry_count": 0})
        except Exception as e:
            print(f"\n[שגיאה] משהו השתבש בזמן התקשורת עם המודל: {e}")
            print("נסי שוב, או בדקי את החיבור לאינטרנט ואת מפתח ה-API ב-.env")
            continue

        print("\n" + "-" * 60)
        if not result.get("is_in_scope", True):
            print(f"בקשה נחסמה: {result.get('guardrail_reason', '')}")
        elif result.get("similar_emoji_found"):
            print(result.get("final_output"))
        else:
            print("הסבר:")
            print(result.get("explanation", ""))
            print("\nמפרט:")
            print_spec(result.get("spec", {}))

        print_usage_and_cost(usage_cb.usage_metadata)
        print("-" * 60)


if __name__ == "__main__":
    run_session()