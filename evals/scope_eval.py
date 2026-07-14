import os
import sys
import csv
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

import agent.graph  # נדרש לפני agent.nodes - ראו הערת circular import ב-cost_eval.py
from agent.nodes import guardrail_node

# שאלות מחוץ לתחום (חייב 5 לפחות לפי הבריף) + כמה בתוך התחום, לביקורת (sanity check)
OUT_OF_SCOPE_QUESTIONS = [
    "תביא לי כוס מים",
    "כתבי לי שיר על האביב",
    "מה מזג האוויר מחר בתל אביב?",
    "תני לי מתכון לעוגת שוקולד",
    "עזרי לי לכתוב קוד פייתון שממיין רשימה",
    "מה הבירה של צרפת?",
    "תני לי עצה זוגיות",
]

IN_SCOPE_QUESTIONS = [
    "פרצוף שמח עם עיניים נוצצות",
    "אימוג'י של רובוט עם עיניים כחולות",
    "ידיים מוחאות כפיים בהתלהבות",
]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "scope_eval.csv")


def check_question(question: str, expected_in_scope: bool) -> dict:
    """מריצה Guardrail בלבד על שאלה, ובודקת אם ההחלטה תואמת את הציפייה."""
    try:
        result = guardrail_node({"user_input": question})
        actual_in_scope = result.get("is_in_scope", True)
        correct = actual_in_scope == expected_in_scope
        return {
            "question": question,
            "expected_in_scope": expected_in_scope,
            "actual_in_scope": actual_in_scope,
            "correct": correct,
            "reason": result.get("guardrail_reason", ""),
        }
    except Exception as e:
        return {
            "question": question,
            "expected_in_scope": expected_in_scope,
            "actual_in_scope": None,
            "correct": False,
            "reason": f"שגיאה: {str(e)[:150]}",
        }


def main():
    all_rows = []

    print("=== שאלות מחוץ לתחום (צריך שכולן ייחסמו) ===")
    for q in OUT_OF_SCOPE_QUESTIONS:
        row = check_question(q, expected_in_scope=False)
        all_rows.append(row)
        status = "✓ נחסם נכון" if row["correct"] else "✗ לא נחסם! (בעיה)"
        print(f"  {status} | \"{q}\"")

    print("\n=== שאלות בתוך התחום (sanity check - צריך שכולן יעברו) ===")
    for q in IN_SCOPE_QUESTIONS:
        row = check_question(q, expected_in_scope=True)
        all_rows.append(row)
        status = "✓ עבר נכון" if row["correct"] else "✗ נחסם בטעות! (בעיה)"
        print(f"  {status} | \"{q}\"")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected_in_scope", "actual_in_scope", "correct", "reason"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nתוצאות נשמרו ל-{RESULTS_PATH}")

    out_of_scope_rows = [r for r in all_rows if not r["expected_in_scope"]]
    refusal_rate = sum(1 for r in out_of_scope_rows if r["correct"]) / len(out_of_scope_rows) * 100

    in_scope_rows = [r for r in all_rows if r["expected_in_scope"]]
    false_positive_rate = sum(1 for r in in_scope_rows if not r["correct"]) / len(in_scope_rows) * 100

    print(f"\n--- סיכום ---")
    print(f"Refusal rate (שאלות מחוץ לתחום שנחסמו נכון): {refusal_rate:.0f}%")
    print(f"False positive rate (שאלות תקינות שנחסמו בטעות): {false_positive_rate:.0f}%")


if __name__ == "__main__":
    main()