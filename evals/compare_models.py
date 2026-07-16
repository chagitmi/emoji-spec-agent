import os
import sys
import time
import csv
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")  # רינדור בלי חלון גרפי - נחוץ להרצה מטרמינל
import matplotlib.pyplot as plt

# מוסיפה את שורש הפרויקט ל-sys.path, כדי ש-"from agent..." יעבוד גם כשמריצים מתוך evals/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# סט תיאורים קבוע - אותו סט לכל המודלים, לצורך השוואה הוגנת
TEST_CASES = [
    "פרצוף שמח עם עיניים נוצצות",
    "אימוג'י של רובוט עם עיניים כחולות מרובעות",
    "פרצוף עייף שמפהק בבוקר",
    "ידיים מוחאות כפיים בהתלהבות",
    "פרצוף עם דמעה אחת ועיניים נוצצות משמחה",
]

MODELS_TO_COMPARE = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "meta-llama/llama-3.2-3b-instruct:free",
]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "compare_models.csv")


def run_single_test(model_name: str, test_input: str, max_retries: int = 2) -> dict:
    """מריצה את כל הגרף פעם אחת, עם מודל נתון, על תיאור נתון. מחזירה תוצאות מדידה.
    מנסה שוב (עד max_retries פעמים) אם השגיאה היא rate-limit זמני (429)."""
    os.environ["MODEL_UNDERSTANDING"] = model_name
    os.environ["MODEL_DRAFTING"] = model_name

    from agent.graph import build_graph
    app = build_graph()

    start = time.time()
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            result = app.invoke({"user_input": test_input, "retry_count": 0})
            elapsed = time.time() - start
            return {
                "model": model_name,
                "test_input": test_input,
                "success": True,
                "error": "",
                "time_seconds": round(elapsed, 2),
                "eval_score": result.get("eval_score"),
            }
        except Exception as e:
            last_error = str(e)[:200]
            if "429" in last_error and attempt < max_retries:
                print(f"     (rate limit, ממתינה 15 שניות ומנסה שוב... ניסיון {attempt + 2}/{max_retries + 1})")
                time.sleep(15)
                continue
            break

    elapsed = time.time() - start
    return {
        "model": model_name,
        "test_input": test_input,
        "success": False,
        "error": last_error,
        "time_seconds": round(elapsed, 2),
        "eval_score": None,
    }

def main():
    all_results = []

    for model in MODELS_TO_COMPARE:
        print(f"\n=== בודקת מודל: {model} ===")
        for test_input in TEST_CASES:
            print(f"  -> \"{test_input}\"")
            row = run_single_test(model, test_input)
            all_results.append(row)
            status = "✓" if row["success"] else "✗ " + row["error"]
            print(f"     {status} | זמן: {row['time_seconds']}s | ציון: {row['eval_score']}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_input", "success", "error", "time_seconds", "eval_score"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nתוצאות נשמרו ל-{RESULTS_PATH}")

    print("\n--- סיכום לפי מודל ---")
    for model in MODELS_TO_COMPARE:
        model_rows = [r for r in all_results if r["model"] == model]
        successes = [r for r in model_rows if r["success"]]
        reliability = len(successes) / len(model_rows) * 100
        avg_time = sum(r["time_seconds"] for r in model_rows) / len(model_rows)
        scores = [r["eval_score"] for r in successes if r["eval_score"] is not None]
        avg_score = (sum(scores) / len(scores)) if scores else 0
        print(f"{model}: אמינות={reliability:.0f}% | זמן ממוצע={avg_time:.2f}s | ציון איכות ממוצע={avg_score:.2f}")
        # --- גרף השוואה ---
    labels = [m.split("/")[-1] for m in MODELS_TO_COMPARE]
    reliabilities, avg_times, avg_scores = [], [], []
    for model in MODELS_TO_COMPARE:
        model_rows = [r for r in all_results if r["model"] == model]
        successes = [r for r in model_rows if r["success"]]
        reliabilities.append(len(successes) / len(model_rows) * 100)
        avg_times.append(sum(r["time_seconds"] for r in model_rows) / len(model_rows))
        scores = [r["eval_score"] for r in successes if r["eval_score"] is not None]
        avg_scores.append((sum(scores) / len(scores) * 100) if scores else 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(labels, reliabilities, color="#4a90d9")
    axes[0].set_title("אמינות (%)")
    axes[0].set_ylim(0, 105)
    axes[1].bar(labels, avg_times, color="#e08a3c")
    axes[1].set_title("זמן ממוצע (שניות)")
    axes[2].bar(labels, avg_scores, color="#5cb85c")
    axes[2].set_title("ציון איכות ממוצע (%)")
    axes[2].set_ylim(0, 105)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "results", "compare_models_chart.png")
    plt.savefig(chart_path, dpi=120)
    print(f"גרף נשמר ל-{chart_path}")


if __name__ == "__main__":
    main()