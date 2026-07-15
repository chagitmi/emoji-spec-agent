# SKILLS.md — כללי עבודה על הפרויקט

מסמך זה מתעד מוסכמות קבועות בקוד, כדי שהוספות עתידיות (על ידי AI או מפתח אנושי) יישארו עקביות עם מה שכבר קיים. יש לקרוא לפני כתיבת קוד חדש בפרויקט.

## מבנה נוד חדש בגרף

כל נוד חדש בגרף חייב:
1. לקבל `state: AgentState` ולהחזיר `dict` עם **רק** השדות שהוא מעדכן (לא את כל ה-state).
2. אם הנוד קורא ל-LLM לצורך פלט מובנה (לא טקסט חופשי) - להגדיר `class XResult(BaseModel)` ייעודי מעל הפונקציה, עם `Field(description=...)` בעברית לכל שדה, ולהשתמש ב-`llm.with_structured_output(XResult)`.
3. אם הנוד קורא ל-LLM לטקסט חופשי (כמו הסברים) - `llm.invoke(prompt)` רגיל, `result.content`.
4. תמיד `print()` בתחילת הנוד (`print("[נוד] שם הנוד")`) ו-print נוסף עם תוצאת ההחלטה המרכזית - זו "שקיפות" קריטית לדיבוג ול-streaming.
5. שימוש ב-`get_llm(os.getenv("MODEL_X"), temperature=...)` - **אף פעם** לא ליצור client חדש עצמאי.
6. `temperature=0.0` לכל נוד ששופט/מסווג (Guardrail, Evaluator, שיפוט דמיון) - `temperature=0.3` (ברירת מחדל) לנודים יצירתיים (ניסוח, הסבר).

## Gates (קצוות מותנים)

פונקציית Gate מקבלת `state`, **לא** משנה אותו, ומחזירה **מחרוזת** (שם היעד הבא) - לא dict. נרשמת ב-`graph.py` דרך `add_conditional_edges`, לא `add_node`.

## RAG collections (ChromaDB)

- כל collection חדש: `metadata={"hnsw:space": "cosine"}` - **אף פעם** לא ברירת המחדל (L2), כפי שגילינו שלא אמין לטקסט.
- דפוס חובה: **retrieve-then-verify** - שליפה וקטורית (`n_results=3`, לצורך שקיפות/דיבוג) ואז LLM נפרד (`temperature=0`) ששופט בפועל אם התוצאה רלוונטית. **אף פעם** לא סף מרחק קבוע (`distance < X`) כקריטריון יחיד - הוכח לא אמין.
- כשהשיפוט תלוי בבקשת המשתמש - חובה להעביר גם את `state.get('user_input')` המקורי לפרומפט השיפוט, לא רק שדות שפורקו על ידי נודים קודמים (ראו: תובנת "פרצוף מחייך" ב-PROGRESS.md).

## משתני סביבה ומודלים

- שם המודל תמיד נקרא דרך `os.getenv("MODEL_X")`, אף פעם לא hardcoded בתוך נוד.
- `MODEL_JUDGE` הוא קבוע תמיד - אסור לשנות אותו בין הרצות eval שמשוות מודלים אחרים.
- כל ערך שנקרא מ-`os.getenv` שעלול להגיע מ-Secrets חיצוניים (לא רק `.env` מקומי) צריך לעבור `_sanitize_env_value()` (ב-`agent/tools.py`) - נועד למנוע `UnicodeEncodeError` מתווים נסתרים.

## עלות וטוקנים

מעקב עלות תמיד דרך `get_usage_metadata_callback()` של LangChain (context manager), לא ספירה ידנית. חישוב עלות דרך `estimate_cost()` ב-`tools.py`, לא נוסחה מקומית חדשה.

## Circular imports

קבצים מחוץ ל-`agent/` שצריכים לקרוא לפונקציות נוד **ישירות** (evals, סקריפטים) חייבים `import agent.graph` **לפני** `from agent import nodes` - אחרת circular import (ראו הסבר מלא ב-PROGRESS.md).

## בדיקות

לפני שמסמנים שינוי כ"הושלם" - להריץ בפועל דרך `python -m agent.cli` (לא רק לכתוב קוד ולהניח שהוא עובד), ולבדוק גם מקרה חיובי וגם מקרה גבולי/שלילי.