import os
from typing import Optional
from pydantic import BaseModel, Field
from agent.graph import AgentState
from agent.tools import get_llm, get_existing_emojis_collection, get_style_guides_collection


# --- נוד 0: Guardrail ---
class GuardrailResult(BaseModel):
    is_in_scope: bool = Field(description="האם הבקשה שייכת לתחום עיצוב אימוג'י חדש")
    reason: str = Field(description="הסבר קצר להחלטה, בעברית")


def guardrail_node(state: AgentState) -> dict:
    """בודק שהבקשה בתחום עיצוב אימוג'י, באמצעות LLM."""
    print("[נוד] Guardrail")

    llm = get_llm(os.getenv("MODEL_UNDERSTANDING"))
    structured_llm = llm.with_structured_output(GuardrailResult)

    prompt = f"""את שופטת אם בקשת משתמש שייכת לתפקיד של אייג'נט שמתמחה אך ורק
בתכנון ועיצוב אימוג'ים חדשים (זיהוי רגש, מחווה, מאפיינים חזותיים, התאמה לפלטפורמות כמו Apple/Google/Samsung).

בקשות שאינן קשורות לעיצוב אימוג'י (למשל: שאלות כלליות, קוד, מתכונים, ייעוץ) - חוסמים אותן.

בקשת המשתמש:
"{state['user_input']}"
"""

    result = structured_llm.invoke(prompt)
    print(f"    -> is_in_scope={result.is_in_scope} | {result.reason}")

    return {"is_in_scope": result.is_in_scope, "guardrail_reason": result.reason}


# --- נוד 1: הבנת רגש ---
class EmotionResult(BaseModel):
    emotion: str = Field(description="הרגש המרכזי שהאימוג'י אמור לבטא, במילה או שתיים בעברית, למשל 'שמחה', 'תסכול', 'הפתעה'")


def emotion_node(state: AgentState) -> dict:
    """מזהה את הרגש המרכזי בתיאור, באמצעות LLM."""
    print("[נוד] הבנת רגש")

    llm = get_llm(os.getenv("MODEL_UNDERSTANDING"))
    structured_llm = llm.with_structured_output(EmotionResult)

    prompt = f"""זהי את הרגש המרכזי שהמשתמש רוצה שהאימוג'י יבטא, מתוך התיאור הבא:
"{state['user_input']}"

החזירי רגש אחד מרכזי, לא רשימה.
"""

    result = structured_llm.invoke(prompt)
    print(f"    -> emotion={result.emotion}")

    return {"emotion": result.emotion}


# --- נוד 2: זיהוי מחווה ---
class GestureResult(BaseModel):
    gesture: str = Field(description="הפעולה/מחווה הפיזית המרכזית, למשל 'עיניים נוצצות', 'יד מנופנפת', 'גבות מורמות'")
    has_clear_gesture: bool = Field(description="האם יש בתיאור מחווה פיזית ברורה, להבדיל מתיאור רגש בלבד")


def gesture_node(state: AgentState) -> dict:
    """מזהה את הפעולה/מחווה הפיזית, באמצעות LLM."""
    print("[נוד] זיהוי מחווה")

    llm = get_llm(os.getenv("MODEL_UNDERSTANDING"))
    structured_llm = llm.with_structured_output(GestureResult)

    prompt = f"""בהינתן התיאור הבא של אימוג'י, והרגש שכבר זוהה בו ("{state.get('emotion')}"),
חלצי את המחווה/הפעולה הפיזית הספציפית שמתוארת (אם יש כזו):

"{state['user_input']}"
"""

    result = structured_llm.invoke(prompt)
    print(f"    -> gesture={result.gesture} | has_clear_gesture={result.has_clear_gesture}")

    return {"gesture": result.gesture}


# --- נוד 3: פירוק למאפיינים ---
class FeaturesResult(BaseModel):
    eyes: str = Field(description="תיאור קצר של העיניים - צורה, גודל, ביטוי")
    eyebrows: str = Field(description="תיאור קצר של הגבות - זווית, מיקום")
    mouth: str = Field(description="תיאור קצר של הפה - צורה, פתיחות, חיוך/עגמומיות")
    additional_elements: list[str] = Field(
        description="רכיבים ויזואליים נוספים ומיוחדים, אם יש (למשל: דמעות, ברק בעיניים, סומק, זיעה) - רשימה ריקה [] אם אין. בלי אימוג'ים, רק תיאור מילולי."
    )
    body_or_hands: Optional[str] = Field(
        default=None,
        description="תיאור יד/גוף אם יש מחווה פיזית שכוללת אותם. השאירי את השדה ריק (אל תכתבי כלום) אם מדובר בפרצוף בלבד."
    )
    


def features_node(state: AgentState) -> dict:
    """מפרק את הרגש+המחווה למאפיינים ויזואליים נפרדים, באמצעות LLM."""
    print("[נוד] פירוק למאפיינים")

    llm = get_llm(os.getenv("MODEL_UNDERSTANDING"))
    structured_llm = llm.with_structured_output(FeaturesResult)

    prompt = f"""בהינתן התיאור המקורי, הרגש שזוהה, והמחווה שזוהתה - פרקי את זה
לרכיבים ויזואליים נפרדים וקונקרטיים, כפי שהם צריכים להיראות באימוג'י:

תיאור מקורי: "{state['user_input']}"
רגש: {state.get('emotion')}
מחווה: {state.get('gesture')}

תני תיאור קצר וקונקרטי לכל רכיב (לא משפט שלם, מילים ספורות).
חשוב: תארי במילים בלבד - אסור להשתמש בתווי אימוג'י (😀🙌 וכו') בתשובה שלך, אנחנו רק מתכננים את העיצוב, לא מציגים אימוג'י קיים.
"""

    result = structured_llm.invoke(prompt)
    print(f"    -> features={result.model_dump()}")

    return {"features": result.model_dump()}

## --- נוד 4: בדיקת דמיון קיים ---
class SimilarityJudgment(BaseModel):
    is_similar_enough: bool = Field(description="האם אחד המועמדים דומה מספיק כדי לוותר על עיצוב אימוג'י חדש")
    matched_unicode: Optional[str] = Field(default=None, description="ה-unicode של האימוג'י שנמצא דומה, אם is_similar_enough=True")
    reasoning: str = Field(description="הסבר קצר לפסיקה, בעברית")


def similarity_check_node(state: AgentState) -> dict:
    """מוצאת מועמדים דומים דרך חיפוש וקטורי (RAG), ואז משתמשת ב-LLM כדי לשפוט אם הם באמת דומים מספיק."""
    print("[נוד] בדיקת דמיון קיים")

    collection = get_existing_emojis_collection()

    features = state.get("features", {}) or {}
    query_text = (
        f"רגש: {state.get('emotion', '')}. "
        f"מחווה: {state.get('gesture', '')}. "
        f"עיניים: {features.get('eyes', '')}. "
        f"פה: {features.get('mouth', '')}. "
        f"רכיבים נוספים: {', '.join(features.get('additional_elements', []))}"
    )

    results = collection.query(query_texts=[query_text], n_results=3)

    candidates = []
    print("    -> מועמדים מהחיפוש הוקטורי:")
    for i in range(len(results["distances"][0])):
        d = results["distances"][0][i]
        m = results["metadatas"][0][i]
        desc = results["documents"][0][i]
        print(f"       {m['unicode']} ({m['name']}) | distance={d:.3f}")
        candidates.append({"unicode": m["unicode"], "name": m["name"], "description": desc, "distance": d})

    llm = get_llm(os.getenv("MODEL_UNDERSTANDING"), temperature=0.0)
    structured_llm = llm.with_structured_output(SimilarityJudgment)

    candidates_text = "\n".join(
        f"- {c['unicode']} ({c['name']}): {c['description']}" for c in candidates
    )

    prompt = f"""בהינתן בקשה ליצירת אימוג'י חדש, ורשימת אימוג'ים קיימים שהכי "קרובים" אליה
(לפי חיפוש וקטורי - אבל זה לא אומר שהם באמת דומים!):

בקשה חדשה:
רגש: {state.get('emotion')}
מחווה: {state.get('gesture')}
מאפיינים: {features}

אימוג'ים קיימים "מועמדים" (מהחיפוש):
{candidates_text}

שפטי: האם אחד מהמועמדים האלה **דומה מספיק בפועל** כדי לומר שאין צורך באימוג'י חדש?
היי קפדנית - "קרוב יחסית במאגר קטן" זה לא אותו דבר כמו "דומה מספיק". אם יש הבדל
משמעותי במשמעות/רגש/מחווה - זה כן צריך אימוג'י חדש, גם אם המרחק הוקטורי נמוך.

חשוב: אם הבקשה **כללית ובסיסית** (בלי פרטים ייחודיים כמו צבע ספציפי, אובייקט
נוסף, או מחווה חריגה), ואחד המועמדים כבר מייצג בדיוק את אותו הרגש הבסיסי -
זה **כן** נחשב דומה מספיק, גם אם הניסוח מילולית שונה. "פרצוף מחייך" הוא
בקשה כללית שכבר יש לה מענה בסיסי קיים; לעומת זאת "פרצוף מחייך עם עיניים
נוצצות וסומק ורוד" כולל פרטים ייחודיים שמצדיקים אימוג'י חדש."""

    judgment = structured_llm.invoke(prompt)
    print(f"    -> is_similar={judgment.is_similar_enough} | {judgment.reasoning}")

    if judgment.is_similar_enough and judgment.matched_unicode:
        matched = next((c for c in candidates if c["unicode"] == judgment.matched_unicode), candidates[0])
        return {
            "similar_emoji_found": True,
            "similar_emoji_details": {"unicode": matched["unicode"], "name": matched["name"], "distance": matched["distance"]},
        }

    return {"similar_emoji_found": False}

# --- נוד 5: Gate ---
def needs_new_emoji_gate(state: AgentState) -> str:
    """פונקציית Gate - לא משנה state, רק מחזירה את שם הצומת הבא."""
    if state.get("similar_emoji_found"):
        return "skip_to_output"
    return "continue_to_rag"


# --- נוד 6: RAG + מפרט ---
class PlatformSpec(BaseModel):
    colors: str = Field(description="תיאור פלטת הצבעים המומלצת, מותאמת לסגנון הפלטפורמה")
    shapes: str = Field(description="תיאור הצורות והגיאומטריה, מותאם לסגנון הפלטפורמה")
    outline_style: str = Field(description="תיאור סגנון קווי המתאר (עובי, האם קיימים בכלל, צבע)")
    unique_traits: list[str] = Field(description="מאפיינים ייחודיים לפלטפורמה שיש לשלב באימוג'י הזה")


def rag_and_spec_node(state: AgentState) -> dict:
    """שולף מדריך סגנון לכל פלטפורמת יעד, ומנסח מפרט מותאם, באמצעות RAG + LLM."""
    print("[נוד] RAG + התאמה לפלטפורמה")

    platforms = state.get("target_platforms") or ["Apple", "Google", "Samsung"]
    features = state.get("features", {}) or {}

    collection = get_style_guides_collection()
    llm = get_llm(os.getenv("MODEL_DRAFTING") or os.getenv("MODEL_UNDERSTANDING"))
    structured_llm = llm.with_structured_output(PlatformSpec)

    spec_by_platform = {}
    all_context = []

    for platform in platforms:
        results = collection.query(
            query_texts=[f"{state.get('emotion', '')} {state.get('gesture', '')}"],
            n_results=3,
            where={"platform": platform},
        )
        context_chunks = results["documents"][0]
        all_context.extend(context_chunks)

        prompt = f"""בהינתן המאפיינים הוויזואליים הבאים של אימוג'י חדש:
רגש: {state.get('emotion')}
מחווה: {state.get('gesture')}
עיניים: {features.get('eyes', '')}
גבות: {features.get('eyebrows', '')}
פה: {features.get('mouth', '')}
רכיבים נוספים: {', '.join(features.get('additional_elements', []))}

ומדריך הסגנון הבא של פלטפורמת {platform}:
{chr(10).join(f"- {c}" for c in context_chunks)}

נסחי מפרט עיצובי מותאם ספציפית לסגנון של {platform} עבור האימוג'י הזה.
"""

        result = structured_llm.invoke(prompt)
        spec_by_platform[platform] = result.model_dump()
        print(f"    -> מפרט עבור {platform} הוכן")

    return {"spec": spec_by_platform, "style_guide_context": all_context}

# --- נוד 7: הסבר החלטות ---
def explanation_node(state: AgentState) -> dict:
    """מנמק את הבחירות שנעשו במפרט, תוך הפניה למדריכי הסגנון שנשלפו."""
    print("[נוד] הסבר החלטות")

    llm = get_llm(os.getenv("MODEL_DRAFTING") or os.getenv("MODEL_UNDERSTANDING"))

    context = state.get("style_guide_context", []) or []
    spec = state.get("spec", {}) or {}

    prompt = f"""בהינתן המפרט העיצובי הבא שנוצר לאימוג'י חדש (לכל פלטפורמת יעד בנפרד):

{spec}

ומדריכי הסגנון ששימשו כבסיס להחלטות:
{chr(10).join(f"- {c}" for c in context)}

כתבי הסבר קצר (2-4 משפטים) בעברית, ברור וידידותי למשתמש, שמנמק
למה המפרט נראה כך - איך כל פלטפורמה משפיעה על הבחירות, תוך התייחסות
לעקרונות מדריך הסגנון הרלוונטיים. אל תחזרי על כל המפרט מילה במילה,
רק תני תובנה כללית ומועילה.
"""

    result = llm.invoke(prompt)
    explanation_text = result.content

    print(f"    -> explanation נוצר ({len(explanation_text)} תווים)")

    return {"explanation": explanation_text}

# --- Evaluator ---
class EvaluationResult(BaseModel):
    relevance: int = Field(description="ציון 1-5 לרלוונטיות המפרט לבקשה המקורית")
    accuracy: int = Field(description="ציון 1-5 לדיוק והעקביות הוויזואלית של המפרט")
    completeness: int = Field(description="ציון 1-5 לשלמות המפרט - כל השדות מלאים לכל פלטפורמה")
    format_score: int = Field(description="ציון 1-5 לניקיון הפורמט - אין TODO, אין אימוג'ים בטקסט")
    feedback: str = Field(description="משוב קצר וקונקרטי - מה חסר או לתקן, בעברית")


def evaluator_node(state: AgentState) -> dict:
    """מדרג את המפרט שנוצר, לפי rubric קבוע (evals/rubric.md), באמצעות LLM-as-judge."""
    print("[נוד] Evaluator")

    retry_count = state.get("retry_count", 0) + 1

    rubric_path = os.path.join(os.path.dirname(__file__), "..", "evals", "rubric.md")
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric_text = f.read()

    judge_model = os.getenv("MODEL_JUDGE") or os.getenv("MODEL_UNDERSTANDING")
    llm = get_llm(judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluationResult)

    prompt = f"""את שופטת מפרט עיצובי לאימוג'י, לפי ה-rubric הבא:

{rubric_text}

--- הבקשה המקורית ---
{state.get('user_input')}

--- המאפיינים שזוהו ---
{state.get('features')}

--- המפרט שנוצר (לכל פלטפורמה) ---
{state.get('spec')}

דרגי לפי ארבעת הקריטריונים ותני משוב קצר.
"""

    result = structured_llm.invoke(prompt)

    total_score = (result.relevance + result.accuracy + result.completeness + result.format_score) / 4 / 5

    print(f"    -> ציון: {total_score:.2f} (רלוונטיות={result.relevance}, דיוק={result.accuracy}, שלמות={result.completeness}, פורמט={result.format_score}) | ניסיון #{retry_count}")
    if result.feedback:
        print(f"    -> משוב: {result.feedback}")

    return {"eval_score": total_score, "eval_feedback": result.feedback, "retry_count": retry_count}
def evaluator_gate(state: AgentState) -> str:
    """פונקציית Gate נוספת - האם לחזור ללולאה או להמשיך ל-Output."""
    score = state.get("eval_score", 0)
    retry_count = state.get("retry_count", 0)
    if score >= 0.7 or retry_count >= 3:
        return "output"
    return "retry"


# --- Output ---
def output_node(state: AgentState) -> dict:
    """מרכיב את הפלט הסופי להצגה ב-CLI."""
    print("[נוד] Output")

    if not state.get("is_in_scope", True):
        return {"final_output": f"בקשה נחסמה: {state.get('guardrail_reason', '')}"}

    if state.get("similar_emoji_found"):
        details = state.get("similar_emoji_details", {})
        return {
            "final_output": (
                f"כבר קיים אימוג'י דומה מאוד: {details.get('unicode')} ({details.get('name')}) "
                f"- לא נדרש עיצוב חדש. אם בכל זאת תרצי אימוג'י חדש, נסי לתאר מה מייחד "
                f"אותו לעומת הקיים."
            )
        }

    return {"final_output": state.get("spec", "לא נמצא מפרט - שגיאה לא צפויה")}