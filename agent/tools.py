import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sys


def resource_path(relative_path: str) -> str:
    """
    מחזירה נתיב תקין לקובץ נתונים (JSON, rubric.md וכו'), בין אם רצים
    בפיתוח רגיל (python -m agent.cli) ובין אם מתוך agent.exe שנארז
    ב-PyInstaller (--onefile). ב-PyInstaller, קבצים שנארזו עם --add-data
    נשלפים בזמן ריצה לתיקייה זמנית שכתובתה ב-sys._MEIPASS.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

load_dotenv()


def _sanitize_env_value(value: str | None) -> str | None:
    """
    מנקה ערך שהגיע ממשתני סביבה מתווים בלתי-נראים (RTL marks, zero-width
    spaces, BOM וכו') שלפעמים "נדבקים" בהעתק-הדבק, ושוברים קידוד ASCII
    הנדרש ל-HTTP headers. שומר רק תווי ASCII רגילים.
    """
    if value is None:
        return None
    cleaned = value.strip()
    cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    return cleaned


OPENROUTER_API_KEY = _sanitize_env_value(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_BASE_URL = _sanitize_env_value(os.getenv("OPENROUTER_BASE_URL")) or "https://openrouter.ai/api/v1"

def get_llm(model_name: str, temperature: float = 0.3, max_tokens: int = 4000) -> ChatOpenAI:
    """
    מחזיר אובייקט LLM מוכן לשימוש, מחובר ל-OpenRouter.
    model_name: מזהה המודל כפי שמופיע ב-OpenRouter, למשל "openai/gpt-4o-mini"
    max_tokens: תקרת טוקנים לתשובה - מוגבלת בכוונה, כדי שמודלים לא "יבקשו" תקציב ענק
                שחורג מהיתרה הזמינה בחשבון (רלוונטי בעיקר למודלים עם max context גדול).
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("חסר OPENROUTER_API_KEY - ודאי שהוא מוגדר בקובץ .env")

    return ChatOpenAI(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
import json
import chromadb

_chroma_client = None
_existing_emojis_collection = None


def get_existing_emojis_collection():
    """
    בונה (בפעם הראשונה) או טוענת מאגר וקטורי (Chroma) עם תיאורי אימוג'ים קיימים,
    לצורך בדיקת דמיון בנוד 4.
    """
    global _chroma_client, _existing_emojis_collection

    if _existing_emojis_collection is not None:
        return _existing_emojis_collection

    _chroma_client = chromadb.Client()
    _existing_emojis_collection = _chroma_client.get_or_create_collection(
        name="existing_emojis",
        metadata={"hnsw:space": "cosine"},
    )

    if _existing_emojis_collection.count() == 0:
        data_path = resource_path(os.path.join("agent", "data", "existing_emojis.json"))
        with open(data_path, "r", encoding="utf-8") as f:
            emojis = json.load(f)

        _existing_emojis_collection.add(
            ids=[e["unicode"] for e in emojis],
            documents=[e["description"] for e in emojis],
            metadatas=[{"unicode": e["unicode"], "name": e["name"]} for e in emojis],
        )

    return _existing_emojis_collection
_style_guides_collection = None


def get_style_guides_collection():
    """בונה (בפעם הראשונה) או טוענת מאגר וקטורי עם קטעי מדריכי סגנון, לצורך נוד 6."""
    global _chroma_client, _style_guides_collection

    if _style_guides_collection is not None:
        return _style_guides_collection

    if _chroma_client is None:
        _chroma_client = chromadb.Client()

    _style_guides_collection = _chroma_client.get_or_create_collection(name="style_guides")

    if _style_guides_collection.count() == 0:
        data_path = resource_path(os.path.join("agent", "data", "style_guides.json"))        
        with open(data_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        _style_guides_collection.add(
            ids=[f"{c['platform']}_{c['topic']}_{i}" for i, c in enumerate(chunks)],
            documents=[c["text"] for c in chunks],
            metadatas=[{"platform": c["platform"], "topic": c["topic"]} for c in chunks],
        )

    return _style_guides_collection

# מחירים משוערים ב-USD לכל מיליון טוקנים (input, output).
# אלו מחירי OpenAI הרשמיים - ב-OpenRouter המחיר בפועל עשוי להיות מעט שונה בהתאם לספק.
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
}
DEFAULT_PRICING = {"input": 0.50, "output": 1.50}  # הערכה גסה למודלים שלא ברשימה


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """מעריכה עלות ב-USD, לפי מחירון ידוע או ברירת מחדל."""
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]