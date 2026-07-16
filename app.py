import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

try:
    for key in st.secrets:
        os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # מריצות מקומיות: אין secrets.toml, ה-.env כבר מטפל בזה בהמשך

from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph
from agent.tools import estimate_cost
from langchain_core.callbacks import get_usage_metadata_callback

#st.set_page_config(page_title="Emoji Spec Agent", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    .rtl-text { direction: rtl; text-align: right; }
    div[data-testid="stTextArea"] textarea { direction: rtl; text-align: right; }
    .platform-card {
        background-color: #f7f7f9;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        direction: rtl;
        text-align: right;
    }
    .platform-title { font-size: 20px; font-weight: 700; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


title_col, help_col = st.columns([6, 1])
with title_col:
    st.title("🎨 Emoji Spec Agent")
with help_col:
    st.write("")  # יישור אנכי מול הכותרת
    with st.popover("❓ עזרה", use_container_width=True):
        st.markdown(
                                            "**איך להשתמש:**\n\n"
            "1. תארי במילים שלך אימוג׳י שתרצי לעצב, או העלי קובץ טקסט (.txt) עם התיאור.\n"
            "2. לחצי על 'צרי אימוג'י'.\n"
            "3. האייג'נט יזהה רגש ומחווה, יבדוק אם כבר קיים אימוג'י דומה, "
            "ואם לא - יכין מפרט עיצובי מלא לכל פלטפורמת יעד (Apple/Google/Samsung), "
            "עם הסבר להחלטות ועלות משוערת.\n\n"
            "**דוגמאות לתיאורים:** \"פרצוף שמח עם עיניים נוצצות\", "
            "\"אימוג'י של רובוט עם עיניים כחולות\", \"ידיים מוחאות כפיים בהתלהבות\"."
        )
st.markdown(
    '<p class="rtl-text" style="font-size:18px; color:gray;">'
    'תארי אימוג\'י במילים שלך, וקבלי מפרט עיצובי מותאם לכל פלטפורמה'
    '</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("איך זה עובד")
    st.markdown(
        '<div class="rtl-text">'
        'האייג\'נט מזהה את הרגש והמחווה בתיאור שלך, בודק אם כבר קיים אימוג\'י '
        'דומה, ואם לא - מייצר מפרט עיצובי מלא (צבעים, צורות, קווי מתאר) '
        'לכל פלטפורמת יעד (Apple / Google / Samsung), עם הסבר להחלטות.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("נבנה עם LangGraph + OpenRouter + RAG + LLM-as-judge")

# --- קלט: טקסט או קובץ ---
input_tab, file_tab = st.tabs(["✍️ תיאור טקסטואלי", "📄 העלאת קובץ"])

with input_tab:
    text_input = st.text_area(
        "תיאור האימוג'י",
        placeholder="לדוגמה: פרצוף שמח עם עיניים נוצצות",
        height=100,
        label_visibility="collapsed",
    )

with file_tab:
    uploaded_file = st.file_uploader("גררי לכאן קובץ טקסט (.txt) עם התיאור", type=["txt"])
    file_content = ""
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8").strip()
            st.text_area("תוכן שנקרא מהקובץ:", value=file_content, height=100, disabled=True)
        except UnicodeDecodeError:
            st.error("לא הצלחתי לקרוא את הקובץ - ודאי שהוא שמור בקידוד UTF-8.")

# עדיפות לקובץ אם הועלה בפועל, אחרת טקסט ידני - עם תיוג ברור של המקור
if file_content.strip():
    user_input = file_content.strip()
    input_source_label = "📄 קובץ שהועלה"
else:
    user_input = text_input.strip()
    input_source_label = "✍️ תיאור טקסטואלי"

generate_clicked = st.button("✨ צרי אימוג'י", type="primary", use_container_width=True)

if generate_clicked and not user_input:
    st.warning("נא להזין תיאור או להעלות קובץ לפני שממשיכים")

elif generate_clicked and user_input:
    app = build_graph()
    status_placeholder = st.empty()

    node_display_names = {
        "guardrail": "בדיקת תחום",
        "emotion": "מזהה רגש...",
        "gesture": "מזהה מחווה...",
        "features": "מפרק למאפיינים...",
        "similarity_check": "בודק דמיון קיים...",
        "rag_and_spec": "מכין מפרט לכל פלטפורמה...",
        "explanation": "מנסח הסבר...",
        "evaluator": "בודק איכות...",
        "output": "מסיים...",
    }

    state = {"user_input": user_input, "retry_count": 0}

    try:
        with get_usage_metadata_callback() as usage_cb:
            for step in app.stream(state, stream_mode="updates"):
                node_name = list(step.keys())[0]
                status_placeholder.info(f"⏳ {node_display_names.get(node_name, node_name)}")
                if step[node_name]:
                    state.update(step[node_name])
        status_placeholder.empty()
        run_failed = False
    except Exception as e:
        status_placeholder.empty()
        print(f"[ERROR] {type(e).__name__}: {e}")
        st.error(
            "😕 משהו השתבש בזמן יצירת המפרט. זה יכול לקרות בגלל עומס זמני על "
            "השירות או בעיית תקשורת. נסי שוב בעוד רגע - אם זה נמשך, יש לפנות לתמיכה."
        )
        run_failed = True

    if not run_failed:
        st.caption(f"תוצאה עבור: {input_source_label}")

        if not state.get("is_in_scope", True):
            st.error(f"🚫 בקשה נחסמה: {state.get('guardrail_reason', '')}")

        elif state.get("similar_emoji_found"):
            details = state.get("similar_emoji_details", {})
            st.warning(
                f"כבר קיים אימוג'י דומה מאוד: **{details.get('unicode')} "
                f"({details.get('name')})** - לא נדרש עיצוב חדש."
            )

        else:
            st.success("המפרט מוכן!")

            st.markdown(
                f'<div class="rtl-text"><b>הסבר:</b><br>{state.get("explanation", "")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

            spec = state.get("spec", {})
            cols = st.columns(len(spec) if spec else 1)

            for col, (platform, details) in zip(cols, spec.items()):
                with col:
                    html = f'<div class="platform-card"><div class="platform-title">{platform}</div>'
                    for key, value in details.items():
                        if isinstance(value, list):
                            html += f"<b>{key}:</b><ul>"
                            for item in value:
                                html += f"<li>{item}</li>"
                            html += "</ul>"
                        else:
                            html += f"<p><b>{key}:</b> {value}</p>"
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)

            total_cost = 0.0
            total_tokens = 0
            for model, usage in usage_cb.usage_metadata.items():
                input_t = usage.get("input_tokens", 0)
                output_t = usage.get("output_tokens", 0)
                total_cost += estimate_cost(model, input_t, output_t)
                total_tokens += input_t + output_t

            st.caption(f"💰 עלות משוערת: ${total_cost:.5f} | {total_tokens} טוקנים")