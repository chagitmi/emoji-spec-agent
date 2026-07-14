import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

# על Streamlit Cloud, Secrets מוגדרים בממשק שלהם - נוודא שהם גם ב-os.environ,
# כדי ש-agent/tools.py (שקורא דרך os.getenv) ימצא אותם באותה צורה בין אם
# מריצים מקומית (.env) או בענן (Secrets).
if hasattr(st, "secrets"):
    for key in st.secrets:
        os.environ[key] = str(st.secrets[key])
        
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph
from agent.tools import estimate_cost
from langchain_core.callbacks import get_usage_metadata_callback

st.set_page_config(page_title="Emoji Spec Agent", page_icon="🎨", layout="wide")

# --- עיצוב בסיסי + תמיכה ב-RTL לטקסט עברי ---
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

st.title("🎨 Emoji Spec Agent")
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

user_input = st.text_area(
    "תיאור האימוג'י",
    placeholder="לדוגמה: פרצוף שמח עם עיניים נוצצות",
    height=100,
)

generate_clicked = st.button("✨ צרי אימוג'י", type="primary", use_container_width=True)

if generate_clicked and user_input.strip():
    app = build_graph()
    status_placeholder = st.empty()

    node_display_names = {
        "guardrail": "בדיקת תחום",
        "emotion": "מזהה רגש...",
        "gesture": "מזהה מחווה...",
        "features": "מפרק למאפיינים...",
        "similarity_check": "בודק דמיון קיים...",
        "rag_and_spec": "מכינה מפרט לכל פלטפורמה...",
        "explanation": "מנסחת הסבר...",
        "evaluator": "בודקת איכות...",
        "output": "מסיימת...",
    }

    state = {"user_input": user_input, "retry_count": 0}

    with get_usage_metadata_callback() as usage_cb:
        for step in app.stream(state, stream_mode="updates"):
            node_name = list(step.keys())[0]
            status_placeholder.info(f"⏳ {node_display_names.get(node_name, node_name)}")
            if step[node_name]:
                state.update(step[node_name])

    status_placeholder.empty()

    # --- תוצאה ---
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

        # --- עלות ---
        total_cost = 0.0
        total_tokens = 0
        for model, usage in usage_cb.usage_metadata.items():
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            total_cost += estimate_cost(model, input_t, output_t)
            total_tokens += input_t + output_t

        st.caption(f"💰 עלות משוערת: ${total_cost:.5f} | {total_tokens} טוקנים")

elif generate_clicked:
    st.warning("נא להזין תיאור לפני שממשיכים")