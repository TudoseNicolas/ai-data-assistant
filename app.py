"""AI Data Assistant: upload a table, fix its problems, ask questions in plain English.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ai_agent as agent
import data_quality as dq

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"
DEMOS = {"Sales (retail)": "sales_demo.csv", "HR (employees)": "hr_demo.csv"}
ACCENT = "#8B8CF8"

st.set_page_config(page_title="AI Data Assistant", page_icon="📊", layout="wide")
st.markdown(f"<style>{(APP_DIR / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)

ICON_CHART = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0B0D12" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"/>'
    '<path d="M7 15l4-4 3 3 5-6"/></svg>'
)
ICON_FILE = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#A5A6FA" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a1 1 0 0 0-1 1v16a1 '
    '1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/></svg>'
)


# helpers


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file. CSV separator and encoding are detected."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(BytesIO(data), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read this file.")


def get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")  # Streamlit Community Cloud
        except Exception:
            key = None
    return key or None


ENGINES = ["Cloud (Gemini)", "Local (Ollama)"]


@st.cache_resource(show_spinner=False)
def get_gemini(api_key: str | None):
    if not api_key:
        return None
    try:
        return agent.GeminiClient(api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_ollama(model: str, host: str):
    return agent.OllamaClient(model=model, host=host)


@st.cache_data(ttl=10, show_spinner=False)
def ollama_models(host: str) -> list[str] | None:
    return agent.OllamaClient(host=host).installed_models()


def choose_engine() -> tuple[object | None, str]:
    """Sidebar AI selector. Returns (client or None, status html)."""
    default = 1 if os.getenv("AI_PROVIDER", "").lower() == "ollama" else 0
    engine = st.selectbox("AI engine", ENGINES, index=default)
    ok, warn = '<span class="dot" style="background:#34D399"></span>', '<span class="dot" style="background:#FCD34D"></span>'

    if engine == ENGINES[0]:
        llm = get_gemini(get_api_key())
        if llm is None:
            return None, warn + "Gemini key missing (GEMINI_API_KEY in .env)"
        return llm, ok + "AI connected · Gemini"

    host = os.getenv("OLLAMA_HOST", agent.DEFAULT_OLLAMA_HOST)
    model = os.getenv("OLLAMA_MODEL", agent.DEFAULT_OLLAMA_MODEL)
    installed = ollama_models(host)
    if installed is None:
        return None, warn + "Ollama is not running"
    if not any(name == model or name.split(":")[0] == model for name in installed):
        return None, warn + f"Model not downloaded: run <code>ollama pull {model}</code>"
    return get_ollama(model, host), ok + f"AI connected · Local ({model})"


def to_excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


def style_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E6E8EE"),
        colorway=[ACCENT, "#2DD4BF", "#FB923C", "#60A5FA", "#F0ABFC"],
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def type_chip(column: str, kind: str) -> str:
    css = {"number": "t-number", "date": "t-date", "boolean": "t-number"}.get(kind, "t-text")
    warn = "should be" in kind
    return (f'<span class="chip{" warn" if warn else ""}">{column} · '
            f'<b class="{"t-warn" if warn else css}">{kind}</b></span>')


def load_source(name: str, data: bytes) -> None:
    """Reset the app state when a different file is chosen."""
    key = (name, len(data))
    if st.session_state.get("source_key") == key:
        return
    df = read_table(name, data)
    st.session_state.update(
        source_key=key, source_name=name, df_original=df, df=df,
        cleaned=False, clean_log=[], chat=[],
    )


# sidebar

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-mark">{ICON_CHART}</div><div>'
        '<div class="brand-name">AI Data Assistant</div>'
        '<div class="brand-sub">Clean · Explore · Ask</div></div></div>'
        '<p class="section-label">Data source</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    demo = st.selectbox("or try a demo dataset", list(DEMOS), disabled=uploaded is not None)

    try:
        if uploaded is not None:
            load_source(uploaded.name, uploaded.getvalue())
        else:
            load_source(DEMOS[demo], (SAMPLE_DIR / DEMOS[demo]).read_bytes())
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    df: pd.DataFrame = st.session_state.df
    label = st.session_state.source_name + (" · cleaned" if st.session_state.cleaned else "")
    st.markdown(
        f'<div class="file-card">{ICON_FILE}<div><div class="name">{label}</div>'
        f'<div class="meta">{len(df):,} rows · {df.shape[1]} columns</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
    llm, status = choose_engine()
    st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)


# main

st.markdown(
    f'<h1 class="page-title" style="margin-bottom:1rem !important">{st.session_state.source_name}</h1>',
    unsafe_allow_html=True,
)
SECTIONS = ["Overview", "Data Quality", "Ask AI"]
picked = st.segmented_control("Section", SECTIONS, default="Overview", key="section_picker",
                              label_visibility="collapsed")
section = picked or st.session_state.get("section_last", "Overview")  # None when the active tab is clicked again
st.session_state.section_last = section
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

# Overview
if section == "Overview":
    info = dq.overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", info["columns"])
    c3.metric("Missing values", f"{info['missing_pct']:.1f}%")
    c4.metric("Duplicate rows", f"{info['duplicates']:,}")

    with st.container(key="preview"):
        st.markdown(f'<h3>Data preview</h3><p class="page-sub">All {len(df):,} rows · scroll, sort, search, or use the '
                    'full-screen button at the top right of the table</p>',
                    unsafe_allow_html=True)
        st.dataframe(df, hide_index=True, width="stretch")

    chips = "".join(type_chip(col, kind) for col, kind in info["types"].items())
    st.markdown(f'<div class="card"><h3>Column types</h3><div class="chips">{chips}</div></div>',
                unsafe_allow_html=True)

# Data Quality
if section == "Data Quality":
    issues = dq.analyze(df)
    left, right = st.columns([2, 1], gap="large")

    with left:
        if not issues:
            st.success("No problems found. Your data looks clean.")
        else:
            st.markdown(f"**{len(issues)} issues found.** Choose what to fix, then clean in one click.")
            selected = []
            for issue in issues:
                box, text = st.columns([0.05, 0.95])
                checked = box.checkbox(issue.title, value=issue.severity != "low",
                                       key=f"fix_{issue.id}", label_visibility="collapsed")
                text.markdown(
                    f'<div><span class="badge {issue.severity}">{issue.severity.upper()}</span>'
                    f'<span class="issue-title">{issue.title}</span>'
                    f'<div class="issue-desc">{issue.description}</div></div>',
                    unsafe_allow_html=True,
                )
                if checked:
                    selected.append(issue.id)
            st.caption(f"{len(selected)} of {len(issues)} fixes selected")
            if st.button("Clean data", type="primary", disabled=not selected):
                cleaned, log = dq.clean(df, issues, selected)
                st.session_state.update(df=cleaned, cleaned=True,
                                        clean_log=st.session_state.clean_log + log, chat=[])
                st.rerun()

        if st.session_state.clean_log:
            with st.expander("What was changed"):
                for line in st.session_state.clean_log:
                    st.markdown(f"- {line}")

    with right:
        original = st.session_state.df_original
        before, after = dq.overview(original), dq.overview(df)
        score = dq.quality_score(df)

        def compare(label: str, old: str, new: str, good: bool = False) -> str:
            old_html = f"<s>{old}</s>" if st.session_state.cleaned and old != new else ""
            return (f'<div class="compare"><span class="label">{label}</span>'
                    f'<span>{old_html}<b class="{"good" if good else ""}">{new}</b></span></div>')

        title = "After cleaning" if st.session_state.cleaned else "Current state"
        st.markdown(
            f'<div class="card"><h3>{title}</h3>'
            + compare("Rows", f"{before['rows']:,}", f"{after['rows']:,}")
            + compare("Missing values", f"{before['missing_pct']:.1f}%", f"{after['missing_pct']:.1f}%",
                      good=after["missing_pct"] == 0)
            + compare("Duplicate rows", f"{before['duplicates']:,}", f"{after['duplicates']:,}",
                      good=after["duplicates"] == 0)
            + f'<div class="score-bar"><div style="width:{score}%"></div></div>'
            + f'<div class="score-text">Data quality score: {score} / 100</div></div>',
            unsafe_allow_html=True,
        )
        stem = Path(st.session_state.source_name).stem + ("_cleaned" if st.session_state.cleaned else "")
        st.download_button("Download as Excel", to_excel(df), f"{stem}.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), f"{stem}.csv",
                           mime="text/csv", width="stretch")
        if st.session_state.cleaned and st.button("Undo cleaning", width="stretch"):
            st.session_state.update(df=original, cleaned=False, clean_log=[], chat=[])
            st.rerun()
        st.caption("Your original file is never changed.")

# Ask AI
if section == "Ask AI":
    st.markdown("Ask questions in plain English. The AI writes the analysis code, "
                "runs it on your data and shows the result.")
    if llm is None:
        st.warning("The AI engine chosen in the sidebar isn’t ready. For **Cloud**, add `GEMINI_API_KEY=...` "
                   "to `.env` and restart. For **Local**, install Ollama from ollama.com and run "
                   f"`ollama pull {os.getenv('OLLAMA_MODEL', agent.DEFAULT_OLLAMA_MODEL)}`.")
    elif isinstance(llm, agent.OllamaClient):
        st.caption("Running on your computer: answers can take 10–60 seconds, and your data never leaves this machine.")

    question = None
    suggestions = agent.suggest_questions(df)
    if suggestions:
        st.caption("Suggested for this file")
        with st.container(key="suggestions"):
            cols = st.columns(2)
            for i, text in enumerate(suggestions):
                if cols[i % 2].button(text, key=f"suggest_{text}", disabled=llm is None, width="stretch"):
                    question = text

    avatars = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
    for message in st.session_state.chat:
        with st.chat_message(message["role"], avatar=avatars[message["role"]]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue
            answer: agent.AgentAnswer = message["content"]
            if answer.error:
                st.error(f"I couldn’t answer this one. {answer.error}")
            else:
                if answer.explanation:
                    st.markdown(answer.explanation.replace("$", r"\$"))  # escape $ (markdown math)
                if answer.fig is not None:
                    st.plotly_chart(style_figure(answer.fig), width="stretch", key=f"fig_{id(answer)}",
                                    config={"displayModeBar": False})
                if isinstance(answer.result, (pd.DataFrame, pd.Series)):
                    st.dataframe(answer.result, width="stretch")
                elif answer.result is not None and answer.fig is None:
                    st.markdown(f"### {answer.result}")
            if answer.code:
                with st.expander("Show generated code"):
                    st.code(answer.code, language="python")
            if not answer.error:
                st.markdown(f'<span class="ok-line">Code ran on all {len(df):,} rows · '
                            f'{answer.attempts} attempt{"s" if answer.attempts > 1 else ""}'
                            f'{" · " + answer.model if answer.model else ""}</span>',
                            unsafe_allow_html=True)

    typed = st.chat_input("Ask a question about your data…", disabled=llm is None)
    question = typed or question
    if question and llm is not None:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            answer = agent.ask(df, question, llm)
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.rerun()
