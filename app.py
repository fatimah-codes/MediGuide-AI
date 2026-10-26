"""MediGuide AI - Streamlit UI.  Run with:  streamlit run app.py

Data flow: form input -> build prompt inputs -> select cache -> build LLM + LLMChain
-> run chain -> parse JSON -> stream narrative -> render dashboard.
All backend logic lives in src/; this file only draws the interface.
"""
import json
import time
import uuid
from string import Template

import altair as alt
import pandas as pd
import streamlit as st

from src import cache_manager as cm
from src import config as cfg
from src import utils as U
from src.chains import (
    benchmark_cache,
    build_assessment_chain,
    build_llm,
    demo_message_conversation,
    friendly_error,
    run_assessment,
    stream_narrative,
    validate_api_key,
)
from src.prompts import (
    ASSESSMENT_CHAT_TEMPLATE,
    JSON_SCHEMA,
    NARRATIVE_SYSTEM,
    PATIENT_PROFILE_PROMPT,
    SYSTEM_PROMPT,
)

st.set_page_config(page_title=cfg.APP_NAME, page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")


# =============================================================================
# 1. Session state defaults
# =============================================================================
def init_state() -> None:
    defaults = {
        "ui_dark": True, "ui_palette": "Aurora", "reduce_motion": False,
        "model": cfg.DEFAULT_MODEL, "temperature": 0.2, "sb_lang": "English",
        "cache_mode": cm.CACHE_MEMORY, "history": [], "current": None, "theme_n": 0,
        "f_age": "", "f_gender": cfg.GENDERS[0], "f_symptoms": [], "f_free": "",
        "f_duration": cfg.DURATION_OPTIONS[0], "f_severity": 3, "f_cond": "",
        "f_meds": "", "f_notes": "", "f_lang": "English",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

# =============================================================================
# 2. Animated theme engine (dark / light, 4 palettes, animated transition)
# =============================================================================
CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
:root{--accent:$accent;--accent2:$accent2;--text:$text;--muted:$muted;--glass:$glass;
--glass-b:$glass_b;--inp:$inp;--shadow:$shadow;--pop:$pop;}
.stApp{background:linear-gradient(-45deg,$bg1,$bg2,$bg3,$bg4);background-size:400% 400%;
animation:gradientShift 24s ease infinite;color:var(--text);}
@keyframes gradientShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea,.stApp td,.stApp th
{font-family:'Poppins',sans-serif !important;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp li,.stApp label,.stApp td,.stApp th,
[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p{color:var(--text) !important;}
[data-testid="stCaptionContainer"] p{color:var(--muted) !important;}

/* floating orbs */
.orb{position:fixed;border-radius:50%;filter:blur(70px);opacity:$orb_op;z-index:0;pointer-events:none;
animation:floaty 16s ease-in-out infinite;}
.orb.o1{width:340px;height:340px;background:var(--accent);top:-90px;left:-70px}
.orb.o2{width:400px;height:400px;background:var(--accent2);bottom:-130px;right:-90px;animation-delay:-5s}
.orb.o3{width:260px;height:260px;background:#ff6ec7;top:48%;left:55%;animation-delay:-9s;opacity:.25}
@keyframes floaty{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-50px) scale(1.12)}66%{transform:translate(-30px,40px) scale(.94)}}

header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{position:relative;z-index:1;padding-top:1.5rem;padding-bottom:6rem;max-width:1250px}
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{0%{opacity:0;transform:scale(.92)}100%{opacity:1;transform:scale(1)}}

/* smooth theme colour transitions */
.stApp,[data-testid="stSidebar"],[data-testid="stForm"],[data-testid="stMetric"],[data-testid="stExpander"] details,
[data-testid="stAlert"],.card,.disclaimer,.step,input,textarea,[data-baseweb="select"]>div{
transition:background-color .7s ease,color .7s ease,border-color .7s ease,box-shadow .4s ease,transform .3s ease;}

/* hero */
.hero{text-align:center;margin:.4rem 0 1.4rem;animation:fadeUp .9s ease both}
.hero .logo{font-size:3.2rem;display:inline-block;animation:beat 1.5s ease-in-out infinite}
@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.18)}30%{transform:scale(1)}45%{transform:scale(1.12)}}
.hero h1{font-size:3rem;font-weight:800;margin:0;line-height:1.1;
background:linear-gradient(90deg,var(--accent),var(--accent2),#ff6ec7,var(--accent));background-size:300% 100%;
-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 7s linear infinite}
@keyframes shine{to{background-position:300% 0}}
.hero p{color:var(--muted) !important;font-weight:300;letter-spacing:.4px;margin:.3rem 0 0}
.ecg{width:min(420px,80%);height:50px;margin-top:.4rem}
.ecg polyline{fill:none;stroke:var(--accent2);stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round;
stroke-dasharray:1;stroke-dashoffset:1;animation:draw 3.2s linear infinite;filter:drop-shadow(0 0 6px var(--accent2))}
@keyframes draw{0%{stroke-dashoffset:1}70%{stroke-dashoffset:0}100%{stroke-dashoffset:-1}}

/* disclaimer */
.disclaimer{border-left:6px solid #ef4444;background:rgba(239,68,68,.14);border-radius:14px;padding:.85rem 1.1rem;
margin:.2rem 0 1rem;font-size:.92rem;animation:fadeUp .8s ease both,glow 3.5s ease-in-out infinite}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.0)}50%{box-shadow:0 0 22px 0 rgba(239,68,68,.35)}}

/* glass cards */
[data-testid="stForm"],[data-testid="stExpander"] details,.card{background:var(--glass);backdrop-filter:blur(14px);
border:1px solid var(--glass-b) !important;border-radius:20px;box-shadow:var(--shadow);}
[data-testid="stForm"]{padding:1rem 1.2rem;animation:fadeUp .9s ease both}
[data-testid="stForm"]:hover{box-shadow:0 14px 48px color-mix(in srgb,var(--accent) 45%,transparent)}
.card{padding:1rem 1.2rem;margin-bottom:.8rem;animation:fadeUp .7s ease both}
[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:var(--text) !important}
[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--glass-b);border-radius:16px;padding:.8rem 1rem;
box-shadow:var(--shadow);animation:popIn .6s ease both}
[data-testid="stMetric"]:hover{transform:translateY(-5px)}
[data-testid="stMetricLabel"] p{color:var(--muted) !important}
[data-testid="stMetricValue"] div{color:var(--text) !important;font-weight:700}

/* sidebar */
[data-testid="stSidebar"]{background:$sidebar !important;backdrop-filter:blur(18px);border-right:1px solid var(--glass-b)}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-weight:700}

/* inputs */
.stTextInput [data-baseweb="input"],.stTextArea [data-baseweb="textarea"],[data-baseweb="select"]>div{
background:var(--inp) !important;border:1px solid var(--glass-b) !important;border-radius:12px !important}
.stTextInput input,.stTextArea textarea{background:transparent !important;color:var(--text) !important}
input::placeholder,textarea::placeholder{color:var(--muted) !important;opacity:1}
.stTextInput [data-baseweb="input"]:focus-within,.stTextArea [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"]>div:focus-within{border-color:var(--accent) !important;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 35%,transparent) !important}
[data-baseweb="select"] *{color:var(--text) !important}
[data-baseweb="select"] svg{fill:var(--text) !important}
[data-baseweb="tag"]{background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;border-radius:999px !important}
[data-baseweb="tag"] *{color:#fff !important}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] ul{background:var(--pop) !important}
[data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:var(--text) !important}
[data-baseweb="popover"] li:hover{background:color-mix(in srgb,var(--accent) 25%,transparent) !important}
[data-testid="stThumbValue"],[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:var(--text) !important;font-weight:600}
[data-testid="stRadio"] label p,[data-testid="stToggle"] label p{color:var(--text) !important}

/* tabs */
button[data-baseweb="tab"] p{color:var(--muted) !important;font-weight:600}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--text) !important}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--accent),var(--accent2)) !important;height:3px !important}
[data-baseweb="tab-border"]{background:var(--glass-b) !important}

/* buttons */
.stButton>button,[data-testid="stFormSubmitButton"]>button,[data-testid="stDownloadButton"]>button{
background:linear-gradient(135deg,var(--accent),var(--accent2),var(--accent));background-size:200% 200%;
border:none;border-radius:14px;font-weight:600;padding:.55rem 1.2rem;width:100%;
transition:transform .25s ease,box-shadow .25s ease,background-position .6s ease}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover,[data-testid="stDownloadButton"]>button:hover{
transform:translateY(-3px) scale(1.02);box-shadow:0 10px 28px color-mix(in srgb,var(--accent) 55%,transparent);background-position:100% 0}
.stButton>button p,[data-testid="stFormSubmitButton"]>button p,[data-testid="stDownloadButton"]>button p{color:#fff !important}

/* alerts (native st.warning / info / error / success) */
[data-testid="stAlert"]{background:var(--glass) !important;border:1px solid var(--glass-b) !important;border-radius:14px !important;
animation:fadeUp .6s ease both}
[data-testid="stAlert"] *{color:var(--text) !important}
[class*="st-key-steps_"] [data-testid="stAlert"]{border-left:6px solid #22c55e !important}
[class*="st-key-qs_"] [data-testid="stAlert"]{border-left:6px solid #38bdf8 !important}
[class*="st-key-warn_"] [data-testid="stAlert"]{border-left:6px solid #ef4444 !important}
[class*="st-key-flag_"] [data-testid="stAlert"]{border-left:6px solid #ef4444 !important;background:rgba(239,68,68,.18) !important}
[class*="st-key-info_"] [data-testid="stAlert"]{border-left:6px solid var(--accent2) !important}
[class*="st-key-disc_"] [data-testid="stAlert"]{border-left:6px solid #f59e0b !important}
table{color:var(--text)}

/* urgency hero card */
.urg{display:flex;align-items:center;gap:1.2rem;border-radius:22px;padding:1.2rem 1.6rem;margin:.6rem 0 1rem;
background:linear-gradient(135deg,var(--u1),var(--u2));box-shadow:0 14px 40px color-mix(in srgb,var(--u1) 45%,transparent);
animation:popIn .7s cubic-bezier(.2,.9,.3,1.3) both}
.urg *{color:#fff !important}
.urg-emoji{font-size:3rem;animation:beat 1.6s ease-in-out infinite}
.urg-label{font-size:.8rem;letter-spacing:2px;text-transform:uppercase;opacity:.85}
.urg-level{font-size:2.2rem;font-weight:800;line-height:1.1}
.urg-advice{font-size:.95rem;opacity:.95}
.urg.emg{animation:popIn .7s ease both,emgpulse 1.2s ease-in-out infinite}
@keyframes emgpulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.7)}50%{box-shadow:0 0 0 18px rgba(239,68,68,0)}}

/* chips + steps */
.chips{display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;margin:.5rem 0}
.chip{padding:.35rem .9rem;border-radius:999px;font-size:.82rem;font-weight:600;color:#fff;
background:linear-gradient(135deg,var(--accent),var(--accent2));animation:fadeUp .8s ease both}
.badge{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-size:.78rem;font-weight:600;
background:rgba(34,197,94,.2);border:1px solid rgba(34,197,94,.7);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,.5)}50%{box-shadow:0 0 0 9px rgba(34,197,94,0)}}

/* fixed footer disclaimer */
.footer-bar{position:fixed;left:0;right:0;bottom:0;z-index:50;text-align:center;font-size:.78rem;padding:.55rem 1rem;
background:$footer;backdrop-filter:blur(10px);border-top:1px solid var(--glass-b);color:var(--text)}

body:has([data-testid="stSidebar"][data-collapsed="false"]) .footer-bar,
body:has([data-testid="stSidebar"][aria-expanded="true"]) .footer-bar{padding-left:21.5rem}

/* ---- widget skin for newer Streamlit (react-aria based inputs / selects) ---- */
[data-testid="stTextInputRootElement"],[data-testid="stTextAreaRootElement"],
[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"],
[data-testid="stNumberInputContainer"]{background:var(--inp) !important;border:1px solid var(--glass-b) !important;
border-radius:12px !important;box-shadow:none !important}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within,[data-testid="stMultiSelect"] [role="group"]:focus-within{
border-color:var(--accent) !important;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 35%,transparent) !important}
[data-testid="stTextInputRootElement"] input,[data-testid="stTextAreaRootElement"] textarea,
[data-testid="stSelectbox"] input,[data-testid="stMultiSelect"] input{background:transparent !important;
color:var(--text) !important;-webkit-text-fill-color:var(--text) !important;caret-color:var(--text)}
[data-testid="stTextInputRootElement"] input::placeholder,[data-testid="stTextAreaRootElement"] textarea::placeholder,
[data-testid="stSelectbox"] input::placeholder,[data-testid="stMultiSelect"] input::placeholder{
color:var(--muted) !important;-webkit-text-fill-color:var(--muted) !important;opacity:1}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg,[data-testid="stTextInputRootElement"] svg{
fill:var(--text) !important;color:var(--text) !important}
[data-testid="stTextInputRootElement"] button,[data-testid="stSelectbox"] button,[data-testid="stMultiSelect"] button{
background:transparent !important;color:var(--text) !important;box-shadow:none !important;width:auto !important}
[role="listbox"]{background:var(--pop) !important;border:1px solid var(--glass-b) !important;border-radius:12px !important}
[role="option"],[role="option"] *{color:var(--text) !important;background:transparent !important}
[role="option"]:hover,[role="option"][data-focused="true"],[role="option"][aria-selected="true"]{
background:color-mix(in srgb,var(--accent) 25%,transparent) !important}
[data-testid="stSelectbox"] [role="group"] > div,[data-testid="stSelectbox"] [role="group"] span{color:var(--text) !important}

/* theme-change reveal curtain */
.curtain{position:fixed;inset:0;z-index:99999;pointer-events:none;
background:radial-gradient(circle at 92% 6%,var(--accent),$bg2);}
.curtain.a{animation:curtainA 1.1s ease-out forwards}
.curtain.b{animation:curtainB 1.1s ease-out forwards}
@keyframes curtainA{0%{clip-path:circle(0% at 92% 6%);opacity:.95}70%{clip-path:circle(150% at 92% 6%);opacity:.55}100%{clip-path:circle(150% at 92% 6%);opacity:0}}
@keyframes curtainB{0%{clip-path:circle(0% at 92% 6%);opacity:.95}70%{clip-path:circle(150% at 92% 6%);opacity:.55}100%{clip-path:circle(150% at 92% 6%);opacity:0}}
$motion
</style>
""")


def theme_tokens(dark: bool, palette_name: str) -> dict:
    pal = cfg.PALETTES[palette_name]
    bg1, bg2, bg3, bg4 = pal["dark"] if dark else pal["light"]
    base = {"accent": pal["accent"], "accent2": pal["accent2"], "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4}
    if dark:
        base.update(text="#f5f3ff", muted="rgba(245,243,255,.72)", glass="rgba(255,255,255,.08)",
                    glass_b="rgba(255,255,255,.18)", inp="rgba(255,255,255,.10)",
                    shadow="0 12px 40px rgba(0,0,0,.35)", pop="#1c1636",
                    sidebar="rgba(10,8,30,.72)", footer="rgba(10,8,30,.82)", orb_op=".42")
    else:
        base.update(text="#1e1b4b", muted="rgba(30,27,75,.72)", glass="rgba(255,255,255,.66)",
                    glass_b="rgba(30,27,75,.14)", inp="rgba(255,255,255,.88)",
                    shadow="0 12px 36px rgba(60,40,120,.18)", pop="#ffffff",
                    sidebar="rgba(255,255,255,.72)", footer="rgba(255,255,255,.85)", orb_op=".30")
    return base


def inject_theme() -> dict:
    dark, palette = st.session_state.ui_dark, st.session_state.ui_palette
    tokens = theme_tokens(dark, palette)
    tokens["motion"] = ("*{animation:none !important;transition:none !important}"
                        if st.session_state.reduce_motion else "")
    st.markdown(CSS.safe_substitute(tokens), unsafe_allow_html=True)
    st.markdown('<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>',
                unsafe_allow_html=True)

    # Play the "curtain" reveal whenever the theme or palette changes.
    sig = (dark, palette)
    prev = st.session_state.get("theme_sig")
    if prev is not None and prev != sig and not st.session_state.reduce_motion:
        st.session_state.theme_n += 1
        cls = "a" if st.session_state.theme_n % 2 else "b"
        st.markdown(f'<div class="curtain {cls}"></div>', unsafe_allow_html=True)
    st.session_state.theme_sig = sig
    return tokens


TOKENS = inject_theme()

ECG = ('<svg class="ecg" viewBox="0 0 400 60" preserveAspectRatio="none"><polyline pathLength="1" '
       'points="0,30 60,30 80,30 95,8 110,52 125,30 190,30 210,30 225,5 240,55 255,30 400,30"/></svg>')


def hero(title: str, subtitle: str, emoji: str = "🩺") -> None:
    st.markdown(U.html_block(f"""
        <div class="hero"><div class="logo">{emoji}</div><h1>{U.esc(title)}</h1>
        <p>{U.esc(subtitle)}</p>{ECG}</div>"""), unsafe_allow_html=True)


def disclaimer_banner() -> None:
    st.markdown(U.html_block(f"""
        <div class="disclaimer">⚠️ <b>Medical disclaimer:</b> {U.esc(cfg.DISCLAIMER_LONG)}
        This tool never provides a confirmed diagnosis. {U.esc(cfg.EMERGENCY_TEXT)}</div>"""),
        unsafe_allow_html=True)


# =============================================================================
# 3. Sidebar (always rendered so widget state is kept; hidden on the key screen)
# =============================================================================
def sync_language() -> None:
    st.session_state["f_lang"] = st.session_state["sb_lang"]


def resolve_api_key() -> str:
    # The user must always type their key on the welcome screen first;
    # a key in .env is deliberately NOT used to skip that step.
    return st.session_state.get("api_key", "")


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(f"## 🩺 {cfg.APP_NAME}")
        st.caption(cfg.APP_DESCRIPTION)
        with st.container(key="disc_sidebar"):
            st.warning(cfg.DISCLAIMER_SHORT, icon="⚠️")

        st.markdown("### 🎨 Appearance")
        st.toggle("🌙 Dark mode", key="ui_dark")
        st.selectbox("Colour palette", list(cfg.PALETTES), key="ui_palette")
        st.toggle("♿ Reduce motion", key="reduce_motion")

        st.markdown("### 🤖 Model configuration")
        st.selectbox("OpenAI model", cfg.MODELS, key="model")
        st.slider("Temperature", 0.0, 1.0, step=0.05, key="temperature",
                  help="Lower = more consistent answers. Higher = more varied wording.")

        st.markdown("### 🌐 Language")
        st.selectbox("Default answer language", cfg.LANGUAGES, key="sb_lang", on_change=sync_language)

        st.markdown("### ⚡ Caching")
        st.radio("LLM cache", cm.CACHE_MODES, key="cache_mode",
                 help="In-memory = RAM, lost on restart. SQLite = file on disk, survives restarts.")
        if st.session_state.cache_mode == cm.CACHE_SQLITE:
            st.caption(f"SQLite file size: {cm.sqlite_size_kb()} KB")
        if st.button("🧹 Clear caches"):
            cm.clear_cache()
            st.toast("Caches cleared", icon="🧹")

        st.markdown("### 🔑 API key")
        key = resolve_api_key()
        if key:
            st.markdown('<span class="badge">● API key connected</span>', unsafe_allow_html=True)
            with st.expander("Use a different key"):
                new_key = st.text_input("New OpenAI API key", type="password", key="new_key_input")
                if st.button("Apply key"):
                    ok, msg = validate_api_key(new_key.strip()) if new_key.strip() else (False, "Enter a key first.")
                    if ok:
                        st.session_state.api_key = new_key.strip()
                        st.rerun()
                    st.error(msg)
        else:
            st.caption("No key yet - enter it on the main screen.")

    return {
        "api_key": resolve_api_key(), "model": st.session_state.model,
        "temperature": st.session_state.temperature, "cache_mode": st.session_state.cache_mode,
    }


SETTINGS = render_sidebar()
cm.setup_cache(SETTINGS["cache_mode"])  # register the chosen cache before any model call

# Footer disclaimer is visible on every screen.
st.markdown(f'<div class="footer-bar">⚠️ {U.esc(cfg.DISCLAIMER_SHORT)} In an emergency, call your local emergency number.</div>',
            unsafe_allow_html=True)


# =============================================================================
# 4. API key gate (shown first when no key is found in .env or the session)
# =============================================================================
def api_key_gate() -> None:
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none}</style>",
                unsafe_allow_html=True)
    hero("Welcome to MediGuide AI", "Enter your OpenAI API key to get started", "🔐")
    disclaimer_banner()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("api_key_form"):
            key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
            go = st.form_submit_button("🚀 Continue")
        st.caption("Your key is kept only in this browser session and is never saved by the app.")
        if go:
            if not key.strip():
                st.warning("Please enter your API key first.")
            else:
                with st.spinner("Verifying your key..."):
                    ok, msg = validate_api_key(key.strip())
                if ok:
                    st.session_state.api_key = key.strip()
                    st.balloons()
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(msg)


if not SETTINGS["api_key"]:
    api_key_gate()
    st.stop()

# =============================================================================
# 5. Assessment pipeline
# =============================================================================
def load_sample(name: str) -> None:
    for key, value in cfg.SAMPLE_SCENARIOS[name].items():
        st.session_state[key] = value


def collect_prompt_inputs() -> dict:
    s = st.session_state
    return {
        "age": s.f_age.strip(), "gender": s.f_gender,
        "symptoms": U.format_symptoms(s.f_symptoms, s.f_free), "duration": s.f_duration,
        "severity": int(s.f_severity),
        "conditions": s.f_cond.strip() or "None reported",
        "medications": s.f_meds.strip() or "None reported",
        "notes": s.f_notes.strip() or "None", "language": s.f_lang,
    }


def validate_form(inputs: dict) -> str | None:
    if not st.session_state.f_symptoms and not st.session_state.f_free.strip():
        return "Please select at least one symptom or describe it in the free-text box. No request was sent."
    if not inputs["age"].isdigit() or not (0 <= int(inputs["age"]) <= 120):
        return "Please enter a valid age as a whole number between 0 and 120."
    return None


def run_pipeline(inputs: dict) -> None:
    """form input -> cache -> LLM + LLMChain -> parse JSON -> store result."""
    blob = " ".join([inputs["symptoms"], inputs["notes"], inputs["conditions"]])
    flags = U.detect_red_flags(blob)
    crisis = U.detect_crisis(blob)

    entry = {
        "id": uuid.uuid4().hex[:8], "time": U.now_str(), "prompt_inputs": inputs,
        "language": inputs["language"], "model": SETTINGS["model"],
        "cache_mode": SETTINGS["cache_mode"], "flags": flags, "crisis": crisis,
        "data": None, "raw": "", "error": None, "elapsed": 0.0, "cache_hit": False,
        "narrative": None,
    }

    with st.status("Analysing your symptoms...", expanded=True) as status:
        st.write("🧩 Building the prompt from your inputs")
        llm = build_llm(SETTINGS["api_key"], SETTINGS["model"], SETTINGS["temperature"], json_mode=True)
        chain = build_assessment_chain(llm)
        st.write(f"📡 Asking {SETTINGS['model']} (cache: {SETTINGS['cache_mode']})")
        start = time.perf_counter()
        try:
            entry["raw"] = run_assessment(chain, inputs)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = friendly_error(exc)
            status.update(label="Request failed", state="error", expanded=True)
            st.session_state.current = entry
            return
        entry["elapsed"] = time.perf_counter() - start
        entry["cache_hit"] = SETTINGS["cache_mode"] != cm.CACHE_OFF and entry["elapsed"] < 0.4
        st.write("🔎 Parsing the JSON safely")
        data, err = U.parse_assessment(entry["raw"])
        if err:
            # A bad reply may have been cached (or be a one-off glitch): retry once
            # with the cache bypassed so the user is never stuck with a cached failure.
            st.write("🔁 Reply was not valid JSON - retrying once without the cache")
            cm.setup_cache(cm.CACHE_OFF)
            try:
                entry["raw"] = run_assessment(chain, inputs)
                data, err = U.parse_assessment(entry["raw"])
            except Exception as exc:  # noqa: BLE001
                err = friendly_error(exc)
            finally:
                cm.setup_cache(SETTINGS["cache_mode"])
            entry["elapsed"] = time.perf_counter() - start
        if err:
            entry["error"] = err
            status.update(label="The model reply could not be parsed", state="error", expanded=True)
        else:
            entry["data"] = data
            status.update(label="Assessment ready", state="complete", expanded=False)

    st.session_state.current = entry
    st.session_state.history.append(entry)


# =============================================================================
# 6. Results dashboard
# =============================================================================
def narrative_generator(entry: dict):
    """Yield the narrative chunk by chunk; replays instantly when cached."""
    inputs = entry["prompt_inputs"]
    memo_key = U.inputs_hash(inputs, entry["model"], SETTINGS["temperature"], entry["data"]["summary"])
    if SETTINGS["cache_mode"] != cm.CACHE_OFF:
        cached = cm.get_narrative(memo_key)
        if cached:
            for piece in U.chunk_text(cached):
                yield piece
                time.sleep(0.012)
            return
    llm = build_llm(SETTINGS["api_key"], entry["model"], SETTINGS["temperature"])
    assessment_json = json.dumps({k: v for k, v in entry["data"].items() if k != "urgency_defaulted"},
                                 ensure_ascii=False)
    parts = []
    try:
        for piece in stream_narrative(llm, inputs, assessment_json):
            parts.append(piece)
            yield piece
    except Exception as exc:  # noqa: BLE001
        yield f"\n\n⚠️ Streaming stopped: {friendly_error(exc)}"
        return
    if SETTINGS["cache_mode"] != cm.CACHE_OFF:
        cm.put_narrative(memo_key, "".join(parts))


def render_results(entry: dict) -> None:
    eid = entry["id"]
    data = entry["data"]

    with st.container(key="results"):
        if entry["language"] in cfg.RTL_LANGUAGES:
            st.markdown("<style>.st-key-results [data-testid='stMarkdownContainer']"
                        "{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)

        st.markdown("## 📋 Your guidance")

        if entry["crisis"]:
            with st.container(key=f"flag_crisis_{eid}"):
                st.error(cfg.CRISIS_MESSAGE, icon="💙")
        for i, msg in enumerate(entry["flags"]):
            with st.container(key=f"flag_{eid}_{i}"):
                st.error(f"{msg} {cfg.EMERGENCY_TEXT}", icon="🚨")

        if entry["error"] or not data:
            st.error(f"Sorry - we could not produce guidance. {entry['error']}", icon="⚠️")
            if entry["raw"]:
                with st.expander("Raw model output (for debugging)"):
                    st.code(entry["raw"], language="json")
            st.info("Nothing crashed - please try again. If the problem continues, switch the model in the sidebar.",
                    icon="ℹ️")
            return

        level = data["urgency_level"]
        style = cfg.URGENCY_STYLE[level]
        st.markdown(U.html_block(f"""
            <div class="urg {'emg' if level == 'EMERGENCY' else ''}" style="--u1:{style['c1']};--u2:{style['c2']}">
            <div class="urg-emoji">{style['emoji']}</div>
            <div><div class="urg-label">Possible urgency level</div>
            <div class="urg-level">{level}</div><div class="urg-advice">{U.esc(style['advice'])}</div></div></div>"""),
            unsafe_allow_html=True)
        if level == "EMERGENCY":
            st.error(f"🚨 EMERGENCY: {cfg.EMERGENCY_TEXT}", icon="🚨")
        if data.get("urgency_defaulted"):
            st.warning("The model did not return a valid urgency level, so MEDIUM was used as a cautious default.",
                       icon="⚠️")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Urgency", f"{style['emoji']} {level}")
        c2.metric("Severity", f"{entry['prompt_inputs']['severity']}/10")
        c3.metric("Conditions", len(data["possible_conditions"]))
        c4.metric("Response time", f"{entry['elapsed']:.2f}s")
        if entry["cache_mode"] == cm.CACHE_OFF:
            c5.metric("Cache", "OFF")
        else:
            c5.metric("Cache", "⚡ HIT" if entry["cache_hit"] else "MISS", entry["cache_mode"], delta_color="off")

        st.markdown("### 🗣️ AI explanation (live)")
        narr_box = st.container()

        t1, t2, t3, t4, t5 = st.tabs(["📝 Summary", "🧬 Possible conditions", "✅ Next steps",
                                      "❓ Ask your doctor", "🚨 Warning signs"])
        with t1:
            profile = PATIENT_PROFILE_PROMPT.format(**entry["prompt_inputs"])
            st.markdown("**Patient symptom summary**")
            st.markdown(U.html_block(f'<div class="card">{U.esc(profile).replace(chr(10), "<br>")}</div>'),
                        unsafe_allow_html=True)
            with st.container(key=f"info_summary_{eid}"):
                st.info(data["summary"] or "No summary returned.", icon="ℹ️")
        with t2:
            st.caption("For education only - these are NOT diagnoses.")
            if not data["possible_conditions"]:
                st.write("No conditions were suggested.")
            for i, cond in enumerate(data["possible_conditions"]):
                with st.expander(f"{i + 1}. {cond['name']}", expanded=(i == 0)):
                    st.write(cond["reason"] or "-")
                    st.caption("Educational information only. Only a clinician can diagnose.")
        with t3:
            with st.container(key=f"steps_{eid}"):
                for step in data["recommended_next_steps"] or ["Monitor your symptoms and rest."]:
                    st.success(step, icon="✅")
        with t4:
            with st.container(key=f"qs_{eid}"):
                for q in data["questions_for_doctor"] or ["What could be causing my symptoms?"]:
                    st.info(q, icon="❓")
        with t5:
            with st.container(key=f"warn_{eid}"):
                for w in data["warning_signs"] or ["Symptoms suddenly get worse."]:
                    st.error(w, icon="🚨")
            st.warning(cfg.EMERGENCY_TEXT, icon="📞")

        dl_box = st.container()
        with st.container(key=f"disc_results_{eid}"):
            st.warning(cfg.DISCLAIMER_LONG, icon="⚠️")

    # Stream the narrative into its reserved spot (only once per result).
    if entry["narrative"] is None:
        with narr_box:
            entry["narrative"] = st.write_stream(narrative_generator(entry))
    else:
        with narr_box:
            st.markdown(entry["narrative"])

    with dl_box:
        st.markdown("#### 📥 Download your report")
        d1, d2, d3 = st.columns(3)
        if entry["language"] in cfg.PDF_SAFE_LANGUAGES:
            try:
                d1.download_button("📄 PDF report", U.build_pdf_report(entry),
                                   f"mediguide_{eid}.pdf", "application/pdf", key=f"pdf_{eid}")
            except Exception as exc:  # noqa: BLE001
                d1.caption(f"PDF unavailable: {exc}")
        else:
            d1.caption("PDF supports Latin-script languages; use the HTML report instead.")
        d2.download_button("🌐 HTML report", U.build_html_report(entry), f"mediguide_{eid}.html",
                           "text/html", key=f"html_{eid}")
        d3.download_button("🧾 Raw JSON", json.dumps(data, indent=2, ensure_ascii=False),
                           f"mediguide_{eid}.json", "application/json", key=f"json_{eid}")


# =============================================================================
# 7. Page layout
# =============================================================================
hero(cfg.APP_NAME, cfg.APP_TAGLINE)
disclaimer_banner()

tab_assess, tab_hist, tab_stats, tab_lab, tab_about = st.tabs(
    ["🩺 Assessment", "🕘 History", "📊 Analytics", "🧪 LangChain Lab", "ℹ️ About & Safety"])

# ---------------------------------------------------------------- Assessment
with tab_assess:
    st.markdown("##### ⚡ Try a sample scenario")
    cols = st.columns(len(cfg.SAMPLE_SCENARIOS))
    for col, name in zip(cols, cfg.SAMPLE_SCENARIOS):
        col.button(name, key=f"sample_{name}", on_click=load_sample, args=(name,))

    with st.form("patient_form"):
        st.markdown("### 🧑‍⚕️ Patient information")
        left, right = st.columns(2)
        with left:
            st.text_input("Patient age (years)", key="f_age", placeholder="e.g. 25")
            st.selectbox("Gender", cfg.GENDERS, key="f_gender")
            st.selectbox("Duration of symptoms", cfg.DURATION_OPTIONS, key="f_duration")
            st.slider("Severity (1 = very mild, 10 = worst imaginable)", 1, 10, key="f_severity")
            st.selectbox("Answer language", cfg.LANGUAGES, key="f_lang")
        with right:
            st.multiselect("Symptoms", cfg.SYMPTOM_OPTIONS, key="f_symptoms",
                           placeholder="Choose one or more symptoms")
            st.text_area("Other symptoms (optional free text)", key="f_free", height=80,
                         placeholder="Describe anything not in the list...")
            st.text_area("Existing medical conditions", key="f_cond", height=80,
                         placeholder="e.g. asthma, diabetes")
        m1, m2 = st.columns(2)
        m1.text_area("Current medications", key="f_meds", height=80, placeholder="e.g. none, or names you take")
        m2.text_area("Additional notes", key="f_notes", height=80, placeholder="Anything else that may help")
        submitted = st.form_submit_button("🔍 Analyse my symptoms")

    if submitted:
        prompt_inputs = collect_prompt_inputs()
        problem = validate_form(prompt_inputs)
        if problem:
            st.warning(problem, icon="⚠️")  # no API call is made
        else:
            run_pipeline(prompt_inputs)

    current = st.session_state.current
    if current:
        render_results(current)
    else:
        st.info("Fill in the form (or pick a sample scenario) and press **Analyse my symptoms**.", icon="👆")

# ------------------------------------------------------------------- History
with tab_hist:
    hist = st.session_state.history
    st.markdown("### 🕘 Session history")
    if not hist:
        st.info("Your assessments from this session will appear here.", icon="🕘")
    else:
        h1, h2 = st.columns([3, 1])
        h2.download_button("⬇️ Export history", json.dumps(
            [{k: v for k, v in e.items() if k in ("time", "prompt_inputs", "data", "narrative")} for e in hist],
            indent=2, ensure_ascii=False), "mediguide_history.json", "application/json")
        for e in reversed(hist):
            level = e["data"]["urgency_level"] if e["data"] else "ERROR"
            icon = cfg.URGENCY_STYLE[level]["emoji"] if e["data"] else "⚠️"
            with st.expander(f"{icon} {e['time']} - {level} - {e['prompt_inputs']['symptoms'][:60]}"):
                if e["data"]:
                    st.write(e["data"]["summary"])
                st.caption(f"Model {e['model']} - severity {e['prompt_inputs']['severity']}/10 - "
                           f"{e['elapsed']:.2f}s - cache {e['cache_mode']}")
                st.button("Open in dashboard", key=f"open_{e['id']}",
                          on_click=lambda ent=e: st.session_state.update(current=ent))
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.session_state.current = None
            st.rerun()

# ----------------------------------------------------------------- Analytics
with tab_stats:
    st.markdown("### 📊 Session analytics")
    ok_entries = [e for e in st.session_state.history if e["data"]]
    if not ok_entries:
        st.info("Run at least one assessment to see analytics.", icon="📊")
    else:
        df = pd.DataFrame([{
            "time": e["time"], "urgency": e["data"]["urgency_level"],
            "severity": e["prompt_inputs"]["severity"], "elapsed": e["elapsed"],
            "symptoms": e["prompt_inputs"]["symptoms"],
        } for e in ok_entries])
        a1, a2, a3 = st.columns(3)
        a1.metric("Assessments", len(df))
        a2.metric("Average severity", f"{df['severity'].mean():.1f}/10")
        a3.metric("Average response", f"{df['elapsed'].mean():.2f}s")

        text_color = TOKENS["text"]
        colors = [cfg.URGENCY_STYLE[l]["c1"] for l in cfg.URGENCY_LEVELS]

        def style_chart(chart):
            return (chart.configure(background="rgba(0,0,0,0)")
                    .configure_view(strokeOpacity=0)
                    .configure_axis(labelColor=text_color, titleColor=text_color, gridColor="rgba(128,128,128,.25)")
                    .configure_legend(labelColor=text_color, titleColor=text_color))

        dist = df["urgency"].value_counts().reindex(cfg.URGENCY_LEVELS, fill_value=0).reset_index()
        dist.columns = ["urgency", "count"]
        bar = alt.Chart(dist).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X("urgency:N", sort=cfg.URGENCY_LEVELS, title="Urgency level"),
            y=alt.Y("count:Q", title="Assessments", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("urgency:N", scale=alt.Scale(domain=cfg.URGENCY_LEVELS, range=colors), legend=None),
            tooltip=["urgency", "count"]).properties(height=260, title="Urgency distribution")
        st.altair_chart(style_chart(bar), width="stretch", theme=None)

        g1, g2 = st.columns(2)
        sym = (", ".join(df["symptoms"]).replace(" and ", ", ").split(", "))
        sym_df = pd.Series([s.strip() for s in sym if s.strip()]).value_counts().head(8).reset_index()
        sym_df.columns = ["symptom", "count"]
        top = alt.Chart(sym_df).mark_bar(cornerRadiusEnd=8, color=TOKENS["accent2"]).encode(
            y=alt.Y("symptom:N", sort="-x", title=None), x=alt.X("count:Q", axis=alt.Axis(tickMinStep=1)),
            tooltip=["symptom", "count"]).properties(height=260, title="Most reported symptoms")
        g1.altair_chart(style_chart(top), width="stretch", theme=None)
        df["n"] = range(1, len(df) + 1)
        line = alt.Chart(df).mark_line(point=True, color=TOKENS["accent"], strokeWidth=3).encode(
            x=alt.X("n:Q", title="Assessment #", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("severity:Q", title="Severity", scale=alt.Scale(domain=[0, 10])),
            tooltip=["time", "severity", "urgency"]).properties(height=260, title="Severity over the session")
        g2.altair_chart(style_chart(line), width="stretch", theme=None)

# ------------------------------------------------------------ LangChain Lab
with tab_lab:
    st.markdown("### 🧪 LangChain Lab")
    st.caption("See every required LangChain concept in action.")

    with st.expander("1. PromptTemplate - reusable single-string template", expanded=True):
        st.code(PATIENT_PROFILE_PROMPT.template, language="text")
        st.markdown("**Rendered with your current form values:**")
        st.code(PATIENT_PROFILE_PROMPT.format(**collect_prompt_inputs()), language="text")

    with st.expander("2. ChatPromptTemplate - System + Human messages"):
        st.markdown("**System message (role + safety rules + schema):**")
        st.code(SYSTEM_PROMPT.replace("{json_schema}", JSON_SCHEMA), language="text")
        st.markdown("**Input variables:** " + ", ".join(f"`{v}`" for v in ASSESSMENT_CHAT_TEMPLATE.input_variables))
        st.markdown("**Narrative system prompt (streamed):**")
        st.code(NARRATIVE_SYSTEM, language="text")

    with st.expander("3. SystemMessage / HumanMessage / AIMessage demo"):
        st.caption("Sends a 4-message conversation: system rules, patient data, an earlier AI reply, and a follow-up.")
        if st.button("▶️ Run message demo"):
            try:
                llm = build_llm(SETTINGS["api_key"], SETTINGS["model"], SETTINGS["temperature"])
                sample = PATIENT_PROFILE_PROMPT.format(
                    age="30", gender="Female", symptoms="Sore throat, runny nose", duration="1-3 days",
                    severity=2, conditions="None reported", medications="None reported", notes="None")
                with st.spinner("Calling the model..."):
                    msgs, reply = demo_message_conversation(llm, sample)
                for m in msgs:
                    role = {"system": "⚙️ System", "human": "🧑 Human", "ai": "🤖 AI"}[m.type]
                    st.markdown(f"**{role}Message**")
                    st.code(m.content, language="text")
                st.markdown("**🤖 Model reply to the follow-up:**")
                st.success(reply, icon="✅")
            except Exception as exc:  # noqa: BLE001
                st.error(friendly_error(exc))

    with st.expander("4. Caching - InMemoryCache vs SQLiteCache"):
        st.markdown("""
| | InMemoryCache | SQLiteCache |
|---|---|---|
| Stored in | RAM | A file on disk (`.cache/langchain_cache.db`) |
| Speed | Fastest | Fast, slightly slower |
| Survives restart? | No | Yes |
| Best for | One session | Reusing results across sessions |
""")
        st.caption("Registered once with `set_llm_cache(...)`; LangChain checks it before every model call.")
        if st.button("⏱️ Run cache benchmark (4 tiny API calls)"):
            with st.spinner("Timing both caches..."):
                try:
                    rows = benchmark_cache(SETTINGS["api_key"], SETTINGS["model"])
                    cm.setup_cache(SETTINGS["cache_mode"])
                    st.table(pd.DataFrame(rows))
                    st.success("The second call of each pair is served from the cache.", icon="⚡")
                except Exception as exc:  # noqa: BLE001
                    st.error(friendly_error(exc))

# --------------------------------------------------------------- About/Safety
with tab_about:
    st.markdown("### ℹ️ How it works")
    steps = ["📝 Form", "🧩 Prompt", "⚡ Cache", "🤖 LLMChain", "🔎 Parse JSON", "🗣️ Stream", "📋 Dashboard"]
    st.markdown('<div class="chips">' + "".join(
        f'<span class="chip" style="animation-delay:{i * 0.12}s">{s}</span>' for i, s in enumerate(steps)) + '</div>',
        unsafe_allow_html=True)
    st.markdown("### 🛡️ Safety rules")
    st.markdown("""
- The app is an **educational prototype**, not a doctor, and **never gives a confirmed diagnosis**.
- A **local red-flag check** runs before the AI is called and shows emergency advice immediately.
- The system prompt forces cautious wording, no drug doses, and a **higher urgency when unsure**.
- EMERGENCY results tell you to **call your local emergency number immediately**.
- If the text suggests self-harm, the app shows supportive crisis guidance.
- Invalid or malformed model output never crashes the app.
""")
    st.markdown("### ✅ Test scenarios from the assignment")
    st.markdown("""
| # | Input | Expected |
|---|---|---|
| 1 | Age 25, runny nose + sore throat, 1-3 days, severity 2 | LOW |
| 2 | Age 40, fever + cough, 4-7 days, severity 6 | MEDIUM/HIGH |
| 3 | Severe chest pain + shortness of breath | HIGH/EMERGENCY |
| 4 | Same form twice (cache on) | 2nd run faster, same result |
| 5 | Empty symptoms | Warning, no API call |
| 6 | Language = Urdu | Guidance in Urdu |
""")