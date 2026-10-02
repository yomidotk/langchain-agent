"""FridgeSnap — LangChain Chef Agent. Step-by-step: snap → pick → cook."""
import re
import json
import html
import base64
import io
import urllib.parse
import requests
import streamlit as st
import dashscope
from http import HTTPStatus
from dashscope.audio.tts import SpeechSynthesizer
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ── Config ─────────────────────────────────────────────────────────────────────
DO_API_KEY = st.secrets.get("DO_API_KEY", "")
ALIBABA_API_KEY = st.secrets.get("ALIBABA_API_KEY", "")
DO_URL = "https://inference.do-ai.run/v1"
DO_MODEL = "openai-gpt-oss-20b"
VISION_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
VISION_MODEL = "qwen-vl-max"

CUISINES = ["Anything","Italian","Mexican","Middle Eastern","Asian","Indian","Mediterranean","French","American"]
DIETS    = ["No restriction","Vegetarian","Vegan","Halal","Gluten-free","High-protein"]

# ── Image helpers ───────────────────────────────────────────────────────────────
def get_dish_image(dish_name: str) -> str | None:
    """Try Wikipedia thumbnail, fall back to Unsplash keyword URL."""
    try:
        slug = urllib.parse.quote(dish_name.replace(" ", "_"))
        url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        r    = requests.get(url, timeout=6, headers={"User-Agent": "FridgeSnapBot/1.0"})
        if r.status_code == 200:
            data  = r.json()
            thumb = data.get("originalimage") or data.get("thumbnail")
            if thumb and thumb.get("source"):
                # Scale up for display
                src = thumb["source"]
                src = re.sub(r"/(\d+)px-", "/800px-", src)
                return src
    except Exception:
        pass
    # Unsplash random by keyword (still resolves as redirect)
    kw = urllib.parse.quote(f"{dish_name} food")
    return f"https://source.unsplash.com/800x500/?{kw}"

# ── Image prep ──────────────────────────────────────────────────────────────────
def prep_image(file_bytes, mime):
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return file_bytes, mime or "image/jpeg"

# ── Vision ──────────────────────────────────────────────────────────────────────
def detect_ingredients(img_bytes, mime):
    try:
        b64 = base64.b64encode(img_bytes).decode()
        prompt = (
            'List every visible food ingredient in this fridge/kitchen photo.\n'
            'Return ONLY valid JSON: {"ingredients":[{"name":"eggs","amount":"6","confidence":"high"}]}\n'
            'Rules: only what is visible; confidence: high/medium/low; '
            'if not a food photo return {"ingredients":[],"not_food":true}. No markdown.'
        )
        r = requests.post(
            VISION_URL,
            headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
            json={"model": VISION_MODEL,
                  "messages": [{"role": "user", "content": [
                      {"type": "text",      "text": prompt},
                      {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
                  "temperature": 0.1},
            timeout=120)
        if r.status_code != 200:
            try:   err = r.json().get("message") or r.json().get("error",{}).get("message","")
            except: err = ""
            return None, f"Vision API error {r.status_code}: {err or r.text[:200]}"
        raw = r.json()["choices"][0]["message"]["content"] or ""
        if isinstance(raw, list):
            raw = " ".join(b.get("text","") for b in raw if isinstance(b, dict))
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return None, f"Vision gave unexpected output: {clean[:200]}"
        out = json.loads(m.group(0))
        if out.get("not_food"):
            return None, "That doesn't look like a fridge photo — try another."
        # Case-insensitive key search
        items_raw = next((v for k, v in out.items() if k.lower() == "ingredients"), None)
        if not items_raw:
            return None, f"No ingredient list found in response: {clean[:200]}"
        items = [it for it in items_raw if isinstance(it, dict) and it.get("name")]
        if not items:
            return None, "Couldn't identify any ingredients from that photo."
        return items, ""
    except Exception as e:
        return None, f"Vision error: {str(e)[:300]}"

# ── LangChain models ────────────────────────────────────────────────────────────
class MissingItem(BaseModel):
    item: str  = Field(description="Missing ingredient name")
    swap: str  = Field(default="", description="A substitute from the provided list")

class Recipe(BaseModel):
    title:        str  = Field(description=(
        "A real, well-known dish name that exists in cookbooks "
        "(e.g. Shakshuka, Pad Thai, Frittata, Chicken Stir-fry). "
        "NOT an invented name."))
    character:    str  = Field(description="QUICK, HEARTY, or CREATIVE")
    time_min:     int  = Field(description="Cook time in minutes")
    calories_est: int  = Field(description="Calories per serving")
    difficulty:   str  = Field(description="Easy, Medium, or Hard")
    uses:         List[str]        = Field(description="Provided ingredients used")
    missing:      List[MissingItem] = Field(description="Up to 3 missing items")
    steps:        List[str]        = Field(description="4-8 clear cooking steps")
    tip:          str  = Field(description="One pro tip")

class RecipeList(BaseModel):
    recipes: List[Recipe] = Field(description="Exactly 3 recipes")

# ── LangChain agent ─────────────────────────────────────────────────────────────
def generate_recipes(ingredients, cuisine, diet, max_time, servings, kcal_target):
    parser = PydanticOutputParser(pydantic_object=RecipeList)
    llm = ChatOpenAI(
        model_name=DO_MODEL, openai_api_key=DO_API_KEY,
        openai_api_base=DO_URL, temperature=0.65, max_tokens=3200)
    ing       = ", ".join(ingredients)
    kcal_line = f"Each serving ≤ ~{kcal_target} kcal." if kcal_target else ""
    diet_line = f"Diet: {diet}." if diet != "No restriction" else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a culinary expert agent. A home cook has: {ingredients}.\n"
         "Basics assumed: salt, pepper, oil, water, sugar.\n"
         "Constraints: cuisine={cuisine}. {diet_line} ≤{max_time} min. {servings} servings. {kcal_line}\n\n"
         "Return 3 DISTINCT real dishes: 1) QUICK (≤20 min), 2) HEARTY (filling), 3) CREATIVE (surprising).\n"
         "title MUST be a real dish name (e.g. 'Shakshuka', 'Frittata'). No invented names.\n"
         "uses[] must only include items from the ingredient list above.\n\n"
         "{format_instructions}"),
        ("user", "Generate 3 real recipes now.")
    ])
    result = (prompt | llm | parser).invoke({
        "ingredients": ing, "cuisine": cuisine, "diet_line": diet_line,
        "max_time": max_time, "servings": servings, "kcal_line": kcal_line,
        "format_instructions": parser.get_format_instructions()
    })
    return [r.model_dump() for r in result.recipes]

# ── Audio TTS ───────────────────────────────────────────────────────────────────
def generate_audio(recipe_idx: int, title: str, steps: list):
    dashscope.api_key = ALIBABA_API_KEY
    text = f"Recipe: {title}. " + " ".join(f"Step {i+1}. {s}" for i, s in enumerate(steps))
    result = SpeechSynthesizer.call(
        model="sambert-zhichu-v1", text=text[:500], format="mp3")
    if result.status_code == HTTPStatus.OK:
        audio_data = result.get_audio_data()
        if audio_data:
            st.session_state[f"audio_{recipe_idx}"] = audio_data
            return
    code = getattr(result, "code", "unknown")
    msg  = getattr(result, "message", str(result))
    st.error(f"Audio failed ({code}): {msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,600;0,700;1,500;1,600&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:      #FDFAF5;
  --surface: #F5EFE4;
  --card:    #FFFFFF;
  --border:  #E8DFD0;
  --border2: #D5C9B5;
  --accent:  #C4541E;
  --accent2: #E87840;
  --muted-accent: #F0E4D8;
  --text:    #1C1714;
  --text2:   #3D3228;
  --text3:   #7A6D5E;
  --text4:   #B0A090;
  --green:   #2E7A4A;
  --green-bg:#EDF5F0;
}
*,*::before,*::after { box-sizing:border-box; }
* { font-family:'Inter',system-ui,sans-serif!important; }
html { scroll-behavior:smooth; }
.stApp { background:var(--bg)!important; }
#MainMenu,footer,header[data-testid="stHeader"] { display:none!important; }
.block-container { max-width:860px!important; padding:0 clamp(16px,4vw,40px) 120px!important; margin:0 auto!important; }
section[data-testid="stSidebar"] { display:none!important; }

/* ── NAV ── */
.nav { display:flex; align-items:center; gap:12px; padding:26px 0 10px; border-bottom:1px solid var(--border); margin-bottom:36px; }
.logo-icon { width:36px;height:36px;border-radius:10px;background:var(--accent);display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 3px 10px rgba(196,84,30,.25); }
.brand-name { font-family:'Lora',serif!important; font-weight:700; font-size:21px; letter-spacing:-.02em; color:var(--text); }
.brand-name span { color:var(--accent); }
.agent-tag { font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--text3);background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:4px 10px; }

/* ── PROGRESS ── */
.progress { display:flex; align-items:center; gap:0; margin-bottom:40px; }
.prog-step { display:flex; align-items:center; gap:8px; flex:1; }
.prog-num { width:28px;height:28px;border-radius:50%;border:2px solid var(--border2);background:var(--bg);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--text3);flex-shrink:0;transition:all .3s; }
.prog-label { font-size:12px;font-weight:600;color:var(--text3); }
.prog-line { flex:1;height:1px;background:var(--border);margin:0 8px; }
.prog-step.active .prog-num { background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:0 2px 8px rgba(196,84,30,.3); }
.prog-step.active .prog-label { color:var(--accent);font-weight:700; }
.prog-step.done .prog-num { background:var(--green);border-color:var(--green);color:#fff; }
.prog-step.done .prog-label { color:var(--green); }

/* ── HERO ── */
.hero { margin-bottom:36px; }
.hero-eyebrow { font-size:11px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:12px; }
.hero h1 { font-family:'Lora',serif!important; font-size:clamp(2.2rem,6vw,3.4rem); font-weight:700; line-height:1.1; letter-spacing:-.02em; color:var(--text); margin:0 0 14px; }
.hero h1 em { font-style:italic; color:var(--accent); }
.hero p { font-size:15.5px; color:var(--text3); line-height:1.75; max-width:500px; }

/* ── CARDS ── */
.card { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:28px 32px; margin-bottom:16px; box-shadow:0 2px 16px rgba(0,0,0,.04); }
.card-kicker { font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:8px; }
.card-title { font-family:'Lora',serif!important; font-size:18px;font-weight:700;color:var(--text);margin-bottom:6px; }
.card-sub { font-size:14px;color:var(--text3);line-height:1.6; }

/* ── BUTTONS ── */
div[data-testid="stButton"] button { background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:14px 24px!important;font-size:14px!important;font-weight:700!important;letter-spacing:-.01em!important;box-shadow:0 3px 12px rgba(196,84,30,.22)!important;transition:all .15s!important; }
div[data-testid="stButton"] button p { color:#fff!important; }
div[data-testid="stButton"] button:hover { filter:brightness(1.08)!important;transform:translateY(-1px)!important;box-shadow:0 5px 18px rgba(196,84,30,.3)!important; }
div[data-testid="stButton"] button:disabled { opacity:.35!important;transform:none!important;box-shadow:none!important; }
div[data-testid="stButton"] button[kind="secondary"] { background:var(--surface)!important;color:var(--text2)!important;border:1px solid var(--border2)!important;box-shadow:none!important; }
div[data-testid="stButton"] button[kind="secondary"] p { color:var(--text2)!important; }

/* ── INPUTS ── */
div[data-testid="stTextInput"] input { background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:12px!important;padding:13px 18px!important;font-size:14px!important;color:var(--text)!important; }
div[data-testid="stTextInput"] input:focus { border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(196,84,30,.1)!important; }
div[data-testid="stTextArea"] textarea { background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:12px!important;color:var(--text)!important; }

/* ── TABS ── */
div[data-testid="stTabs"] [data-baseweb="tab-list"] { background:var(--surface)!important;border-radius:12px!important;padding:4px!important;border:1px solid var(--border)!important; }
div[data-testid="stTabs"] [data-baseweb="tab"] { border-radius:9px!important;font-weight:600!important;color:var(--text3)!important;border:none!important;font-size:13.5px!important;padding:9px 18px!important;background:transparent!important; }
div[data-testid="stTabs"] [aria-selected="true"] { background:var(--card)!important;color:var(--text)!important;box-shadow:0 1px 4px rgba(0,0,0,.1)!important; }

/* ── PILLS ── */
div[data-testid="stPills"] button { background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:999px!important;color:var(--text2)!important;font-weight:600!important;font-size:13px!important; }
div[data-testid="stPills"] button[aria-pressed="true"] { background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important; }

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] { border:2px dashed var(--border2)!important;border-radius:16px!important;background:var(--surface)!important; }
div[data-testid="stFileUploader"]:hover { border-color:var(--accent)!important; }

/* ── PHOTO ── */
.photo { border-radius:16px;overflow:hidden;border:1px solid var(--border);margin:14px 0; }
.photo img { width:100%;display:block; }

/* ── EXPANDER ── */
div[data-testid="stExpander"] { background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:14px!important;margin-top:12px; }

/* ── RECIPE SELECTION CARDS ── */
.recipe-pick { background:var(--card);border:2px solid var(--border);border-radius:20px;overflow:hidden;transition:border-color .2s,box-shadow .2s;cursor:pointer;margin-bottom:14px; }
.recipe-pick:hover { border-color:var(--accent2);box-shadow:0 4px 24px rgba(196,84,30,.1); }
.recipe-pick-img { width:100%;height:200px;object-fit:cover;display:block;background:var(--surface); }
.recipe-pick-body { padding:20px 22px 18px; }
.recipe-pick-char { font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:8px; }
.recipe-pick-title { font-family:'Lora',serif!important; font-size:18px;font-weight:700;color:var(--text);margin-bottom:10px;line-height:1.25; }
.recipe-pick-meta { display:flex;flex-wrap:wrap;gap:7px; }
.meta-chip { font-size:12px;font-weight:600;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:5px 11px;color:var(--text2); }

/* ── RECIPE DETAIL ── */
.recipe-hero-img { width:100%;height:280px;object-fit:cover;border-radius:20px 20px 0 0;display:block; }
.recipe-detail { background:var(--card);border:1px solid var(--border);border-radius:20px;overflow:hidden;margin-bottom:16px;box-shadow:0 4px 24px rgba(0,0,0,.05); }
.recipe-detail-body { padding:28px 32px; }
.recipe-char-tag { display:inline-flex;align-items:center;gap:6px;font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);background:var(--muted-accent);border-radius:6px;padding:5px 12px;margin-bottom:14px; }
.recipe-detail-title { font-family:'Lora',serif!important; font-size:clamp(1.6rem,5vw,2.2rem);font-weight:700;color:var(--text);letter-spacing:-.02em;margin:0 0 14px;line-height:1.1; }
.recipe-meta-row { display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px; }
.recipe-meta-chip { font-size:12.5px;font-weight:600;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:7px 14px;color:var(--text2); }
.recipe-meta-chip.kcal { color:var(--accent);border-color:rgba(196,84,30,.2);background:var(--muted-accent); }

.ingredient-section { margin:20px 0; }
.ing-label { font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--text3);margin-bottom:10px; }
.ing-grid { display:flex;flex-wrap:wrap;gap:7px; }
.ing-have { font-size:13px;font-weight:600;background:var(--green-bg);border:1px solid rgba(46,122,74,.2);color:var(--green);border-radius:9px;padding:6px 13px; }
.ing-need { font-size:13px;font-weight:600;background:var(--muted-accent);border:1px solid rgba(196,84,30,.2);color:var(--accent);border-radius:9px;padding:6px 13px; }

.divider { height:1px;background:var(--border);margin:22px 0; }

.steps-label { font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--text3);margin-bottom:16px; }
.step-row { display:flex;gap:16px;margin-bottom:16px; }
.step-num { width:32px;height:32px;border-radius:10px;background:var(--surface);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:var(--accent);flex-shrink:0; }
.step-text { font-size:15px;line-height:1.72;color:var(--text2);padding-top:4px; }

.tip-block { background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:15px 20px;font-size:13.5px;color:var(--text3);line-height:1.65;margin-top:20px; }
.tip-block b { color:var(--accent); }

/* ── AUDIO ── */
div[data-testid="stAudio"] { margin-top:16px;border-radius:12px;overflow:hidden;border:1px solid var(--border); }

/* ── SHOP ── */
.shop { background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:28px;margin-top:24px; }
.shop-title { font-family:'Lora',serif!important; font-size:19px;font-weight:700;color:var(--text);margin-bottom:5px; }
.shop-sub { font-size:13.5px;color:var(--text3);margin-bottom:16px; }
.shop-row { display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--border); }
.shop-row:last-child { border-bottom:none; }
.shop-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);flex-shrink:0; }
.shop-name { font-size:14px;font-weight:600;color:var(--text2); }
.shop-swap { font-size:12.5px;color:var(--text4);margin-left:auto; }

/* ── MISC STREAMLIT ── */
.stAlert { border-radius:12px!important; }
div[data-testid="stSlider"] { accent-color:var(--accent); }
.footer { text-align:center;color:var(--text4);font-size:12.5px;margin-top:80px;padding:24px 0;border-top:1px solid var(--border);line-height:1.7; }

@media(max-width:640px){
  .card,.recipe-detail-body{padding:20px;}
  .hero h1{font-size:2.1rem;}
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# APP START
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FridgeSnap", layout="centered", page_icon="🍳")
st.markdown(_CSS, unsafe_allow_html=True)

# NAV
st.markdown("""
<div class="nav">
  <div class="logo-icon">🍳</div>
  <div class="brand-name">Fridge<span>Snap</span></div>
  <div class="agent-tag">LangChain · AI Chef</div>
</div>""", unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("⚠️ Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in Streamlit Secrets.")
    st.stop()

# ── Session state ───────────────────────────────────────────────────────────────
defaults = {"stage": 1, "photo": None, "detected": None, "confirmed": [], "recipes": None, "picked": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

stage = st.session_state.stage

# ── Progress bar ────────────────────────────────────────────────────────────────
def _ps(i, label):
    cls = "active" if i == stage else ("done" if i < stage else "")
    num = "✓" if i < stage else str(i)
    line = '<div class="prog-line"></div>' if i < 4 else ""
    return f'<div class="prog-step {cls}"><div class="prog-num">{num}</div><div class="prog-label">{label}</div></div>{line}'

st.markdown(f"""
<div class="progress">
  {_ps(1,"Photo")}
  {_ps(2,"Ingredients")}
  {_ps(3,"Pick Recipe")}
  {_ps(4,"Cook!")}
</div>""", unsafe_allow_html=True)

def sync_pills():
    st.session_state.confirmed = [lbl.split(" (")[0] for lbl in st.session_state.get("pill_sel", [])]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Photo
# ══════════════════════════════════════════════════════════════════════════════
if stage == 1:
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Step 1 of 4</div>
  <h1>Snap your fridge.<br><em>Cook what's inside.</em></h1>
  <p>Upload a photo — the AI spots every ingredient, then builds three real recipes just for you.</p>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-kicker">Add photo</div><div class="card-title">Show me your fridge 📸</div><div class="card-sub">Clear, bright photos from the front work best.</div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs(["📤 Upload photo", "📷 Take photo"])
    with tab_up:
        up = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if up:
            st.session_state.photo = (up.getvalue(), up.type)
    with tab_cam:
        cam = st.camera_input("Snap", label_visibility="collapsed")
        if cam:
            st.session_state.photo = (cam.getvalue(), cam.type)

    if st.session_state.photo:
        raw, mime = st.session_state.photo
        st.markdown(f'<div class="photo"><img src="data:{mime};base64,{base64.b64encode(raw).decode()}"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("🔍 Scan with AI →", use_container_width=True):
                with st.spinner("Vision model scanning your fridge…"):
                    small, mime2 = prep_image(raw, mime)
                    items, err = detect_ingredients(small, mime2)
                if err:
                    st.error(err)
                else:
                    st.session_state.detected  = items
                    st.session_state.confirmed = [it["name"] for it in items if it.get("confidence") != "low"]
                    st.session_state.stage     = 2
                    st.rerun()
        with c2:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.photo = None
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⌨️ No photo? Type your ingredients"):
        manual = st.text_area("One per line or comma-separated", placeholder="eggs, chicken, tomatoes, cheese…", label_visibility="collapsed", height=100)
        if st.button("Use these →", use_container_width=True):
            names = [x.strip() for x in re.split(r"[,\n]+", manual or "") if x.strip()]
            if not names:
                st.warning("Type at least one ingredient.")
            else:
                seen, items = set(), []
                for n in names:
                    if n.lower() not in seen:
                        seen.add(n.lower())
                        items.append({"name": n, "amount": "", "confidence": "high"})
                st.session_state.detected  = items
                st.session_state.confirmed = [it["name"] for it in items]
                st.session_state.stage     = 2
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Ingredients + Preferences
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 2:
    st.markdown('<div class="hero"><div class="hero-eyebrow">Step 2 of 4</div><h1>Confirm your <em>ingredients.</em></h1><p>Toggle anything wrong off, then set your preferences before we cook.</p></div>', unsafe_allow_html=True)

    # Ingredients
    st.markdown('<div class="card"><div class="card-kicker">Detected</div><div class="card-title">&#x1F96C; What\'s in your fridge</div><div class="card-sub">Tap to deselect. Green = confident, orange = unsure.</div>', unsafe_allow_html=True)
    pill_labels   = [f"{it['name']} ({it['amount']})" if it.get("amount") else it["name"] for it in st.session_state.detected]
    default_pills = [lbl for lbl, it in zip(pill_labels, st.session_state.detected) if it["name"] in st.session_state.confirmed]
    st.pills("Ingredients", pill_labels, selection_mode="multi", default=default_pills, key="pill_sel", on_change=sync_pills, label_visibility="collapsed")
    extra = st.text_input("➕ Add anything it missed", placeholder="e.g. rice, soy sauce", label_visibility="collapsed")
    if extra.strip():
        for x in re.split(r"[,\n]+", extra):
            x = x.strip()
            if x and x.lower() not in {c.lower() for c in st.session_state.confirmed}:
                st.session_state.confirmed.append(x)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preferences
    st.markdown('<div class="card"><div class="card-kicker">Preferences</div><div class="card-title">⚙️ Dial it in</div><div class="card-sub">Customise before the agent generates your recipes.</div>', unsafe_allow_html=True)
    cuisine  = st.pills("Cuisine", CUISINES, default="Anything", label_visibility="collapsed") or "Anything"
    st.markdown("---")
    diet     = st.pills("Diet", DIETS, default="No restriction", label_visibility="collapsed") or "No restriction"
    st.markdown("---")
    c1, c2   = st.columns(2)
    with c1:
        st.markdown("**Max cook time**")
        max_time = st.slider("Max time", 10, 90, 30, step=5, format="%d min", label_visibility="collapsed")
    with c2:
        st.markdown("**Servings**")
        servings = st.pills("Servings", [1,2,3,4,5,6], default=2, label_visibility="collapsed") or 2
    use_kcal    = st.checkbox("🎯 Set a calorie target")
    kcal_target = st.slider("kcal", 200, 1200, 500, step=50, format="%d kcal", label_visibility="collapsed") if use_kcal else 0
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Back", type="secondary", use_container_width=True):
            st.session_state.stage = 1
            st.rerun()
    with c2:
        if st.button("🍳 Generate 3 recipes →", use_container_width=True, disabled=not st.session_state.confirmed):
            with st.spinner("LangChain agent is thinking…"):
                try:
                    recs = generate_recipes(st.session_state.confirmed, cuisine, diet, max_time, servings, kcal_target)
                    st.session_state.recipes = recs
                    # Prefetch images
                    for i, r in enumerate(recs):
                        st.session_state[f"img_{i}"] = get_dish_image(r["title"])
                    st.session_state.stage = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"Recipe error: {html.escape(str(e))[:300]}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Pick a Recipe
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown('<div class="hero"><div class="hero-eyebrow">Step 3 of 4</div><h1>Pick your <em>recipe.</em></h1><p>Three real dishes built around your fridge. Tap one to get the full recipe.</p></div>', unsafe_allow_html=True)

    char_icons = {"QUICK": "⚡", "HEARTY": "💪", "CREATIVE": "✨"}
    for idx, r in enumerate(st.session_state.recipes):
        char = r.get("character", "")
        icon = char_icons.get(char, "🍽")
        img  = st.session_state.get(f"img_{idx}")

        img_html = f'<img class="recipe-pick-img" src="{img}" onerror="this.style.display=\'none\'">' if img else '<div style="width:100%;height:140px;background:var(--surface);"></div>'
        st.markdown(f"""
<div class="recipe-pick">
  {img_html}
  <div class="recipe-pick-body">
    <div class="recipe-pick-char">{icon} {html.escape(char)}</div>
    <div class="recipe-pick-title">{html.escape(r.get('title',''))}</div>
    <div class="recipe-pick-meta">
      <div class="meta-chip">⏱ {r.get('time_min','?')} min</div>
      <div class="meta-chip">🔥 {r.get('calories_est','?')} kcal</div>
      <div class="meta-chip">👨‍🍳 {html.escape(str(r.get('difficulty','')))}</div>
      <div class="meta-chip">Uses {len(r.get('uses',[]))}/{max(len(st.session_state.confirmed),1)} ingredients</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        if st.button(f"Cook this → {r.get('title','')[:30]}", key=f"pick_{idx}", use_container_width=True):
            st.session_state.picked = idx
            st.session_state.stage  = 4
            st.rerun()

    if st.button("← Back to preferences", type="secondary"):
        st.session_state.stage = 2
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Full Recipe
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 4:
    r    = st.session_state.recipes[st.session_state.picked]
    img  = st.session_state.get(f"img_{st.session_state.picked}")
    char = r.get("character", "")
    icon = char_icons = {"QUICK": "⚡", "HEARTY": "💪", "CREATIVE": "✨"}.get(char, "🍽")
    uses   = r.get("uses", [])
    miss   = r.get("missing", [])
    steps  = r.get("steps", [])

    st.markdown('<div class="hero"><div class="hero-eyebrow">Step 4 of 4</div><h1>Time to <em>cook.</em></h1><p>Follow the steps below — and hit the audio button to hear the recipe hands-free.</p></div>', unsafe_allow_html=True)

    # Big image + card
    st.markdown('<div class="recipe-detail">', unsafe_allow_html=True)
    if img:
        st.markdown(f'<img class="recipe-hero-img" src="{img}" onerror="this.style.display=\'none\'">', unsafe_allow_html=True)
    st.markdown(f"""<div class="recipe-detail-body">
  <div class="recipe-char-tag">{icon} {html.escape(char)}</div>
  <div class="recipe-detail-title">{html.escape(r.get('title',''))}</div>
  <div class="recipe-meta-row">
    <div class="recipe-meta-chip">⏱ {r.get('time_min','?')} min</div>
    <div class="recipe-meta-chip kcal">🔥 {r.get('calories_est','?')} kcal / serving</div>
    <div class="recipe-meta-chip">👨‍🍳 {html.escape(str(r.get('difficulty','')))}</div>
  </div>""", unsafe_allow_html=True)

    # Ingredients
    if uses or miss:
        have_html = "".join(f'<span class="ing-have">✓ {html.escape(u)}</span>' for u in uses)
        need_html = "".join(f'<span class="ing-need">+ {html.escape(m.get("item","") if isinstance(m,dict) else str(m))}</span>' for m in miss)
        st.markdown(f"""
<div class="ingredient-section">
  <div class="ing-label">From your fridge</div>
  <div class="ing-grid">{have_html}</div>
  {f'<div class="ing-label" style="margin-top:12px">You\'ll also need</div><div class="ing-grid">{need_html}</div>' if need_html else ""}
</div>
<div class="divider"></div>""", unsafe_allow_html=True)

    # Steps
    if steps:
        steps_html = "".join(f'<div class="step-row"><div class="step-num">{i+1}</div><div class="step-text">{html.escape(str(s))}</div></div>' for i, s in enumerate(steps))
        st.markdown(f'<div class="steps-label">Instructions</div>{steps_html}', unsafe_allow_html=True)

    if r.get("tip"):
        st.markdown(f'<div class="tip-block"><b>💡 Pro tip:</b> {html.escape(str(r["tip"]))}</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Audio
    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("🔊 Hear this recipe aloud", use_container_width=True):
            with st.spinner("Generating voice…"):
                generate_audio(st.session_state.picked, r.get("title",""), steps)
    with c2:
        if st.button("← Choose another", type="secondary", use_container_width=True):
            st.session_state.stage = 3
            st.rerun()

    if f"audio_{st.session_state.picked}" in st.session_state:
        st.audio(st.session_state[f"audio_{st.session_state.picked}"], format="audio/mp3")

    # Shopping list
    all_missing = {m["item"]: m.get("swap","") for m in miss if isinstance(m, dict) and m.get("item")}
    if all_missing:
        rows = "".join(f'<div class="shop-row"><div class="shop-dot"></div><div class="shop-name">{html.escape(k)}</div>' + (f'<div class="shop-swap">or: {html.escape(v)}</div>' if v else "") + "</div>" for k, v in all_missing.items())
        st.markdown(f'<div class="shop"><div class="shop-title">🛒 Shopping List</div><div class="shop-sub">What you\'ll need to grab.</div>{rows}</div>', unsafe_allow_html=True)

    if st.button("🔄 Start over", type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.markdown('<div class="footer">FridgeSnap · Powered by LangChain + Alibaba Vision + DashScope TTS<br>Calorie estimates are rough guides only.</div>', unsafe_allow_html=True)
