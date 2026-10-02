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

CUISINES = ["Anything", "Italian", "Mexican", "Asian", "Indian", "Mediterranean", "French", "American"]
DIETS    = ["No restriction", "Vegetarian", "Vegan", "Halal", "Gluten-free", "High-protein"]

# ── Image helpers ───────────────────────────────────────────────────────────────
def get_dish_image(dish_name: str) -> str | None:
    """Fetch high-res food photo using Wikipedia API with fallback to Foodish."""
    s = requests.Session()
    s.headers.update({"User-Agent": "FridgeSnapBot/1.0 (contact@fridgesnap.org)"})

    # 1. Try exact Wikipedia page summary
    try:
        slug = urllib.parse.quote(dish_name.replace(" ", "_"))
        r = s.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}", timeout=4)
        if r.status_code == 200:
            d = r.json()
            thumb = d.get("thumbnail", {}).get("source") or d.get("originalimage", {}).get("source")
            if thumb:
                # Scale up to crisp resolution
                return re.sub(r"/(\d+)px-", "/640px-", thumb)
    except Exception:
        pass

    # 2. Try Wikipedia generator search for dish + food
    try:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch="
            + urllib.parse.quote(dish_name + " food dish")
            + "&gsrlimit=1&prop=pageimages&pithumbsize=640&format=json"
        )
        r2 = s.get(url, timeout=4)
        if r2.status_code == 200:
            pages = r2.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "thumbnail" in page:
                    return page["thumbnail"]["source"]
    except Exception:
        pass

    # 3. Fallback to Foodish API
    try:
        r3 = s.get("https://foodish-api.com/api", timeout=3)
        if r3.status_code == 200:
            img = r3.json().get("image")
            if img:
                return img
    except Exception:
        pass

    # 4. Reliable static fallback food image
    return "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop&q=80"

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
    item: str = Field(description="Missing ingredient name")
    swap: str = Field(default="", description="A substitute from the provided list")

class Recipe(BaseModel):
    title:        str              = Field(description="A real, well-known dish name (e.g. Shakshuka, Pad Thai, Frittata). NOT an invented name.")
    character:    str              = Field(description="QUICK, HEARTY, or CREATIVE")
    time_min:     int              = Field(description="Cook time in minutes")
    calories_est: int              = Field(description="Calories per serving")
    difficulty:   str              = Field(description="Easy, Medium, or Hard")
    uses:         List[str]        = Field(description="Provided ingredients used")
    missing:      List[MissingItem] = Field(description="Up to 3 missing items")
    steps:        List[str]        = Field(description="4-8 clear cooking steps")
    tip:          str              = Field(description="One pro tip")

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
    text = f"Recipe for {title}. " + " ".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    
    # Singapore/International endpoint first
    endpoints = [
        ("https://dashscope-intl.aliyuncs.com/api/v1", "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"),
        ("https://dashscope.aliyuncs.com/api/v1", "wss://dashscope.aliyuncs.com/api-ws/v1/inference")
    ]
    models = ["cosyvoice-v1", "sambert-zhichu-v1"]
    
    last_err = ""
    for http_url, ws_url in endpoints:
        dashscope.base_http_api_url = http_url
        dashscope.base_websocket_api_url = ws_url
        for m in models:
            try:
                res = SpeechSynthesizer.call(model=m, text=text[:500], format="mp3")
                if res.status_code == HTTPStatus.OK:
                    data = res.get_audio_data()
                    if data:
                        st.session_state[f"audio_{recipe_idx}"] = data
                        return
                last_err = getattr(res, "message", str(res))
            except Exception as e:
                last_err = str(e)
                
    st.error(f"Voice generation: {last_err or 'Could not connect to TTS service.'}")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Compact, High-Contrast, No-Scroll Optimized
# ═══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

:root {
  --bg:           #FBF8F3;
  --surface:      #F4EDE2;
  --surface-hover:#EFE6D7;
  --card:         #FFFFFF;
  --border:       #E4D9C8;
  --border-focus: #C4541E;
  --accent:       #C4541E;
  --accent-light: #FDF0E9;
  --accent-hover: #AF4312;
  --text:         #1C1714;
  --text-muted:   #4A3F35;
  --text-dim:     #7C6E60;
  --green:        #1E6F3D;
  --green-bg:     #EBF6EE;
  --shadow-sm:    0 1px 3px rgba(35,25,15,0.06);
  --shadow-md:    0 4px 14px rgba(35,25,15,0.08);
}

*, *::before, *::after { box-sizing: border-box; }

body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }

/* Constrain width cleanly without empty space */
.block-container {
  max-width: 1040px !important;
  padding: 10px 20px 24px !important;
  margin: 0 auto !important;
}

/* ── Typography & Global Contrast ── */
h1, h2, h3, h4, h5, h6 {
  color: var(--text) !important;
  font-weight: 700 !important;
}
p, span, div, label {
  color: var(--text) !important;
}
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  margin-bottom: 2px !important;
}

/* ── NAV HEADER ── */
.top-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0 10px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
}
.brand-group {
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-icon {
  width: 32px;
  height: 32px;
  background: var(--accent);
  color: #fff;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 2px 6px rgba(196,84,30,0.25);
}
.brand-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 20px;
  font-weight: 700;
  color: var(--text) !important;
  letter-spacing: -0.01em;
}
.brand-title span { color: var(--accent); }
.badge-tag {
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: var(--accent-light);
  color: var(--accent) !important;
  border: 1px solid rgba(196,84,30,0.25);
  border-radius: 999px;
  padding: 3px 10px;
}

/* ── PROGRESS BAR ── */
.stepper-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 14px;
  background: var(--surface);
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid var(--border);
}
.step-item {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  padding: 4px 8px;
  border-radius: 7px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-dim) !important;
}
.step-item.active {
  background: var(--card);
  color: var(--accent) !important;
  box-shadow: var(--shadow-sm);
  font-weight: 700;
}
.step-item.done {
  color: var(--green) !important;
}
.step-circle {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  background: var(--border);
  color: var(--text-dim);
}
.step-item.active .step-circle {
  background: var(--accent);
  color: #fff;
}
.step-item.done .step-circle {
  background: var(--green);
  color: #fff;
}

/* ── HERO BANNER (SUPER COMPACT) ── */
.stage-hero {
  margin-bottom: 12px;
}
.stage-hero h2 {
  font-family: 'Playfair Display', serif !important;
  font-size: 22px;
  font-weight: 700;
  margin: 0 0 2px !important;
  line-height: 1.2;
}
.stage-hero p {
  font-size: 13px;
  color: var(--text-dim) !important;
  margin: 0 !important;
}

/* ── STREAMLIT CONTAINERS (NATIVE CARDS) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--border) !important;
  border-radius: 12px !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 4px !important;
}

/* ── FILE UPLOADER COMPACT & STYLED ── */
div[data-testid="stFileUploader"] {
  margin-bottom: 0 !important;
}
div[data-testid="stFileUploader"] section {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
  padding: 14px 10px !important;
  transition: all 0.2s ease;
}
div[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
  background: var(--accent-light) !important;
}
div[data-testid="stFileUploaderDropzone"] p,
div[data-testid="stFileUploaderDropzone"] span {
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
div[data-testid="stFileUploaderDropzone"] small {
  color: var(--text-dim) !important;
  font-size: 11px !important;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  padding: 8px 16px !important;
  box-shadow: 0 2px 6px rgba(196,84,30,0.2) !important;
  transition: transform 0.1s ease, filter 0.15s ease !important;
}
div[data-testid="stButton"] button p {
  color: #ffffff !important;
  font-weight: 700 !important;
}
div[data-testid="stButton"] button:hover {
  filter: brightness(1.08) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
div[data-testid="stButton"] button[kind="secondary"] p {
  color: var(--text) !important;
}

/* ── PILLS ── */
div[data-testid="stPills"] button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  color: var(--text) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 4px 12px !important;
}
div[data-testid="stPills"] button p {
  color: var(--text) !important;
}
div[data-testid="stPills"] button[aria-pressed="true"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #ffffff !important;
}
div[data-testid="stPills"] button[aria-pressed="true"] p {
  color: #ffffff !important;
}

/* ── INPUTS & SLIDERS ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-size: 13px !important;
  padding: 8px 12px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent) !important;
  background: #ffffff !important;
}
div[data-testid="stSlider"] div,
div[data-testid="stSlider"] span,
div[data-testid="stSlider"] p {
  color: var(--text) !important;
  font-weight: 600 !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 8px !important;
  padding: 3px !important;
  gap: 4px !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--text-dim) !important;
  padding: 6px 14px !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--card) !important;
  color: var(--text) !important;
  font-weight: 700 !important;
  box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stTabs"] [aria-selected="true"] p {
  color: var(--text) !important;
}

/* ── RECIPE CARDS (STAGE 3) ── */
.recipe-card-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  height: 100%;
}
.recipe-card-box:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--accent);
}
.recipe-badge-quick {
  background: #FEF3C7;
  color: #B45309 !important;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 5px;
  display: inline-block;
  margin-bottom: 6px;
}
.recipe-badge-hearty {
  background: #E0E7FF;
  color: #3730A3 !important;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 5px;
  display: inline-block;
  margin-bottom: 6px;
}
.recipe-badge-creative {
  background: #FCE7F3;
  color: #9D174D !important;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 5px;
  display: inline-block;
  margin-bottom: 6px;
}
.recipe-meta-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 8px 0;
}
.recipe-tag {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 2px 7px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted) !important;
}

/* ── STAGE 4 DETAIL VIEW ── */
.detail-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin-bottom: 12px;
}
.detail-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 26px;
  font-weight: 700;
  color: var(--text) !important;
  margin: 0 0 6px !important;
}
.detail-meta-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.step-item-card {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  margin-bottom: 10px;
  background: var(--card);
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
}
.step-num-badge {
  background: var(--accent);
  color: #ffffff !important;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
}
.step-content {
  font-size: 13px;
  line-height: 1.5;
  color: var(--text) !important;
}
.pro-tip-box {
  background: var(--accent-light);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 12.5px;
  color: var(--text) !important;
  margin-top: 12px;
}
.ing-tag-have {
  background: var(--green-bg);
  border: 1px solid rgba(30,111,61,0.25);
  color: var(--green) !important;
  border-radius: 6px;
  padding: 3px 8px;
  font-size: 11.5px;
  font-weight: 600;
  display: inline-block;
  margin: 2px;
}
.ing-tag-need {
  background: #FFF1F0;
  border: 1px solid rgba(207,19,34,0.25);
  color: #CF1322 !important;
  border-radius: 6px;
  padding: 3px 8px;
  font-size: 11.5px;
  font-weight: 600;
  display: inline-block;
  margin: 2px;
}

/* ── FOOTER ── */
.app-footer {
  text-align: center;
  font-size: 11px;
  color: var(--text-dim);
  border-top: 1px solid var(--border);
  padding: 10px 0 4px;
  margin-top: 20px;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# APP START
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FridgeSnap", layout="wide", page_icon="🍳")
st.markdown(_CSS, unsafe_allow_html=True)

# NAV
st.markdown("""
<div class="top-nav">
  <div class="brand-group">
    <div class="brand-icon">🍳</div>
    <div class="brand-title">Fridge<span>Snap</span></div>
  </div>
  <div class="badge-tag">LangChain · AI Chef</div>
</div>""", unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("⚠️ Missing API keys — configure `DO_API_KEY` and `ALIBABA_API_KEY` in Streamlit Secrets.")
    st.stop()

# Session State
defaults = {"stage": 1, "photo": None, "detected": None, "confirmed": [], "recipes": None, "picked": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

stage = st.session_state.stage

# Progress bar
def render_stepper(cur):
    steps = [(1, "Snap Fridge"), (2, "Ingredients & Style"), (3, "Pick Recipe"), (4, "Cook!")]
    items = []
    for num, lbl in steps:
        if num < cur:
            cls = "done"
            icon = "✓"
        elif num == cur:
            cls = "active"
            icon = str(num)
        else:
            cls = ""
            icon = str(num)
        items.append(f'<div class="step-item {cls}"><div class="step-circle">{icon}</div><span>{lbl}</span></div>')
    return f'<div class="stepper-bar">{"".join(items)}</div>'

st.markdown(render_stepper(stage), unsafe_allow_html=True)

def sync_pills():
    st.session_state.confirmed = [lbl.split(" (")[0] for lbl in st.session_state.get("pill_sel", [])]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Snap or Input
# ══════════════════════════════════════════════════════════════════════════════
if stage == 1:
    st.markdown("""
    <div class="stage-hero">
      <h2>Snap your fridge, cook what's inside.</h2>
      <p>Show the agent what you have — it detects ingredients, tailors to your diet, and generates 3 custom recipes.</p>
    </div>""", unsafe_allow_html=True)

    col_photo, col_manual = st.columns([1.1, 0.9], gap="medium")

    with col_photo:
        with st.container(border=True):
            st.markdown("**📸 Option A: Add a Fridge Photo**")
            tab_up, tab_cam = st.tabs(["📤 Upload Image", "📷 Take Snapshot"])
            with tab_up:
                up = st.file_uploader("Upload fridge photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
                if up:
                    st.session_state.photo = (up.getvalue(), up.type)
            with tab_cam:
                cam = st.camera_input("Take photo", label_visibility="collapsed")
                if cam:
                    st.session_state.photo = (cam.getvalue(), cam.type)

            if st.session_state.photo:
                raw, mime = st.session_state.photo
                c_p1, c_p2 = st.columns([1, 1])
                with c_p1:
                    st.image(raw, width=180, caption="Selected photo")
                with c_p2:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    if st.button("🔍 Scan with AI →", use_container_width=True):
                        with st.spinner("Alibaba Vision scanning your ingredients…"):
                            small, mime2 = prep_image(raw, mime)
                            items, err = detect_ingredients(small, mime2)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.detected = items
                            st.session_state.confirmed = [it["name"] for it in items if it.get("confidence") != "low"]
                            st.session_state.stage = 2
                            st.rerun()
                    if st.button("🗑 Change photo", type="secondary", use_container_width=True):
                        st.session_state.photo = None
                        st.rerun()

    with col_manual:
        with st.container(border=True):
            st.markdown("**⌨️ Option B: Quick Type Ingredients**")
            st.caption("No photo on hand? Type or pick what you have in the fridge:")
            manual = st.text_area("Type ingredients", placeholder="e.g. eggs, cheddar cheese, tomatoes, spinach, garlic…", height=85, label_visibility="collapsed")
            
            # Quick starter buttons
            st.markdown("<small style='color:var(--text-dim);font-weight:600;'>Quick pantry staples:</small>", unsafe_allow_html=True)
            preset_cols = st.columns(4)
            staples = ["Eggs", "Cheese", "Tomatoes", "Chicken", "Onion", "Garlic", "Pasta", "Rice"]
            for i, stp in enumerate(staples):
                col_target = preset_cols[i % 4]
                if col_target.button(f"+ {stp}", key=f"quick_{stp}", use_container_width=True):
                    current_list = [x.strip() for x in manual.split(",") if x.strip()]
                    if stp not in current_list:
                        current_list.append(stp)
                    manual = ", ".join(current_list)

            if st.button("Use Ingredients →", use_container_width=True):
                names = [x.strip() for x in re.split(r"[,\n]+", manual or "") if x.strip()]
                if not names:
                    st.warning("Please type or select at least one ingredient.")
                else:
                    seen, items = set(), []
                    for n in names:
                        if n.lower() not in seen:
                            seen.add(n.lower())
                            items.append({"name": n, "amount": "", "confidence": "high"})
                    st.session_state.detected = items
                    st.session_state.confirmed = [it["name"] for it in items]
                    st.session_state.stage = 2
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Confirm Ingredients + Dial In Preferences (Clean 2-Column Grid)
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 2:
    st.markdown("""
    <div class="stage-hero">
      <h2>Confirm ingredients & customize your meal.</h2>
      <p>Review what was found, select your flavor profile, and let the LangChain agent build your menu.</p>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown("**🥬 Detected Ingredients**")
            st.caption("Click to toggle ingredients. Add any extras below:")

            detected_items = st.session_state.detected or []
            pill_labels = [f"{it['name']} ({it['amount']})" if it.get("amount") else it["name"] for it in detected_items]
            default_pills = [lbl for lbl, it in zip(pill_labels, detected_items) if it["name"] in st.session_state.confirmed]

            st.pills("Ingredients", pill_labels, selection_mode="multi", default=default_pills, key="pill_sel", on_change=sync_pills, label_visibility="collapsed")

            c_add1, c_add2 = st.columns([3, 1])
            with c_add1:
                extra = st.text_input("Add ingredient", placeholder="e.g. olive oil, chili flakes", label_visibility="collapsed")
            with c_add2:
                if st.button("Add", use_container_width=True):
                    if extra.strip():
                        for x in re.split(r"[,\n]+", extra):
                            x = x.strip()
                            if x and x.lower() not in {c.lower() for c in st.session_state.confirmed}:
                                st.session_state.confirmed.append(x)
                        st.rerun()

            st.markdown(f"<div style='font-size:12px;color:var(--text-dim);margin-top:6px;'>Selected: <b>{len(st.session_state.confirmed)}</b> ingredients</div>", unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown("**⚙️ Preferences**")
            
            st.markdown("<small style='font-weight:700;'>Cuisine Style</small>", unsafe_allow_html=True)
            cuisine = st.pills("Cuisine", CUISINES, default="Anything", label_visibility="collapsed") or "Anything"

            st.markdown("<small style='font-weight:700;'>Dietary Restriction</small>", unsafe_allow_html=True)
            diet = st.pills("Diet", DIETS, default="No restriction", label_visibility="collapsed") or "No restriction"

            c_time, c_serv = st.columns(2)
            with c_time:
                st.markdown("<small style='font-weight:700;'>Max Cook Time</small>", unsafe_allow_html=True)
                max_time = st.slider("Time", 10, 90, 30, step=5, format="%d min", label_visibility="collapsed")
            with c_serv:
                st.markdown("<small style='font-weight:700;'>Servings</small>", unsafe_allow_html=True)
                servings = st.pills("Servings", [1, 2, 3, 4, 6], default=2, label_visibility="collapsed") or 2

            use_kcal = st.checkbox("🎯 Target calorie cap per serving")
            kcal_target = st.slider("Calories", 250, 1000, 500, step=50, format="%d kcal", label_visibility="collapsed") if use_kcal else 0

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            c_back, c_cook = st.columns([1, 2])
            with c_back:
                if st.button("← Back", type="secondary", use_container_width=True):
                    st.session_state.stage = 1
                    st.rerun()
            with c_cook:
                if st.button("🍳 Generate 3 Recipes →", use_container_width=True, disabled=not st.session_state.confirmed):
                    with st.spinner("Chef Agent reasoning across your ingredients…"):
                        try:
                            recs = generate_recipes(st.session_state.confirmed, cuisine, diet, max_time, servings, kcal_target)
                            st.session_state.recipes = recs
                            # Preload images
                            for i, r in enumerate(recs):
                                st.session_state[f"img_{i}"] = get_dish_image(r["title"])
                            st.session_state.stage = 3
                            st.rerun()
                        except Exception as e:
                            st.error(f"Recipe error: {html.escape(str(e))[:300]}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Pick Recipe (Side-by-Side 3-Column Display, Zero Scrolling)
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown("""
    <div class="stage-hero">
      <h2>Pick your dish.</h2>
      <p>Here are 3 unique recipes created around your fridge. Select one to see full instructions & audio.</p>
    </div>""", unsafe_allow_html=True)

    char_badge_map = {
        "QUICK": ("recipe-badge-quick", "⚡ Quick & Easy"),
        "HEARTY": ("recipe-badge-hearty", "💪 Hearty & Filling"),
        "CREATIVE": ("recipe-badge-creative", "✨ Creative Twist")
    }

    cols = st.columns(3, gap="medium")
    recipes = st.session_state.recipes or []

    for idx, r in enumerate(recipes):
        col = cols[idx % 3]
        char = r.get("character", "QUICK").upper()
        badge_cls, badge_text = char_badge_map.get(char, ("recipe-badge-quick", f"🍽 {char}"))
        img_url = st.session_state.get(f"img_{idx}")

        with col:
            with st.container(border=True):
                st.markdown(f'<div class="{badge_cls}">{badge_text}</div>', unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin:0 0 6px;font-size:17px;font-family:Playfair Display,serif;'>{html.escape(r.get('title',''))}</h3>", unsafe_allow_html=True)
                
                if img_url:
                    st.image(img_url, use_container_width=True)
                
                st.markdown(f"""
                <div class="recipe-meta-tags">
                  <span class="recipe-tag">⏱ {r.get('time_min','?')}m</span>
                  <span class="recipe-tag">🔥 {r.get('calories_est','?')} kcal</span>
                  <span class="recipe-tag">👨‍🍳 {r.get('difficulty','Easy')}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"<div style='font-size:11.5px;color:var(--text-dim);margin-bottom:8px;'>Uses <b>{len(r.get('uses',[]))}</b> of your ingredients</div>", unsafe_allow_html=True)

                if st.button("Cook this →", key=f"btn_cook_{idx}", use_container_width=True):
                    st.session_state.picked = idx
                    st.session_state.stage = 4
                    st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("← Back to Preferences", type="secondary"):
        st.session_state.stage = 2
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Cooking Dashboard (2-Column Tablet Layout)
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 4:
    r = st.session_state.recipes[st.session_state.picked]
    img = st.session_state.get(f"img_{st.session_state.picked}")
    char = r.get("character", "QUICK").upper()
    uses = r.get("uses", [])
    miss = r.get("missing", [])
    steps = r.get("steps", [])

    st.markdown("""
    <div class="stage-hero">
      <h2>Let's cook!</h2>
      <p>Follow along below or hit the audio button for hands-free voice guidance.</p>
    </div>""", unsafe_allow_html=True)

    col_meta, col_steps = st.columns([0.85, 1.15], gap="large")

    with col_meta:
        with st.container(border=True):
            if img:
                st.image(img, use_container_width=True)

            st.markdown(f"""
            <div class="detail-header">
              <div class="detail-title">{html.escape(r.get('title',''))}</div>
              <div class="detail-meta-pills">
                <span class="recipe-tag">⏱ {r.get('time_min','?')} mins</span>
                <span class="recipe-tag" style="color:var(--accent)!important;">🔥 {r.get('calories_est','?')} kcal / serving</span>
                <span class="recipe-tag">👨‍🍳 {r.get('difficulty','Easy')}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Audio Reader
            st.markdown("<small style='font-weight:700;'>🎧 Hands-free Chef Voice</small>", unsafe_allow_html=True)
            if st.button("🔊 Read Recipe Aloud", key="btn_audio_trigger", use_container_width=True):
                with st.spinner("Generating chef narration…"):
                    generate_audio(st.session_state.picked, r.get("title", ""), steps)

            if f"audio_{st.session_state.picked}" in st.session_state:
                st.audio(st.session_state[f"audio_{st.session_state.picked}"], format="audio/mp3")

            # Ingredients Breakdown
            st.markdown("<div style='margin-top:10px;'><small style='font-weight:700;'>From your fridge:</small></div>", unsafe_allow_html=True)
            have_html = "".join(f'<span class="ing-tag-have">✓ {html.escape(u)}</span>' for u in uses)
            st.markdown(have_html or "<small style='color:var(--text-dim);'>Basic pantry items</small>", unsafe_allow_html=True)

            if miss:
                st.markdown("<div style='margin-top:8px;'><small style='font-weight:700;'>Additional items:</small></div>", unsafe_allow_html=True)
                need_html = "".join(f'<span class="ing-tag-need">+ {html.escape(m.get("item","") if isinstance(m,dict) else str(m))}</span>' for m in miss)
                st.markdown(need_html, unsafe_allow_html=True)

    with col_steps:
        with st.container(border=True):
            st.markdown("**🧑‍🍳 Step-by-Step Instructions**")
            
            for i, step_text in enumerate(steps):
                st.markdown(f"""
                <div class="step-item-card">
                  <div class="step-num-badge">{i+1}</div>
                  <div class="step-content">{html.escape(str(step_text))}</div>
                </div>
                """, unsafe_allow_html=True)

            if r.get("tip"):
                st.markdown(f"""
                <div class="pro-tip-box">
                  <b>💡 Pro Tip:</b> {html.escape(str(r["tip"]))}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            c_nav1, c_nav2 = st.columns(2)
            with c_nav1:
                if st.button("← Pick Another Recipe", type="secondary", use_container_width=True):
                    st.session_state.stage = 3
                    st.rerun()
            with c_nav2:
                if st.button("🔄 Start Fresh", type="secondary", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()

st.markdown('<div class="app-footer">FridgeSnap · LangChain Culinary Agent · Alibaba Vision & TTS</div>', unsafe_allow_html=True)
