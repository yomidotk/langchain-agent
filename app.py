"""FridgeSnap — LangChain Chef Agent. Snap your fridge, hear your recipes."""
import re
import json
import html
import time
import base64
import io
import requests
import streamlit as st
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ================= CONFIG =================
DO_API_KEY = st.secrets.get("DO_API_KEY", "")
ALIBABA_API_KEY = st.secrets.get("ALIBABA_API_KEY", "")
DO_URL = "https://inference.do-ai.run/v1"
DO_MODEL = "openai-gpt-oss-20b"
VISION_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
VISION_MODEL = "qwen-vl-max"

CUISINES = ["Anything", "Italian", "Mexican", "Middle Eastern", "Asian", "Indian", "Mediterranean", "French", "American"]
DIETS = ["No restriction", "Vegetarian", "Vegan", "Halal", "Gluten-free", "High-protein"]

# ================= IMAGE PREP =================
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

# ================= VISION =================
def detect_ingredients(img_bytes, mime):
    try:
        b64 = base64.b64encode(img_bytes).decode()
        prompt = (
            "You are a precise kitchen assistant. Look at this fridge/kitchen photo and list every "
            "identifiable food ingredient or item you can see.\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{\"ingredients\": [{\"name\": \"eggs\", \"amount\": \"about 6\", \"confidence\": \"high\"}]}\n\n'
            "Rules:\n"
            "- Only list items you can actually SEE.\n"
            "- Names in plain English.\n"
            "- amount: short visible estimate, or empty string if unclear.\n"
            "- confidence: high / medium / low.\n"
            "- If this is NOT a food/kitchen photo, return {\"ingredients\": [], \"not_food\": true}.\n"
            "- Return STRICT JSON only. No markdown fences."
        )
        r = requests.post(
            VISION_URL,
            headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
            json={"model": VISION_MODEL,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
                  "temperature": 0.1},
            timeout=120)
        if r.status_code != 200:
            try:
                err = r.json().get("message") or r.json().get("error", {}).get("message") or r.text[:300]
            except Exception:
                err = r.text[:300]
            return None, f"Vision API error (HTTP {r.status_code}): {err}"

        data = r.json()
        raw = data["choices"][0]["message"]["content"] or ""
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") for b in raw if isinstance(b, dict))
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return None, f"Vision returned unexpected output: {clean[:200]}"
        out = json.loads(m.group(0))
        if out.get("not_food"):
            return None, "That doesn't look like a fridge photo — try another."

        # Case-insensitive key match for "ingredients"
        items_raw = None
        for k, v in out.items():
            if k.lower() == "ingredients":
                items_raw = v
                break
        if not items_raw:
            return None, f"Vision returned no ingredient list. Raw: {clean[:200]}"

        items = [it for it in items_raw if isinstance(it, dict) and it.get("name")]
        if not items:
            return None, f"Couldn't identify any items. Raw: {clean[:200]}"
        return items, ""
    except Exception as e:
        return None, f"Vision error: {str(e)[:300]}"

# ================= LANGCHAIN MODELS =================
class MissingIngredient(BaseModel):
    item: str = Field(description="The missing item name")
    swap: str = Field(description="A substitute from the provided ingredients, or empty string")

class Recipe(BaseModel):
    title: str = Field(description="Recipe name")
    character: str = Field(description="QUICK, HEARTY, or CREATIVE")
    time_min: int = Field(description="Cook time in minutes")
    calories_est: int = Field(description="Estimated calories per serving")
    difficulty: str = Field(description="Easy, Medium, or Hard")
    uses: List[str] = Field(description="Provided ingredients used in this recipe")
    missing: List[MissingIngredient] = Field(description="Up to 3 extra items needed")
    steps: List[str] = Field(description="Step by step instructions, 4-8 steps")
    tip: str = Field(description="One helpful pro tip")

class RecipeList(BaseModel):
    recipes: List[Recipe] = Field(description="Exactly 3 recipes")

# ================= LANGCHAIN AGENT =================
def generate_recipes_langchain(ingredients, cuisine, diet, max_time, servings, kcal_target):
    parser = PydanticOutputParser(pydantic_object=RecipeList)
    llm = ChatOpenAI(
        model_name=DO_MODEL,
        openai_api_key=DO_API_KEY,
        openai_api_base=DO_URL,
        temperature=0.6,
        max_tokens=3000
    )
    ing = ", ".join(ingredients)
    kcal_line = f"Each serving must stay under ~{kcal_target} kcal." if kcal_target else ""
    diet_line = f"Dietary rule: {diet}." if diet != "No restriction" else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a creative, practical chef agent. A home cook has these ingredients:\n{ingredients}\n\n"
         "Assume basics: salt, pepper, oil, water, sugar.\n"
         "Constraints: cuisine = {cuisine}. {diet_line} Under {max_time} min. Makes {servings} servings. {kcal_line}\n\n"
         "Write exactly 3 recipes: 1. QUICK (under 20 min), 2. HEARTY (most filling), 3. CREATIVE (surprising).\n"
         "Rules: uses[] must only contain items from the list. missing[] max 3 items.\n\n"
         "{format_instructions}"),
        ("user", "Generate the recipes now.")
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "ingredients": ing, "cuisine": cuisine, "diet_line": diet_line,
        "max_time": max_time, "servings": servings, "kcal_line": kcal_line,
        "format_instructions": parser.get_format_instructions()
    })
    return [r.model_dump() for r in result.recipes]

# ================= AUDIO =================
def generate_audio(recipe_idx, title, steps):
    dashscope.api_key = ALIBABA_API_KEY
    text = f"Recipe: {title}. " + " ".join(f"Step {i+1}. {s}" for i, s in enumerate(steps))
    text = text[:500]
    result = SpeechSynthesizer.call(model="sambert-zhichu-v1", text=text, format="mp3")
    if result.get_audio_data() is not None:
        st.session_state[f"audio_{recipe_idx}"] = result.get_audio_data()
    else:
        st.error(f"Audio failed: {result.message}")

# ================= CSS =================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg:#0E0C0A; --surface:#181510; --card:#1F1C17;
  --border:#2E2920; --border2:#3D3830;
  --amber:#E8A23C; --amber2:#F5C97A; --amber-dim:#7A541C;
  --text:#F5EFE4; --text2:#C2B89E; --text3:#8A7F6E; --text4:#5C5449;
  --green:#3DAA6A; --red:#D9614A;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
*{font-family:'Inter',sans-serif!important;}
html{scroll-behavior:smooth;}
.stApp{background:var(--bg)!important;min-height:100vh;}
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
.block-container{max-width:900px!important;padding:0 clamp(16px,4vw,40px) 120px!important;margin:0 auto!important;}
section[data-testid="stSidebar"]{display:none!important;}

/* NAV */
.nav{display:flex;align-items:center;gap:12px;padding:28px 0 12px;}
.logo-mark{width:38px;height:38px;border-radius:12px;background:var(--amber);display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 16px rgba(232,162,60,.35);}
.brand{font-family:'Bricolage Grotesque',sans-serif!important;font-weight:800;font-size:22px;color:var(--text);letter-spacing:-.03em;}
.brand span{color:var(--amber);}
.beta{font-size:10px;font-weight:600;letter-spacing:.12em;color:var(--amber-dim);background:rgba(232,162,60,.1);border:1px solid rgba(232,162,60,.2);border-radius:6px;padding:3px 8px;text-transform:uppercase;}

/* STEPPER */
.stepper{display:flex;gap:8px;margin:8px 0 32px;}
.stp{flex:1;display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;}
.stp-num{width:24px;height:24px;border-radius:8px;background:var(--border2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--text3);flex-shrink:0;}
.stp-label{font-size:12px;font-weight:600;color:var(--text3);white-space:nowrap;}
.stp.on{border-color:var(--amber);background:rgba(232,162,60,.07);}
.stp.on .stp-num{background:var(--amber);color:#0E0C0A;}
.stp.on .stp-label{color:var(--amber);}
.stp.done{border-color:rgba(61,170,106,.3);background:rgba(61,170,106,.05);}
.stp.done .stp-num{background:var(--green);color:#fff;}
.stp.done .stp-label{color:var(--green);}

/* HERO */
.hero{margin:48px 0 40px;}
.hero-eyebrow{font-size:11px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--amber);margin-bottom:14px;display:flex;align-items:center;gap:8px;}
.hero-eyebrow::before{content:'';width:20px;height:1px;background:var(--amber);}
.hero h1{font-family:'Bricolage Grotesque',sans-serif!important;font-size:clamp(2.4rem,7vw,4.2rem);font-weight:800;color:var(--text);letter-spacing:-.04em;line-height:1.05;margin-bottom:16px;}
.hero h1 em{font-style:normal;color:var(--amber);}
.hero p{font-size:16px;color:var(--text3);line-height:1.8;max-width:520px;}

/* CARDS */
.section-label{font-size:10.5px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;display:flex;align-items:center;gap:10px;}
.section-label::after{content:'';flex:1;height:1px;background:var(--border);}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:clamp(20px,4vw,32px);margin-bottom:16px;}
.card-title{font-family:'Bricolage Grotesque',sans-serif!important;font-size:17px;font-weight:700;color:var(--text);margin-bottom:6px;}
.card-sub{font-size:13.5px;color:var(--text3);line-height:1.55;margin-bottom:20px;}

/* BUTTONS */
div[data-testid="stButton"] button{background:var(--amber)!important;color:#0E0C0A!important;border:none!important;border-radius:12px!important;padding:14px 22px!important;font-size:14px!important;font-weight:700!important;letter-spacing:-.01em!important;box-shadow:0 4px 16px rgba(232,162,60,.25)!important;transition:all .15s!important;}
div[data-testid="stButton"] button p{color:#0E0C0A!important;}
div[data-testid="stButton"] button:hover{filter:brightness(1.1)!important;transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(232,162,60,.35)!important;}
div[data-testid="stButton"] button[kind="secondary"]{background:var(--surface)!important;color:var(--text2)!important;border:1px solid var(--border2)!important;box-shadow:none!important;}
div[data-testid="stButton"] button[kind="secondary"] p{color:var(--text2)!important;}
div[data-testid="stButton"] button[kind="secondary"]:hover{background:var(--card)!important;border-color:var(--text3)!important;}
div[data-testid="stButton"] button:disabled{opacity:.35!important;transform:none!important;box-shadow:none!important;}

/* INPUTS */
div[data-testid="stTextInput"] input{background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:12px!important;padding:14px 18px!important;font-size:14px!important;color:var(--text)!important;}
div[data-testid="stTextInput"] input::placeholder{color:var(--text4)!important;}
div[data-testid="stTextInput"] input:focus{border-color:var(--amber)!important;box-shadow:0 0 0 3px rgba(232,162,60,.12)!important;}
div[data-testid="stTextArea"] textarea{background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:12px!important;color:var(--text)!important;}

/* TABS */
div[data-testid="stTabs"] [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:1px solid var(--border)!important;}
div[data-testid="stTabs"] [data-baseweb="tab"]{border-radius:9px!important;font-weight:600!important;color:var(--text3)!important;background:transparent!important;border:none!important;font-size:13.5px!important;padding:9px 18px!important;}
div[data-testid="stTabs"] [aria-selected="true"]{background:var(--card)!important;color:var(--text)!important;box-shadow:0 1px 4px rgba(0,0,0,.4)!important;}

/* PILLS */
div[data-testid="stPills"] button{background:var(--surface)!important;border:1px solid var(--border2)!important;border-radius:999px!important;color:var(--text3)!important;font-weight:600!important;font-size:13px!important;}
div[data-testid="stPills"] button[aria-pressed="true"]{background:var(--amber)!important;border-color:var(--amber)!important;color:#0E0C0A!important;}

/* FILE UPLOADER */
div[data-testid="stFileUploader"]{border:2px dashed var(--border2)!important;border-radius:16px!important;background:var(--surface)!important;}
div[data-testid="stFileUploader"]:hover{border-color:var(--amber)!important;}

/* PHOTO */
.photo{border-radius:16px;overflow:hidden;border:1px solid var(--border);margin:14px 0;}
.photo img{width:100%;display:block;}

/* INGREDIENT BADGES */
.badges{display:flex;flex-wrap:wrap;gap:7px;margin:6px 0 16px;}
.ibadge{display:inline-flex;align-items:center;gap:6px;background:var(--surface);border:1px solid var(--border2);border-radius:10px;padding:7px 14px;font-size:13px;color:var(--text2);font-weight:500;}
.ibadge .conf-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.conf-high{background:var(--green);}
.conf-medium{background:var(--amber);}
.conf-low{background:var(--red);}

/* RECIPE CARDS */
.recipe-card{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:32px;margin-bottom:20px;position:relative;overflow:hidden;}
.recipe-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--amber),var(--amber2));}
.char-badge{display:inline-flex;align-items:center;gap:6px;font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--amber);background:rgba(232,162,60,.1);border:1px solid rgba(232,162,60,.2);border-radius:6px;padding:4px 10px;margin-bottom:16px;}
.recipe-title{font-family:'Bricolage Grotesque',sans-serif!important;font-size:clamp(1.3rem,4vw,1.8rem);font-weight:800;color:var(--text);letter-spacing:-.03em;margin-bottom:12px;}
.meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;}
.meta-tag{font-size:12.5px;font-weight:600;color:var(--text2);background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 12px;}
.meta-tag.kcal{color:var(--amber2);border-color:var(--amber-dim);}
.uses-section{margin-bottom:14px;}
.uses-label{font-size:10.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--green);margin-bottom:8px;}
.uses-pills{display:flex;flex-wrap:wrap;gap:6px;}
.use-pill{font-size:12px;background:rgba(61,170,106,.08);border:1px solid rgba(61,170,106,.2);color:var(--green);border-radius:8px;padding:4px 10px;font-weight:600;}
.miss-label{font-size:10.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);margin:14px 0 8px;}
.miss-pills{display:flex;flex-wrap:wrap;gap:6px;}
.miss-pill{font-size:12px;background:rgba(232,162,60,.08);border:1px solid rgba(232,162,60,.2);color:var(--amber2);border-radius:8px;padding:4px 10px;font-weight:600;}
.steps-section{margin:20px 0 0;}
.steps-label{font-size:10.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;}
.step-item{display:flex;gap:14px;margin-bottom:14px;align-items:flex-start;}
.step-num{width:28px;height:28px;border-radius:8px;background:var(--border2);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:var(--text2);flex-shrink:0;margin-top:1px;}
.step-text{font-size:14.5px;line-height:1.7;color:var(--text2);}
.tip-box{background:rgba(232,162,60,.06);border:1px solid rgba(232,162,60,.15);border-radius:12px;padding:14px 18px;font-size:13.5px;color:var(--amber2);line-height:1.65;margin-top:20px;}
.tip-box b{color:var(--amber);font-weight:700;}

/* SLIDER */
div[data-testid="stSlider"] div[data-testid="stThumbValue"]{color:var(--amber)!important;}

/* EXPANDER */
div[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:14px!important;margin-top:12px;}

/* AUDIO */
div[data-testid="stAudio"]{margin-top:16px;border-radius:12px;overflow:hidden;border:1px solid var(--border);}

/* SHOP */
.shop{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:28px;margin-top:24px;}
.shop-title{font-family:'Bricolage Grotesque',sans-serif!important;font-size:18px;font-weight:800;color:var(--text);margin-bottom:6px;}
.shop-sub{font-size:13.5px;color:var(--text3);margin-bottom:18px;}
.shop-item{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--border);}
.shop-item:last-child{border-bottom:none;}
.shop-dot{width:7px;height:7px;border-radius:50%;background:var(--amber);flex-shrink:0;}
.shop-name{font-size:14px;font-weight:600;color:var(--text2);}
.shop-swap{font-size:12.5px;color:var(--text4);margin-left:auto;}

/* STREAMLIT ALERTS */
.stAlert{border-radius:12px!important;background:rgba(217,97,74,.08)!important;border:1px solid rgba(217,97,74,.3)!important;}

.footer{text-align:center;color:var(--text4);font-size:12.5px;margin-top:80px;padding:24px 0;border-top:1px solid var(--border);line-height:1.7;}

@media(max-width:640px){
  .stepper{flex-direction:column;gap:6px;}
  .card{padding:18px;}
  .recipe-card{padding:22px;}
  .meta-row{gap:6px;}
}
</style>
"""

# ================= APP =================
st.set_page_config(page_title="FridgeSnap", layout="centered", page_icon="🥑")
st.markdown(_CSS, unsafe_allow_html=True)

# NAV
st.markdown("""
<div class="nav">
  <div class="logo-mark">🍳</div>
  <div class="brand">Fridge<span>Snap</span></div>
  <div class="beta">LangChain Agent</div>
</div>""", unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("⚠️ Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in Streamlit Secrets.")
    st.stop()

# ---------- Session State ----------
for k, v in [("photo", None), ("detected", None), ("recipes", None), ("confirmed", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

def sync_pills():
    st.session_state.confirmed = [lbl.split(" (")[0] for lbl in st.session_state.get("pill_sel", [])]

# ---------- Stepper ----------
stage = 1
if st.session_state.detected: stage = 2
if st.session_state.recipes: stage = 3

def _stp(i, label, emoji):
    cls = "on" if i == stage else ("done" if i < stage else "")
    return f'<div class="stp {cls}"><div class="stp-num">{emoji if i < stage else i}</div><div class="stp-label">{label}</div></div>'

st.markdown(f"""
<div class="stepper">
  {_stp(1,"Snap Photo","✓")}
  {_stp(2,"Ingredients","✓")}
  {_stp(3,"Recipes","✓")}
</div>
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Chef</div>
  <h1>Snap your fridge.<br><em>Cook something great.</em></h1>
  <p>Upload a photo — the agent spots what's inside, you choose your vibe, and it builds three real recipes from <b style="color:var(--text)">exactly what you have</b>.</p>
</div>""", unsafe_allow_html=True)

# ---------- STEP 1: Photo ----------
st.markdown('<div class="section-label">Step 1 — Add your photo</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><div class="card-title">📸 Show me your fridge</div><div class="card-sub">Upload a photo or take one now. Bright, straight-on shots give the best results.</div>', unsafe_allow_html=True)

tab_up, tab_cam = st.tabs(["📤 Upload photo", "📷 Take photo"])
with tab_up:
    up = st.file_uploader("fridge photo", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if up:
        val = (up.getvalue(), up.type)
        if st.session_state.photo != val:
            st.session_state.photo = val
            st.session_state.detected = None
            st.session_state.recipes = None
with tab_cam:
    cam = st.camera_input("snap a photo", label_visibility="collapsed")
    if cam:
        val = (cam.getvalue(), cam.type)
        if st.session_state.photo != val:
            st.session_state.photo = val
            st.session_state.detected = None
            st.session_state.recipes = None

if st.session_state.photo:
    raw, mime = st.session_state.photo
    st.markdown(f'<div class="photo"><img src="data:{mime};base64,{base64.b64encode(raw).decode()}"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("🔍 Analyse with AI", use_container_width=True):
            with st.spinner("Vision model scanning your fridge…"):
                small, mime2 = prep_image(raw, mime)
                items, err = detect_ingredients(small, mime2)
            if err:
                st.error(err)
            else:
                st.session_state.detected = items
                st.session_state.confirmed = [it["name"] for it in items if it.get("confidence") != "low"]
                st.session_state.recipes = None
                st.rerun()
    with c2:
        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            st.session_state.update({"photo": None, "detected": None, "recipes": None, "confirmed": []})
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("⌨️ No photo? Type your ingredients manually"):
    manual = st.text_area("One per line or comma-separated", placeholder="eggs, chicken breast, tomatoes, milk…", label_visibility="collapsed", height=100)
    if st.button("Use these ingredients →", use_container_width=True):
        names = [x.strip() for x in re.split(r"[,\n]+", manual or "") if x.strip()]
        if not names:
            st.warning("Type at least one ingredient first.")
        else:
            seen, items = set(), []
            for n in names:
                if n.lower() not in seen:
                    seen.add(n.lower())
                    items.append({"name": n, "amount": "", "confidence": "high"})
            st.session_state.update({"detected": items, "confirmed": [it["name"] for it in items], "recipes": None})
            st.rerun()

# ---------- STEP 2: Ingredients ----------
if st.session_state.detected:
    st.markdown('<div class="section-label">Step 2 — Confirm ingredients</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">🥬 What\'s in there</div><div class="card-sub">Toggle to deselect anything the AI got wrong. Green dot = confident, orange = unsure.</div>', unsafe_allow_html=True)

    pill_labels = [f"{it['name']} ({it['amount']})" if it.get("amount") else it["name"] for it in st.session_state.detected]
    default_pills = [lbl for lbl, it in zip(pill_labels, st.session_state.detected) if it["name"] in st.session_state.confirmed]
    st.pills("Pick ingredients", pill_labels, selection_mode="multi", default=default_pills, key="pill_sel", on_change=sync_pills, label_visibility="collapsed")

    extra = st.text_input("➕ Add anything it missed", placeholder="e.g. rice, soy sauce", label_visibility="collapsed")
    if extra.strip():
        for x in re.split(r"[,\n]+", extra):
            x = x.strip()
            if x and x.lower() not in {c.lower() for c in st.session_state.confirmed}:
                st.session_state.confirmed.append(x)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- STEP 3: Options ----------
    st.markdown('<div class="section-label">Step 3 — Your preferences</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">⚙️ Dial it in</div><div class="card-sub">Customise the recipes before the agent cooks.</div>', unsafe_allow_html=True)

    st.markdown("**Cuisine**")
    cuisine = st.pills("Cuisine", CUISINES, default="Anything", label_visibility="collapsed") or "Anything"
    st.markdown("**Dietary preference**")
    diet = st.pills("Diet", DIETS, default="No restriction", label_visibility="collapsed") or "No restriction"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Max cook time**")
        max_time = st.slider("Max time", 10, 90, 30, step=5, format="%d min", label_visibility="collapsed")
    with c2:
        st.markdown("**Servings**")
        servings = st.pills("Servings", [1,2,3,4,5,6], default=2, label_visibility="collapsed") or 2

    use_kcal = st.checkbox("🎯 Set calorie target per serving")
    kcal_target = st.slider("Calorie target", 200, 1200, 500, step=50, format="%d kcal", label_visibility="collapsed") if use_kcal else 0

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🍳 Generate my recipes", type="primary", use_container_width=True, disabled=not st.session_state.confirmed):
        with st.spinner("LangChain Agent is thinking and cooking…"):
            try:
                st.session_state.recipes = generate_recipes_langchain(st.session_state.confirmed, cuisine, diet, max_time, servings, kcal_target)
            except Exception as e:
                st.error(f"Recipe error: {html.escape(str(e))[:300]}")
                st.session_state.recipes = None
        st.rerun()

# ---------- STEP 4: Recipes ----------
if st.session_state.recipes:
    st.markdown('<div class="section-label">Your recipes</div>', unsafe_allow_html=True)
    total_have = max(len(st.session_state.confirmed), 1)
    all_missing = {}

    for idx, r in enumerate(st.session_state.recipes):
        uses = r.get("uses", [])
        miss = r.get("miss", r.get("missing", []))
        char_icons = {"QUICK": "⚡", "HEARTY": "💪", "CREATIVE": "✨"}
        char = r.get("character", "")
        icon = char_icons.get(char, "🍽")

        st.markdown(f"""<div class="recipe-card">
  <div class="char-badge">{icon} {html.escape(char)}</div>
  <div class="recipe-title">{html.escape(r.get('title',''))}</div>
  <div class="meta-row">
    <div class="meta-tag">⏱ {r.get('time_min','?')} min</div>
    <div class="meta-tag kcal">🔥 {r.get('calories_est','?')} kcal/serving</div>
    <div class="meta-tag">👨‍🍳 {html.escape(str(r.get('difficulty','')))} </div>
    <div class="meta-tag">👥 Uses {len(uses)}/{total_have} ingredients</div>
  </div>""", unsafe_allow_html=True)

        if uses:
            pills_html = "".join(f'<span class="use-pill">✓ {html.escape(u)}</span>' for u in uses)
            st.markdown(f'<div class="uses-section"><div class="uses-label">From your fridge</div><div class="uses-pills">{pills_html}</div></div>', unsafe_allow_html=True)

        if miss:
            pills_html = "".join(f'<span class="miss-pill">+ {html.escape(str(m.get("item","") if isinstance(m,dict) else m))}</span>' for m in miss)
            st.markdown(f'<div class="miss-label">You\'ll need</div><div class="miss-pills">{pills_html}</div>', unsafe_allow_html=True)
            for m in miss:
                if isinstance(m, dict) and m.get("item"):
                    all_missing[m["item"]] = m.get("swap", "")

        steps = r.get("steps", [])
        if steps:
            steps_html = "".join(f'<div class="step-item"><div class="step-num">{i+1}</div><div class="step-text">{html.escape(str(s))}</div></div>' for i, s in enumerate(steps))
            st.markdown(f'<div class="steps-section"><div class="steps-label">Instructions</div>{steps_html}</div>', unsafe_allow_html=True)

        if r.get("tip"):
            st.markdown(f'<div class="tip-box"><b>💡 Pro tip:</b> {html.escape(str(r["tip"]))}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Audio
        if st.button(f"🔊 Hear this recipe aloud", key=f"aud_{idx}"):
            with st.spinner("Generating voice…"):
                generate_audio(idx, r.get("title",""), steps)
        if f"audio_{idx}" in st.session_state:
            st.audio(st.session_state[f"audio_{idx}"], format="audio/mp3")

    # Shopping list
    if all_missing:
        items_html = "".join(f'<div class="shop-item"><div class="shop-dot"></div><div class="shop-name">{html.escape(k)}</div>' + (f'<div class="shop-swap">or: {html.escape(v)}</div>' if v else "") + "</div>" for k, v in all_missing.items())
        st.markdown(f'<div class="shop"><div class="shop-title">🛒 Shopping List</div><div class="shop-sub">Everything missing across all 3 recipes.</div>{items_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">FridgeSnap · Powered by LangChain + Alibaba Vision · Calorie estimates are rough guides only.</div>', unsafe_allow_html=True)
