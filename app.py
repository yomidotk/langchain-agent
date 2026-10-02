"""FridgeSnap LangChain Agent — snap your fridge, get recipes, hear them."""
import re
import json
import html
import time
import base64
import io
import requests
import streamlit as st
from gtts import gTTS
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

# ================= ALIBABA VISION =================
def detect_ingredients(img_bytes, mime):
    try:
        b64 = base64.b64encode(img_bytes).decode()
        prompt = """You are a precise kitchen assistant. Look at this fridge/kitchen photo and list every identifiable food ingredient or item you can see. Return ONLY valid JSON: {"ingredients": [{"name": "eggs", "amount": "about 6", "confidence": "high"}]}"""
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
            return None, f"Vision failed (HTTP {r.status_code})"
        
        data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
        clean = re.sub(r'```(?:json)?', '', content).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        out = json.loads(m.group(0) if m else clean)
        items = [it for it in out.get("ingredients", []) if isinstance(it, dict) and it.get("name")]
        if not items:
            return None, "Couldn't spot any ingredients."
        return items, ""
    except Exception as e:
        return None, f"Vision error: {str(e)[:220]}"

# ================= LANGCHAIN AGENT (RECIPES) =================
class MissingIngredient(BaseModel):
    item: str = Field(description="The missing item")
    swap: str = Field(description="A possible substitute from the provided ingredients")

class Recipe(BaseModel):
    title: str = Field(description="Name of the recipe")
    character: str = Field(description="Either QUICK, HEARTY, or CREATIVE")
    time_min: int = Field(description="Time to cook in minutes")
    calories_est: int = Field(description="Estimated calories per serving")
    difficulty: str = Field(description="Difficulty level")
    uses: List[str] = Field(description="List of provided ingredients used")
    missing: List[MissingIngredient] = Field(description="List of up to 3 missing ingredients")
    steps: List[str] = Field(description="Step by step cooking instructions")
    tip: str = Field(description="A helpful cooking tip")

class RecipeList(BaseModel):
    recipes: List[Recipe] = Field(description="Exactly 3 recipes")

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
        ("system", "You are a creative, practical chef agent. A home cook has exactly these ingredients:\n{ingredients}\n\nAssume basics are available (salt, pepper, oil, water). Constraints: cuisine = {cuisine}. {diet_line} Ready in under {max_time} minutes. Makes {servings} servings. {kcal_line}\n\nWrite exactly 3 recipes with DISTINCT characters: 1. QUICK, 2. HEARTY, 3. CREATIVE.\n\n{format_instructions}"),
        ("user", "Generate the recipes now.")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "ingredients": ing,
            "cuisine": cuisine,
            "diet_line": diet_line,
            "max_time": max_time,
            "servings": servings,
            "kcal_line": kcal_line,
            "format_instructions": parser.get_format_instructions()
        })
        return [r.model_dump() for r in result.recipes]
    except Exception as e:
        raise RuntimeError(f"LangChain generation failed: {e}")

# ================= AUDIO TTS =================
def generate_audio(recipe_idx, title, steps):
    text = f"Here is the recipe for {title}. " + " ".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    tts = gTTS(text=text, lang='en')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    st.session_state[f"audio_{recipe_idx}"] = buf.getvalue()

# ================= UI & CSS =================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@400;600;700;800&display=swap');

:root {
  --bg: #fff; --bg2: #F9FAFB; --bg3: #F3F4F6;
  --border: #E5E7EB; --border2: #D1D5DB;
  --text: #0A0A0A; --text2: #374151; --text3: #6B7280; --text4: #9CA3AF;
  --green: #059669;
}
*,*::before,*::after{box-sizing:border-box;}
*{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}
.stApp{background:var(--bg)!important;min-height:100vh;}
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
.block-container{max-width:960px!important;padding:0 clamp(16px,4vw,48px) 100px!important;margin:0 auto!important;}
section[data-testid="stSidebar"]{display:none!important;}

/* NAV */
.nav{position:sticky;top:0;z-index:100;background:rgba(255,255,255,0.92);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);margin:0 clamp(-16px,-4vw,-48px);padding:0 clamp(16px,4vw,48px);}
.nav-inner{max-width:1160px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:60px;}
.logo{font-family:'Sora',sans-serif!important;font-weight:800;font-size:18px;letter-spacing:-.04em;color:var(--text);display:flex;align-items:center;gap:9px;}
.logo-icon{width:30px;height:30px;border-radius:8px;background:var(--text)!important;color:#fff!important;display:flex;align-items:center;justify-content:center;font-size:14px;}
.logo-text span{color:var(--text3);}

/* HERO */
.hero{text-align:center;padding:clamp(60px,11vw,90px) 16px clamp(20px,4vw,40px);}
.hero h1{font-family:'Sora',sans-serif!important;font-size:clamp(2.6rem,6vw,4rem);font-weight:800;letter-spacing:-.05em;line-height:1.03;color:var(--text);margin:0 0 22px;}
.hero p.sub{font-size:clamp(.95rem,2.5vw,1.15rem);color:var(--text3);max-width:560px;margin:0 auto 8px;line-height:1.75;}
.hero p.sub b{color:var(--text);}

/* STEPS */
.stepper{display:flex;gap:6px;margin:24px auto 8px;max-width:600px;justify-content:center;}
.step{flex:1;text-align:center;font-size:12px;font-weight:700;color:var(--text4);padding:10px 4px;border-radius:10px;background:var(--bg2);letter-spacing:.02em;border:1px solid var(--border);}
.step.on{background:var(--text);color:#fff;border-color:var(--text);}
.step.done{background:var(--bg3);color:var(--text2);}

/* CARDS */
.card{background:var(--bg);border:1px solid var(--border);border-radius:20px;padding:32px;margin-top:24px;box-shadow:0 4px 20px rgba(0,0,0,.03);}
.card .k{font-size:11.5px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--green);margin-bottom:8px;}
.card h3{margin:0 0 6px;font-size:20px;font-weight:700;letter-spacing:-.01em;color:var(--text);}
.card .sub{font-size:14px;color:var(--text3);margin:0 0 20px;line-height:1.55;}

/* BUTTONS */
div[data-testid="stButton"] button{background:var(--text)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:16px 24px!important;font-size:14.5px!important;font-weight:600!important;letter-spacing:-.01em!important;box-shadow:0 1px 3px rgba(0,0,0,.12)!important;transition:opacity .15s,transform .15s!important;}
div[data-testid="stButton"] button p{color:#fff!important;}
div[data-testid="stButton"] button:hover{opacity:.86!important;transform:translateY(-1px)!important;box-shadow:0 4px 12px rgba(0,0,0,.15)!important;}
div[data-testid="stButton"] button[kind="secondary"]{background:var(--bg2)!important;color:var(--text2)!important;border:1px solid var(--border)!important;}
div[data-testid="stButton"] button[kind="secondary"] p{color:var(--text2)!important;}

/* INPUTS */
div[data-testid="stTextInput"] input{border-radius:12px!important;padding:14px 20px!important;font-size:14.5px!important;border:1px solid var(--border2)!important;background:var(--bg)!important;}
div[data-testid="stTextInput"] input:focus{border-color:var(--text)!important;box-shadow:0 0 0 3px rgba(10,10,10,.08)!important;}

/* EXPANDER */
div[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:14px!important;background:var(--bg2)!important;margin-top:16px;}
.photo{border-radius:14px;overflow:hidden;border:1px solid var(--border);margin:12px 0;}
.photo img{width:100%;display:block;}

/* RECIPES */
.badges{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 12px;}
.badge{font-size:12px;font-weight:600;background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:6px 12px;color:var(--text2);}
.badge.char{background:var(--text);border-color:var(--text);color:#fff;letter-spacing:.04em;text-transform:uppercase;font-size:11px;}
.useup{font-size:12px;font-weight:700;color:var(--green);background:rgba(5,150,105,.08);border-radius:8px;padding:6px 12px;display:inline-block;margin:12px 0 4px;}
.rtitle{font-size:24px;font-weight:800;letter-spacing:-.02em;color:var(--text);margin:12px 0 4px;font-family:'Sora',sans-serif!important;}
.have{font-size:14px;color:var(--text3);margin:8px 0;line-height:1.7;}
.have b{color:var(--text);}
.miss{font-size:14px;color:var(--text3);margin:8px 0;line-height:1.7;}
.miss b{color:var(--text);}
.steps{margin:20px 0 0;padding:0;list-style:none;counter-reset:s;}
.steps li{counter-increment:s;font-size:15px;line-height:1.75;color:var(--text2);margin-bottom:12px;padding-left:42px;position:relative;}
.steps li::before{content:counter(s);position:absolute;left:0;top:2px;width:28px;height:28px;border-radius:8px;background:var(--bg2);border:1px solid var(--border);color:var(--text2);font-size:13px;font-weight:700;display:flex;align-items:center;justify-content:center;}
.tip{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--text);border-radius:10px;padding:16px 20px;font-size:14px;color:var(--text3);margin-top:20px;line-height:1.6;}

/* SHOPPING */
.shop{background:var(--bg2);border:1px solid var(--border);border-radius:20px;padding:32px;margin-top:32px;}
.shop h3{color:var(--text);margin:0 0 8px;font-size:20px;font-weight:700;letter-spacing:-.01em;}
.shop .sub{color:var(--text3);font-size:14px;margin:0 0 16px;}
.shop ul{margin:0;padding-left:24px;}
.shop li{font-size:15px;line-height:2.1;color:var(--text2);}
.shop li span{color:var(--text4);font-size:13.5px;}

/* AUDIO */
div[data-testid="stAudio"]{margin-top:16px;border-radius:10px;overflow:hidden;border:1px solid var(--border);}

.footer{text-align:center;color:var(--text4);font-size:13px;margin-top:80px;line-height:1.7;padding-bottom:30px;border-top:1px solid var(--border);padding-top:30px;}
@media(max-width:640px){.card{padding:24px;}}
</style>
"""

st.set_page_config(page_title="FridgeSnap (LangChain)", layout="centered")
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown('''<div class="nav"><div class="nav-inner">
  <div class="logo">
    <div class="logo-icon">🥑</div>
    <div class="logo-text">Fridge<span>Snap</span> 🤖</div>
  </div>
</div></div>''', unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

# ---------- state ----------
for k, v in [("photo", None), ("detected", None), ("recipes", None), ("confirmed", []), ("extra_val", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

def sync_extra():
    val = st.session_state.get("extra_input", "").strip()
    if val:
        for x in re.split(r"[,\n]+", val):
            x = x.strip()
            if x and x.lower() not in {c.lower() for c in st.session_state.confirmed}:
                st.session_state.confirmed.append(x)
    st.session_state.extra_val = ""

def sync_pills():
    st.session_state.confirmed = [lbl.split(" · ")[0] for lbl in st.session_state.pill_selection]

# ---------- stepper ----------
stage = 1
if st.session_state.detected: stage = 2
if st.session_state.recipes: stage = 3
steps = ["📸 Photo", "🥬 Ingredients", "🍳 Recipes"]
st.markdown('<div class="stepper">' + "".join(
    f'<div class="step {"on" if i + 1 == stage else ("done" if i + 1 < stage else "")}">{s}</div>'
    for i, s in enumerate(steps)) + "</div>", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>LangChain Chef Agent<br>Cook what\'s inside.</h1>'
            '<p class="sub">Show me what you\'ve got — I\'ll spot the ingredients, you pick a cuisine, '
            'and my LangChain brain will build three recipes around <b>your</b> food.</p></div>',
            unsafe_allow_html=True)

# ---------- 1. photo ----------
st.markdown('<div class="card"><div class="k">Step 1</div><h3>Show me your fridge</h3>'
            '<p class="sub">Upload a photo or snap one now. Bright, straight-on shots work best.</p></div>',
            unsafe_allow_html=True)
tab_up, tab_cam = st.tabs(["📤 Upload photo", "📷 Take photo"])
with tab_up:
    up = st.file_uploader("fridge photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if up and st.session_state.photo != (up.getvalue(), up.type):
        st.session_state.photo = (up.getvalue(), up.type)
        st.session_state.detected = None
        st.session_state.recipes = None
with tab_cam:
    cam = st.camera_input("take a photo", label_visibility="collapsed")
    if cam and st.session_state.photo != (cam.getvalue(), cam.type):
        st.session_state.photo = (cam.getvalue(), cam.type)
        st.session_state.detected = None
        st.session_state.recipes = None

if st.session_state.photo:
    st.markdown(f'<div class="photo"><img src="data:{st.session_state.photo[1] or "image/jpeg"};base64,'
                f'{base64.b64encode(st.session_state.photo[0]).decode()}"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Identify ingredients", use_container_width=True):
            with st.spinner("Agent is analyzing your fridge…"):
                raw, mime = st.session_state.photo
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
        if st.button("🔄 New photo", use_container_width=True, type="secondary"):
            for k in ("photo", "detected", "recipes"):
                st.session_state[k] = None
            st.session_state.confirmed = []
            st.rerun()

with st.expander("⌨️ No photo? Type your ingredients instead"):
    manual = st.text_area("one per line or comma-separated", placeholder="eggs, chicken breast, tomatoes…", label_visibility="collapsed")
    if st.button("Use these ingredients", use_container_width=True):
        names = [x.strip() for x in re.split(r"[,\n]+", manual or "") if x.strip()]
        if not names:
            st.warning("Type at least one ingredient first.")
        else:
            seen, items, confirmed = set(), [], []
            for n in names:
                if n.lower() not in seen:
                    seen.add(n.lower())
                    items.append({"name": n, "amount": "", "confidence": "high"})
                    confirmed.append(n)
            st.session_state.detected = items
            st.session_state.confirmed = confirmed
            st.session_state.recipes = None
            st.rerun()

# ---------- 2. ingredients ----------
if st.session_state.detected:
    st.markdown('<div class="card"><div class="k">Step 2</div><h3>What\'s in there</h3>'
                '<p class="sub">Tap to toggle — deselect anything the AI got wrong.</p></div>', unsafe_allow_html=True)
    
    pill_labels = [it["name"] + (f" · {it['amount']}" if it.get("amount") else "") for it in st.session_state.detected]
    default_pills = [lbl for lbl in pill_labels if lbl.split(" · ")[0] in st.session_state.confirmed]
    
    st.pills("ingredients", pill_labels, selection_mode="multi", default=default_pills, 
             key="pill_selection", on_change=sync_pills, label_visibility="collapsed")
    
    st.text_input("Add what it missed", placeholder="e.g. rice, soy sauce + Enter", 
                  key="extra_input", value=st.session_state.extra_val, on_change=sync_extra, label_visibility="collapsed")

    # ---------- 3. options ----------
    st.markdown('<div class="card"><div class="k">Step 3</div><h3>How do you want it?</h3>'
                '<p class="sub">Dial it in — then let the LangChain Agent cook.</p></div>', unsafe_allow_html=True)
    
    cuisine = st.pills("Cuisine", CUISINES, default="Anything") or "Anything"
    diet = st.pills("Diet", DIETS, default="No restriction") or "No restriction"
    
    t1, t2 = st.columns(2)
    with t1:
        max_time = st.slider("Max time", 10, 90, 30, step=5, format="%d min")
    with t2:
        servings = st.pills("Servings", [1, 2, 3, 4, 5, 6], default=2) or 2
        
    use_kcal = st.checkbox("🎯 Set a calorie target per serving")
    kcal_target = st.slider("kcal target", 200, 1200, 500, step=50, format="%d kcal", label_visibility="collapsed") if use_kcal else 0

    if st.button("🍳 Get my recipes", type="primary", use_container_width=True, disabled=not st.session_state.confirmed):
        with st.spinner("LangChain Agent is cooking up three recipes…"):
            try:
                st.session_state.recipes = generate_recipes_langchain(st.session_state.confirmed, cuisine, diet, max_time, servings, kcal_target)
            except Exception as e:
                st.error(f"Couldn't generate recipes: {html.escape(str(e))[:250]}")
                st.session_state.recipes = None
        st.rerun()

# ---------- 4. recipes ----------
if st.session_state.recipes:
    total_have = max(len(st.session_state.confirmed), 1)
    all_missing = {}
    for idx, r in enumerate(st.session_state.recipes):
        uses = r.get("uses", []) or []
        pct = round(100 * len(uses) / total_have)
        st.markdown(f"""<div class="card">
          <div class="badges"><span class="badge char">{html.escape(str(r.get('character', '')))}</span>
          <span class="badge">⏱ {r.get('time_min', '?')} min</span>
          <span class="badge kcal">{r.get('calories_est', '?')} kcal/serv</span>
          <span class="badge">{html.escape(str(r.get('difficulty', '')))}</span></div>
          <div class="rtitle">{html.escape(r.get('title', ''))}</div>
          <span class="useup">Uses {len(uses)} of your {total_have} ingredients ({pct}%)</span>
          <p class="have"><b>✓ From your fridge:</b> {html.escape(", ".join(uses) or "—")}</p>""", unsafe_allow_html=True)
          
        miss = r.get("missing", [])
        if miss:
            miss_html = "; ".join(f"<b>{html.escape(str(m.get('item', '')))}</b>" + (f" <span>(or {html.escape(str(m['swap']))})</span>" if m.get("swap") else "") for m in miss if isinstance(m, dict))
            st.markdown(f'<p class="miss"><b>＋ You\'ll need:</b> {miss_html}</p>', unsafe_allow_html=True)
            for m in miss:
                if isinstance(m, dict) and m.get("item"):
                    all_missing[m["item"]] = m.get("swap", "")
                    
        steps_html = "".join(f"<li>{html.escape(str(s))}</li>" for s in r.get("steps", []))
        st.markdown(f'<ol class="steps">{steps_html}</ol>', unsafe_allow_html=True)
        if r.get("tip"):
            st.markdown(f'<div class="tip">💡 {html.escape(str(r["tip"]))}</div>', unsafe_allow_html=True)
        
        # Audio feature
        if st.button(f"🔊 Hear Recipe {idx+1}", key=f"btn_audio_{idx}"):
            with st.spinner("Generating audio..."):
                generate_audio(idx, r.get('title', ''), r.get('steps', []))
                
        if f"audio_{idx}" in st.session_state:
            st.audio(st.session_state[f"audio_{idx}"], format="audio/mp3")

        st.markdown("</div>", unsafe_allow_html=True)

    if all_missing:
        lis = "".join(f"<li>{html.escape(k)}" + (f" <span>— or use {html.escape(v)}</span>" if v else "") + "</li>" for k, v in all_missing.items())
        st.markdown(f'<div class="shop"><h3>🛒 Shopping list</h3><p class="sub">Everything missing, across all three recipes.</p><ul>{lis}</ul></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">FridgeSnap LangChain Agent — snap it, cook it.<br>Calorie estimates are rough guides, not medical advice.</div>', unsafe_allow_html=True)
