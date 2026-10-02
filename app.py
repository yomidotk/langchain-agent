"""FridgeSnap — snap your fridge, get recipes.
Upload (or photograph) your fridge -> AI lists what's inside -> pick cuisine,
calorie target, diet, time -> 3 recipes built from YOUR ingredients, with a
combined shopping list for what's missing.
"""
import re
import json
import html
import time
import base64
import io

import requests
import streamlit as st

# ================= CONFIG =================
DO_API_KEY = st.secrets.get("DO_API_KEY", "")
ALIBABA_API_KEY = st.secrets.get("ALIBABA_API_KEY", "")
DO_URL = "https://inference.do-ai.run/v1/chat/completions"
DO_MODEL = "openai-gpt-oss-20b"
# Vision: DashScope OpenAI-compatible endpoint (Alibaba key). qwen-vl-max is strong
# at reading fridge photos. Alternatives: qwen-vl-plus, qwen2.5-vl-72b-instruct.
VISION_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
VISION_MODEL = "qwen-vl-max"

STRICT_JSON = """
OUTPUT RULES (follow exactly):
- Return STRICT JSON ONLY. No prose before or after, no markdown fences, no comments.
- Double quotes on all keys and strings. No trailing commas.
- Include every required field. If a field is truly unknown, use "" or [] — never invent a value.
- Your response must start with { and end with }."""

CUISINES = ["Anything", "Italian", "Mexican", "Middle Eastern", "Asian", "Indian",
            "Mediterranean", "French", "American"]
DIETS = ["No restriction", "Vegetarian", "Vegan", "Halal", "Gluten-free", "High-protein"]

# ================= IMAGE PREP =================
def prep_image(file_bytes, mime):
    """Downscale to max 1024px JPEG so the vision call is fast and cheap."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return file_bytes, mime or "image/jpeg"

# ================= VISION (what's in the fridge?) =================
def detect_ingredients(img_bytes, mime):
    """Returns (ingredients_list, error_str). ingredients: [{name, amount, confidence}]."""
    b64 = base64.b64encode(img_bytes).decode()
    prompt = """You are a precise kitchen assistant. Look at this fridge/kitchen photo and list every identifiable food ingredient or item you can see.

Return STRICT JSON: {"ingredients": [{"name": "eggs", "amount": "about 6", "confidence": "high"}, {"name": "milk", "amount": "half bottle", "confidence": "medium"}]}

Rules:
- Only list what you can actually SEE. Never guess what's behind closed doors or in opaque containers.
- Names in plain English, singular-ish ("chicken breast", not "chx brst").
- amount: short visible estimate ("3", "a bunch", "half jar"). Use "" if unclear.
- confidence: high/medium/low.
- Merge obvious duplicates (two milk bottles -> one entry "2 bottles").
- If this is not a fridge/kitchen/food photo at all, return {"ingredients": [], "not_food": true}.
""" + STRICT_JSON
    try:
        r = requests.post(
            VISION_URL,
            headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
            json={"model": VISION_MODEL,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
                  "temperature": 0.1},
            timeout=120)
    except Exception as e:
        return None, f"vision request failed: {str(e)[:150]}"
    if r.status_code != 200:
        return None, f"vision HTTP {r.status_code}: {r.text[:200]}"
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return None, f"bad vision response: {str(e)[:120]}"
    clean = re.sub(r'```(?:json)?', '', content).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    try:
        out = json.loads(m.group(0) if m else clean)
    except Exception:
        return None, "vision returned non-JSON"
    if out.get("not_food"):
        return None, "that doesn't look like a fridge or kitchen photo — try another one"
    items = [it for it in out.get("ingredients", []) if isinstance(it, dict) and it.get("name")]
    if not items:
        return None, "couldn't spot any ingredients — try a clearer, brighter photo"
    return items, ""

# ================= TEXT LLM (recipes) =================
def _do_text(prompt, max_tokens, temperature, label="recipe step", _tries=3):
    last_err = None
    for attempt in range(_tries):
        if attempt:
            time.sleep(2 * attempt)
        try:
            r = requests.post(
                DO_URL,
                headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"},
                json={"model": DO_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": int(max_tokens * (1.5 ** attempt)),
                      "temperature": 0.0 if attempt else temperature,
                      "response_format": {"type": "json_object"}, "stream": False},
                timeout=180)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last_err = e
            continue
        clean = re.sub(r'```(?:json)?', '', content).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        try:
            parsed = json.loads(m.group(0) if m else clean)
            if isinstance(parsed, dict):
                return parsed
            last_err = ValueError("not a JSON object")
        except Exception as e:
            last_err = e
            prompt = ("Your last response was not valid JSON. Return the COMPLETE object again, "
                      "valid JSON only, no truncation.\n\nBroken output:\n" + (m.group(0) if m else clean)[:4000])
    raise RuntimeError(f"{label} failed: {last_err}")

def generate_recipes(ingredients, cuisine, diet, max_time, servings, kcal_target):
    ing = ", ".join(ingredients)
    kcal_line = f"Each serving must stay under ~{kcal_target} kcal (honest estimate)." if kcal_target else ""
    diet_line = f"Dietary rule: {diet}." if diet != "No restriction" else ""
    prompt = f"""You are a creative, practical chef. A home cook has exactly these ingredients on hand:
{ing}

Assume basics are available: salt, pepper, oil, water, sugar.
Constraints: cuisine = {cuisine}. {diet_line} Ready in under {max_time} minutes. Makes {servings} servings. {kcal_line}

Write exactly 3 recipes with DISTINCT characters:
1. QUICK — under 20 minutes, minimal effort.
2. HEARTY — the most filling, complete meal of the three.
3. CREATIVE — a surprising but delicious combo.

RULES (follow exactly):
- "uses" may ONLY contain items from the list above (match names loosely).
- "missing": at most 3 extra items per recipe, common pantry things only, each with a "swap" (something from the list that could replace it, or "").
- Never invent exotic ingredients. If a recipe can't work with what's here, don't fake it — pick one that can.
- Steps: 4-8 concrete, ordered, beginner-friendly. No vague "cook until done" — give times and cues.
- calories_est: honest per-serving estimate. difficulty: Easy/Medium.
- The 3 recipes must not repeat each other's main idea.

Return STRICT JSON: {{"recipes": [{{"title": "...", "character": "QUICK|HEARTY|CREATIVE", "time_min": 15, "calories_est": 450, "difficulty": "Easy", "uses": ["eggs", "tomatoes"], "missing": [{{"item": "feta", "swap": "any white cheese"}}], "steps": ["..."], "tip": "one pro tip"}}]}}
""" + STRICT_JSON
    out = _do_text(prompt, 3000, 0.6, label="recipe generation")
    recs = [r for r in out.get("recipes", []) if isinstance(r, dict) and r.get("title")]
    if not recs:
        raise RuntimeError("the AI returned no recipes — try again")
    return recs[:3]

# ================= UI =================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#1d1a16;--paper:#FFFCF7;--card:#fff;--line:#EDE4D6;--muted:#8a7f70;--green:#2E7D4F;--amber:#C2703D;}
*{font-family:'Inter',-apple-system,'Segoe UI',sans-serif!important;}
.stApp{background:var(--paper)!important;}
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
.block-container{max-width:860px!important;padding:0 20px 90px!important;}
section[data-testid="stSidebar"]{display:none!important;}

.nav{display:flex;align-items:center;gap:10px;padding:18px 0;border-bottom:1px solid var(--line);}
.brand-mark{width:30px;height:30px;border-radius:9px;background:var(--green);display:flex;align-items:center;justify-content:center;font-size:15px;}
.brand{font-weight:800;font-size:18px;letter-spacing:-.02em;color:var(--ink);}
.hero{text-align:center;padding:52px 8px 8px;}
.hero h1{font-size:clamp(2rem,6vw,3.2rem);font-weight:800;letter-spacing:-.04em;line-height:1.06;margin:0 0 14px;color:var(--ink);}
.hero h1 em{font-style:normal;color:var(--green);}
.hero p{font-size:16px;color:var(--muted);max-width:520px;margin:0 auto;line-height:1.65;}

.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;margin-top:22px;box-shadow:0 2px 10px rgba(60,40,20,.04);}
.card h3{margin:0 0 4px;font-size:17px;font-weight:700;letter-spacing:-.01em;color:var(--ink);}
.card .sub{font-size:13.5px;color:var(--muted);margin:0 0 16px;}

div[data-testid="stFileUploader"],div[data-testid="stCameraInput"]{margin-top:6px;}
div[data-testid="stButton"] button{background:var(--green)!important;color:#fff!important;border:none!important;border-radius:11px!important;padding:13px 22px!important;font-size:15px!important;font-weight:700!important;transition:opacity .15s;}
div[data-testid="stButton"] button:hover{opacity:.88!important;}
div[data-testid="stButton"] button:disabled{opacity:.45!important;}

.ing{display:inline-flex;align-items:center;gap:7px;background:#F3EFE6;border:1px solid var(--line);border-radius:999px;padding:7px 14px;margin:0 8px 10px 0;font-size:13.5px;font-weight:600;color:var(--ink);}
.ing small{color:var(--muted);font-weight:500;}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px;}
.badge{font-size:12px;font-weight:700;background:#F3EFE6;border:1px solid var(--line);border-radius:7px;padding:5px 11px;color:var(--ink);}
.badge.kcal{background:#FFF4E8;border-color:#F0DCC2;color:#9A5B1E;}
.badge.char{background:var(--green);border-color:var(--green);color:#fff;}

.have{font-size:13.5px;color:var(--ink);margin:6px 0;line-height:1.9;}
.have b{color:var(--green);}
.miss{font-size:13.5px;color:var(--ink);margin:6px 0;line-height:1.9;}
.miss b{color:var(--amber);}
.steps{margin:14px 0 0;padding:0;list-style:none;counter-reset:s;}
.steps li{counter-increment:s;font-size:14.5px;line-height:1.65;color:#3d382f;margin-bottom:10px;padding-left:38px;position:relative;}
.steps li::before{content:counter(s);position:absolute;left:0;top:1px;width:24px;height:24px;border-radius:50%;background:var(--green);color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;}
.tip{background:#FFF8EC;border:1px dashed #E4C88F;border-radius:10px;padding:12px 16px;font-size:13.5px;color:#7a5a22;margin-top:16px;}

.shop{background:var(--ink);color:#fff;border-radius:16px;padding:26px;margin-top:26px;}
.shop h3{color:#fff;margin:0 0 12px;font-size:17px;}
.shop ul{margin:0;padding-left:20px;}
.shop li{font-size:14.5px;line-height:2;color:rgba(255,255,255,.85);}
.shop li span{color:rgba(255,255,255,.55);font-size:13px;}

.sec-t{font-size:13px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:34px 0 4px;}
.stAlert{border-radius:10px!important;}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;background:#fff!important;}
@media(max-width:640px){.card{padding:18px;}.hero{padding-top:36px;}}
</style>
"""

st.set_page_config(page_title="FridgeSnap — snap it, cook it", layout="centered")
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="brand-mark">🥑</div><div class="brand">FridgeSnap</div></div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Snap your fridge.<br><em>Cook what\'s inside.</em></h1>'
            '<p>Take a photo of your fridge, confirm what the AI spots, pick a cuisine — '
            'get three recipes built from <b>your</b> ingredients, plus a shopping list for what\'s missing.</p></div>',
            unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

# ---------- session state ----------
for k, v in [("photo", None), ("detected", None), ("recipes", None), ("confirmed", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- 1. photo ----------
st.markdown('<div class="card"><h3>1 · Show me your fridge</h3>'
            '<p class="sub">Upload a photo or snap one right now — brighter and fuller is better.</p></div>',
            unsafe_allow_html=True)
tab_up, tab_cam = st.tabs(["📤 Upload", "📷 Camera"])
with tab_up:
    up = st.file_uploader("fridge photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if up:
        st.session_state.photo = (up.getvalue(), up.type)
with tab_cam:
    cam = st.camera_input("take a photo", label_visibility="collapsed")
    if cam:
        st.session_state.photo = (cam.getvalue(), cam.type)

if st.session_state.photo:
    st.image(st.session_state.photo[0], caption="Your fridge", use_container_width=True)
    if st.button("🔍 Identify ingredients", use_container_width=True):
        with st.spinner("Peeking inside your fridge…"):
            raw, mime = st.session_state.photo
            small, mime2 = prep_image(raw, mime)
            items, err = detect_ingredients(small, mime2)
        if err:
            st.error(err)
            st.session_state.detected = None
        else:
            st.session_state.detected = items
            st.session_state.confirmed = [it["name"] for it in items if it.get("confidence") != "low"]
            st.session_state.recipes = None
            st.rerun()

# ---------- 2. confirm ingredients ----------
if st.session_state.detected:
    st.markdown('<div class="card"><h3>2 · Confirm what\'s in there</h3>'
                '<p class="sub">Uncheck anything the AI got wrong, add what it missed.</p></div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    checked = []
    for i, it in enumerate(st.session_state.detected):
        name = it["name"]
        label = f"{name}" + (f"  ·  {it['amount']}" if it.get("amount") else "")
        with cols[i % 2]:
            if st.checkbox(label, value=name in st.session_state.confirmed, key=f"ing_{i}"):
                checked.append(name)
    extra = st.text_input("Add missing items (comma separated)",
                          placeholder="e.g. rice, soy sauce, garlic")
    if extra.strip():
        checked += [x.strip() for x in extra.split(",") if x.strip()]
    # de-dupe, keep order
    seen, confirmed = set(), []
    for x in checked:
        if x.lower() not in seen:
            seen.add(x.lower())
            confirmed.append(x)
    st.session_state.confirmed = confirmed
    if confirmed:
        st.markdown("<div style='margin-top:6px'>" + "".join(
            f'<span class="ing">{html.escape(c)}</span>' for c in confirmed) + "</div>",
            unsafe_allow_html=True)

    # ---------- 3. options ----------
    st.markdown('<div class="card"><h3>3 · How do you want it?</h3>'
                '<p class="sub">Tune it to your craving and your goals.</p></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cuisine = st.selectbox("Cuisine", CUISINES)
        diet = st.selectbox("Diet", DIETS)
    with c2:
        max_time = st.slider("Max cooking time (min)", 10, 90, 30, step=5)
        servings = st.number_input("Servings", 1, 8, 2)
    use_kcal = st.checkbox("Set a calorie target per serving")
    kcal_target = st.slider("Max kcal per serving", 200, 1200, 500, step=50) if use_kcal else 0

    if st.button("🍳 Get my recipes", type="primary", use_container_width=True, disabled=not confirmed):
        with st.spinner("Cooking up three recipes…"):
            try:
                st.session_state.recipes = generate_recipes(
                    confirmed, cuisine, diet, max_time, servings, kcal_target)
            except Exception as e:
                st.error(f"Couldn't generate recipes: {html.escape(str(e))[:250]}")
                st.session_state.recipes = None

# ---------- 4. recipes ----------
if st.session_state.recipes:
    st.markdown('<div class="sec-t">Your recipes</div>', unsafe_allow_html=True)
    all_missing = {}
    for r in st.session_state.recipes:
        have_n = len(r.get("uses", []))
        st.markdown(f"""<div class="card">
          <div class="badges"><span class="badge char">{html.escape(str(r.get('character', '')))}</span>
          <span class="badge">⏱ {r.get('time_min', '?')} min</span>
          <span class="badge kcal">{r.get('calories_est', '?')} kcal/serv</span>
          <span class="badge">{html.escape(str(r.get('difficulty', '')))}</span></div>
          <h3 style="font-size:20px;margin-top:10px">{html.escape(r.get('title', ''))}</h3>
          <p class="have"><b>✓ From your fridge ({have_n}):</b> {html.escape(", ".join(r.get('uses', [])) or "—")}</p>""",
            unsafe_allow_html=True)
        miss = r.get("missing", [])
        if miss:
            miss_html = "; ".join(
                f"<b>{html.escape(str(m.get('item', '')))}</b>"
                + (f" <span style='color:var(--muted)'>(or {html.escape(str(m['swap']))})</span>" if m.get("swap") else "")
                for m in miss if isinstance(m, dict))
            st.markdown(f'<p class="miss"><b>＋ You\'ll need:</b> {miss_html}</p>', unsafe_allow_html=True)
            for m in miss:
                if isinstance(m, dict) and m.get("item"):
                    all_missing[m["item"]] = m.get("swap", "")
        steps_html = "".join(f"<li>{html.escape(str(s))}</li>" for s in r.get("steps", []))
        st.markdown(f'<ol class="steps">{steps_html}</ol>', unsafe_allow_html=True)
        if r.get("tip"):
            st.markdown(f'<div class="tip">💡 {html.escape(str(r["tip"]))}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if all_missing:
        lis = "".join(
            f"<li>{html.escape(k)}" + (f" <span>— or use {html.escape(v)}</span>" if v else "") + "</li>"
            for k, v in all_missing.items())
        st.markdown(f'<div class="shop"><h3>🛒 Shopping list</h3><ul>{lis}</ul></div>',
                    unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#a89c8a;font-size:12.5px;margin-top:56px">'
            'FridgeSnap — snap it, cook it. Estimates only, not medical nutrition advice.</div>',
            unsafe_allow_html=True)
