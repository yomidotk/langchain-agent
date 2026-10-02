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
# Vision: DashScope OpenAI-compatible endpoint (Alibaba key). qwen-vl-max reads
# fridge photos well. Alternatives: qwen-vl-plus, qwen2.5-vl-72b-instruct.
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
    """Returns (ingredients_list, error_str). Never raises — every failure path
    returns a clear, actionable message instead of crashing the page."""
    try:
        b64 = base64.b64encode(img_bytes).decode()
        prompt = """You are a precise kitchen assistant. Look at this fridge/kitchen photo and list every identifiable food ingredient or item you can see.

Return STRICT JSON: {"ingredients": [{"name": "eggs", "amount": "about 6", "confidence": "high"}, {"name": "milk", "amount": "half bottle", "confidence": "medium"}]}

Rules:
- Only list what you can actually SEE. Never guess what's behind closed doors or in opaque containers.
- Names in plain English ("chicken breast", not "chx brst").
- amount: short visible estimate ("3", "a bunch", "half jar"). Use "" if unclear.
- confidence: high/medium/low.
- Merge obvious duplicates (two milk bottles -> one entry "2 bottles").
- If this is not a fridge/kitchen/food photo at all, return {"ingredients": [], "not_food": true}.
""" + STRICT_JSON
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
            # surface DashScope's real reason (quota? key? model?) — not a bare HTTP code
            detail = r.text[:300]
            try:
                ej = r.json()
                detail = str(ej.get("message") or ej.get("code") or detail)
            except Exception:
                pass
            return None, f"vision failed (HTTP {r.status_code}): {detail[:220]}"
        data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
        if isinstance(content, list):  # some VL models return content blocks
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        clean = re.sub(r'```(?:json)?', '', content).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        out = json.loads(m.group(0) if m else clean)
        if out.get("not_food"):
            return None, "that doesn't look like a fridge or kitchen photo — try another one"
        items = [it for it in out.get("ingredients", []) if isinstance(it, dict) and it.get("name")]
        if not items:
            return None, "couldn't spot any ingredients — try a clearer, brighter photo"
        return items, ""
    except Exception as e:
        return None, f"vision error: {str(e)[:220]}"

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
:root{--ink:#1d1a16;--paper:#FFFCF7;--card:#fff;--line:#EDE4D6;--muted:#8a7f70;
--green:#2E7D4F;--green-d:#1f5c38;--amber:#C2703D;--cream:#F6F1E7;}
*{font-family:'Inter',-apple-system,'Segoe UI',sans-serif!important;}
.stApp{background:var(--paper)!important;}
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
.block-container{max-width:880px!important;padding:0 20px 90px!important;}
section[data-testid="stSidebar"]{display:none!important;}

/* nav */
.nav{display:flex;align-items:center;gap:10px;padding:20px 0 6px;}
.brand-mark{width:34px;height:34px;border-radius:10px;background:var(--green);display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 3px 8px rgba(46,125,79,.3);}
.brand{font-weight:800;font-size:19px;letter-spacing:-.02em;color:var(--ink);}
.brand small{font-weight:500;color:var(--muted);font-size:12px;margin-left:6px;}

/* stepper */
.stepper{display:flex;gap:6px;margin:18px 0 4px;}
.step{flex:1;text-align:center;font-size:12px;font-weight:700;color:var(--muted);padding:10px 4px;border-radius:10px;background:var(--cream);letter-spacing:.02em;}
.step.on{background:var(--green);color:#fff;}
.step.done{background:#E3EFE7;color:var(--green-d);}

/* hero */
.hero{text-align:center;padding:34px 8px 6px;}
.hero h1{font-size:clamp(2rem,6vw,3.1rem);font-weight:800;letter-spacing:-.04em;line-height:1.06;margin:0 0 12px;color:var(--ink);}
.hero h1 em{font-style:normal;color:var(--green);}
.hero p{font-size:15.5px;color:var(--muted);max-width:540px;margin:0 auto;line-height:1.65;}

/* cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:26px;margin-top:20px;box-shadow:0 3px 14px rgba(60,40,20,.05);}
.card .k{font-size:11.5px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--green);margin-bottom:6px;}
.card h3{margin:0 0 4px;font-size:18px;font-weight:700;letter-spacing:-.01em;color:var(--ink);}
.card .sub{font-size:13.5px;color:var(--muted);margin:0 0 14px;line-height:1.55;}

/* buttons */
div[data-testid="stButton"] button{background:var(--green)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:13px 22px!important;font-size:15px!important;font-weight:700!important;box-shadow:0 4px 12px rgba(46,125,79,.28)!important;transition:transform .12s,box-shadow .12s!important;}
div[data-testid="stButton"] button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 16px rgba(46,125,79,.34)!important;}
div[data-testid="stButton"] button:disabled{opacity:.45!important;transform:none!important;box-shadow:none!important;}
div[data-testid="stButton"] button[kind="secondary"]{background:#fff!important;color:var(--ink)!important;border:1.5px solid var(--line)!important;box-shadow:none!important;}

/* tabs */
div[data-testid="stTabs"] button{font-weight:600!important;}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:8px;}
div[data-testid="stTabs"] [data-baseweb="tab"]{border:1.5px solid var(--line)!important;border-radius:10px!important;padding:8px 18px!important;background:#fff!important;}
div[data-testid="stTabs"] [aria-selected="true"]{border-color:var(--green)!important;color:var(--green-d)!important;background:#EFF6F1!important;}

/* pills */
div[data-testid="stPills"] button{border-radius:999px!important;font-weight:600!important;border:1.5px solid var(--line)!important;}
div[data-testid="stPills"] button[aria-pressed="true"]{background:var(--green)!important;border-color:var(--green)!important;color:#fff!important;}

/* uploader */
div[data-testid="stFileUploader"]{border:2px dashed #D8CCB8!important;border-radius:14px!important;padding:18px!important;background:#FFFEFB!important;}
div[data-testid="stFileUploader"]:hover{border-color:var(--green)!important;}

/* photo */
.photo{border-radius:14px;overflow:hidden;border:1px solid var(--line);margin:6px 0 4px;}
.photo img{width:100%;display:block;}

/* badges + recipes */
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 8px;}
.badge{font-size:12px;font-weight:700;background:var(--cream);border:1px solid var(--line);border-radius:8px;padding:5px 11px;color:var(--ink);}
.badge.kcal{background:#FFF4E8;border-color:#F0DCC2;color:#9A5B1E;}
.badge.char{background:var(--green);border-color:var(--green);color:#fff;letter-spacing:.06em;font-size:11px;}
.useup{font-size:12.5px;font-weight:700;color:var(--green-d);background:#EAF3EC;border-radius:8px;padding:5px 11px;display:inline-block;margin:8px 0 2px;}
.rtitle{font-size:21px;font-weight:800;letter-spacing:-.02em;color:var(--ink);margin:10px 0 4px;}
.have{font-size:13.5px;color:var(--ink);margin:8px 0;line-height:1.9;}
.have b{color:var(--green);}
.miss{font-size:13.5px;color:var(--ink);margin:8px 0;line-height:1.9;}
.miss b{color:var(--amber);}
.steps{margin:14px 0 0;padding:0;list-style:none;counter-reset:s;}
.steps li{counter-increment:s;font-size:14.5px;line-height:1.65;color:#3d382f;margin-bottom:10px;padding-left:40px;position:relative;}
.steps li::before{content:counter(s);position:absolute;left:0;top:1px;width:25px;height:25px;border-radius:50%;background:var(--green);color:#fff;font-size:12.5px;font-weight:700;display:flex;align-items:center;justify-content:center;}
.tip{background:#FFF8EC;border:1px dashed #E4C88F;border-radius:10px;padding:12px 16px;font-size:13.5px;color:#7a5a22;margin-top:16px;line-height:1.6;}

.shop{background:var(--ink);color:#fff;border-radius:18px;padding:28px;margin-top:26px;box-shadow:0 6px 20px rgba(29,26,22,.18);}
.shop h3{color:#fff;margin:0 0 6px;font-size:18px;font-weight:700;}
.shop .sub{color:rgba(255,255,255,.6);font-size:13px;margin:0 0 12px;}
.shop ul{margin:0;padding-left:20px;}
.shop li{font-size:14.5px;line-height:2.1;color:rgba(255,255,255,.88);}
.shop li span{color:rgba(255,255,255,.55);font-size:13px;}

.sec-t{font-size:13px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:38px 0 2px;}
.opt-label{font-size:12.5px;font-weight:700;color:var(--ink);margin:16px 0 8px;letter-spacing:.02em;}
.stAlert{border-radius:12px!important;}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;background:#fff!important;margin-top:12px;}
.footer{text-align:center;color:#a89c8a;font-size:12.5px;margin-top:60px;line-height:1.7;}
@media(max-width:640px){.card{padding:20px 18px;}.hero{padding-top:26px;}.stepper{gap:4px;}.step{font-size:10.5px;padding:8px 2px;}}
</style>
"""

st.set_page_config(page_title="FridgeSnap — snap it, cook it", layout="centered")
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="brand-mark">🥑</div>'
            '<div class="brand">FridgeSnap <small>snap it, cook it</small></div></div>',
            unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

# ---------- session state ----------
for k, v in [("photo", None), ("detected", None), ("recipes", None), ("confirmed", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- stepper ----------
stage = 1
if st.session_state.detected:
    stage = 2
if st.session_state.recipes:
    stage = 3
steps = ["📸 Photo", "🥬 Ingredients", "🍳 Recipes"]
st.markdown('<div class="stepper">' + "".join(
    f'<div class="step {"on" if i + 1 == stage else ("done" if i + 1 < stage else "")}">{s}</div>'
    for i, s in enumerate(steps)) + "</div>", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>Snap your fridge.<br><em>Cook what\'s inside.</em></h1>'
            '<p>Show me what you\'ve got — I\'ll spot the ingredients, you pick a cuisine, '
            'and I\'ll build three recipes around <b>your</b> food. Nothing wasted.</p></div>',
            unsafe_allow_html=True)

# ---------- 1. photo ----------
st.markdown('<div class="card"><div class="k">Step 1</div><h3>Show me your fridge</h3>'
            '<p class="sub">Upload a photo or snap one now. Bright, straight-on shots work best.</p></div>',
            unsafe_allow_html=True)
tab_up, tab_cam = st.tabs(["📤 Upload photo", "📷 Take photo"])
with tab_up:
    up = st.file_uploader("fridge photo", type=["jpg", "jpeg", "png", "webp"],
                          label_visibility="collapsed")
    if up:
        st.session_state.photo = (up.getvalue(), up.type)
        st.session_state.detected = None
        st.session_state.recipes = None
with tab_cam:
    cam = st.camera_input("take a photo", label_visibility="collapsed")
    if cam:
        st.session_state.photo = (cam.getvalue(), cam.type)
        st.session_state.detected = None
        st.session_state.recipes = None

if st.session_state.photo:
    st.markdown(f'<div class="photo"><img src="data:{st.session_state.photo[1] or "image/jpeg"};base64,'
                f'{base64.b64encode(st.session_state.photo[0]).decode()}"></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Identify ingredients", use_container_width=True):
            with st.spinner("Peeking inside your fridge…"):
                raw, mime = st.session_state.photo
                small, mime2 = prep_image(raw, mime)
                items, err = detect_ingredients(small, mime2)  # never raises
            if err:
                st.error(err)
            else:
                st.session_state.detected = items
                st.session_state.confirmed = [it["name"] for it in items if it.get("confidence") != "low"]
                st.session_state.recipes = None
                st.rerun()
    with c2:
        if st.button("🔄 New photo", use_container_width=True, type="secondary"):
            for k in ("photo", "detected", "recipes", "confirmed"):
                st.session_state[k] = None if k != "confirmed" else []
            st.rerun()

with st.expander("⌨️ No photo? Type your ingredients instead"):
    manual = st.text_area("one per line or comma-separated", placeholder="eggs, chicken breast, tomatoes…",
                          label_visibility="collapsed")
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
                '<p class="sub">Tap to toggle — deselect anything the AI got wrong.</p></div>',
                unsafe_allow_html=True)
    label_to_name = {}
    pill_labels = []
    for it in st.session_state.detected:
        lbl = it["name"] + (f" · {it['amount']}" if it.get("amount") else "")
        label_to_name[lbl] = it["name"]
        pill_labels.append(lbl)
    picked = st.pills("ingredients", pill_labels,
                      selection_mode="multi",
                      default=[l for l, n in label_to_name.items() if n in st.session_state.confirmed],
                      label_visibility="collapsed")
    confirmed = [label_to_name[l] for l in (picked or [])]
    extra = st.text_input("Add what it missed", placeholder="e.g. rice, soy sauce + Enter",
                          label_visibility="collapsed")
    if extra.strip():
        for x in re.split(r"[,\n]+", extra):
            x = x.strip()
            if x and x.lower() not in {c.lower() for c in confirmed}:
                confirmed.append(x)
    st.session_state.confirmed = confirmed

    # ---------- 3. options ----------
    st.markdown('<div class="card"><div class="k">Step 3</div><h3>How do you want it?</h3>'
                '<p class="sub">Dial it in — then hit the big green button.</p></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="opt-label">Cuisine</div>', unsafe_allow_html=True)
    cuisine = st.pills("cuisine", CUISINES, default="Anything", label_visibility="collapsed") or "Anything"
    st.markdown('<div class="opt-label">Diet</div>', unsafe_allow_html=True)
    diet = st.pills("diet", DIETS, default="No restriction", label_visibility="collapsed") or "No restriction"
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="opt-label">Max time</div>', unsafe_allow_html=True)
        max_time = st.slider("max time", 10, 90, 30, step=5, format="%d min", label_visibility="collapsed")
    with t2:
        st.markdown('<div class="opt-label">Servings</div>', unsafe_allow_html=True)
        servings = st.pills("servings", [1, 2, 3, 4, 5, 6], default=2, label_visibility="collapsed") or 2
    use_kcal = st.checkbox("🎯 Set a calorie target per serving")
    kcal_target = (st.slider("kcal target", 200, 1200, 500, step=50, format="%d kcal",
                             label_visibility="collapsed") if use_kcal else 0)

    if st.button("🍳 Get my recipes", type="primary", use_container_width=True, disabled=not confirmed):
        with st.spinner("Cooking up three recipes…"):
            try:
                st.session_state.recipes = generate_recipes(
                    confirmed, cuisine, diet, max_time, servings, kcal_target)
            except Exception as e:
                st.error(f"Couldn't generate recipes: {html.escape(str(e))[:250]}")
                st.session_state.recipes = None
        st.rerun()

# ---------- 4. recipes ----------
if st.session_state.recipes:
    st.markdown('<div class="sec-t">Your recipes</div>', unsafe_allow_html=True)
    all_missing = {}
    total_have = len(st.session_state.confirmed) or 1
    for r in st.session_state.recipes:
        uses = r.get("uses", []) or []
        pct = round(100 * len(uses) / total_have)
        st.markdown(f"""<div class="card">
          <div class="badges"><span class="badge char">{html.escape(str(r.get('character', '')))}</span>
          <span class="badge">⏱ {r.get('time_min', '?')} min</span>
          <span class="badge kcal">{r.get('calories_est', '?')} kcal/serv</span>
          <span class="badge">{html.escape(str(r.get('difficulty', '')))}</span></div>
          <div class="rtitle">{html.escape(r.get('title', ''))}</div>
          <span class="useup">Uses {len(uses)} of your {total_have} ingredients ({pct}%)</span>
          <p class="have"><b>✓ From your fridge:</b> {html.escape(", ".join(uses) or "—")}</p>""",
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
        st.markdown(f'<div class="shop"><h3>🛒 Shopping list</h3>'
                    f'<p class="sub">Everything missing, across all three recipes.</p><ul>{lis}</ul></div>',
                    unsafe_allow_html=True)

st.markdown('<div class="footer">FridgeSnap — snap it, cook it.<br>'
            'Calorie estimates are rough guides, not medical advice.</div>',
            unsafe_allow_html=True)
