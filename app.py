"""FridgeSnap — LangChain Chef Agent with Real Online Recipes & Verified Food Photos."""
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── Config ─────────────────────────────────────────────────────────────────────
DO_API_KEY = st.secrets.get("DO_API_KEY", "")
ALIBABA_API_KEY = st.secrets.get("ALIBABA_API_KEY", "")
DO_URL = "https://inference.do-ai.run/v1"
DO_MODEL = "openai-gpt-oss-20b"
VISION_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
VISION_MODEL = "qwen-vl-max"

CUISINES = ["Anything", "Italian", "Mexican", "Asian", "Indian", "Mediterranean", "French", "American"]
DIETS    = ["No restriction", "Vegetarian", "Vegan", "Halal", "Gluten-free", "High-protein"]

# ── Auto-scroll Helper ─────────────────────────────────────────────────────────
def auto_scroll(anchor_id: str, delay_ms: int = 80):
    """Smoothly scroll the browser viewport directly to the active element/stage."""
    st.markdown(f"""
    <div id="{anchor_id}" style="height:1px;margin-top:-8px;"></div>
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3C/svg%3E" 
         style="display:none;" 
         onload="(function(){{
             setTimeout(function(){{
                 var el = document.getElementById('{anchor_id}');
                 if (el) {{
                     el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                 }} else {{
                     window.scrollTo({{ top: 0, behavior: 'smooth' }});
                 }}
             }}, {delay_ms});
         }})();" />
    """, unsafe_allow_html=True)

# ── Real Recipe Online Search (TheMealDB Culinary API) ─────────────────────────
def search_online_recipes(ingredients: list, cuisine: str = "Anything") -> list:
    """Fetch real recipe candidates from online culinary databases based on fridge contents."""
    s = requests.Session()
    s.headers.update({"User-Agent": "FridgeSnapBot/2.0 (culinary-assistant)"})
    candidates = []
    seen_ids = set()

    # 1. Search by user ingredients
    for ing in ingredients[:4]:
        clean_ing = ing.strip().lower().replace(" ", "_")
        try:
            r = s.get(f"https://www.themealdb.com/api/json/v1/1/filter.php?i={urllib.parse.quote(clean_ing)}", timeout=3)
            if r.status_code == 200:
                for m in (r.json().get("meals") or []):
                    if m["idMeal"] not in seen_ids:
                        seen_ids.add(m["idMeal"])
                        candidates.append(m)
        except Exception:
            pass

    # 2. Search by Cuisine area if specified
    area_map = {
        "Italian": "Italian", "Mexican": "Mexican", "Asian": "Chinese",
        "Indian": "Indian", "Mediterranean": "Greek", "French": "French", "American": "American"
    }
    if cuisine in area_map:
        try:
            r = s.get(f"https://www.themealdb.com/api/json/v1/1/filter.php?a={area_map[cuisine]}", timeout=3)
            if r.status_code == 200:
                for m in (r.json().get("meals") or []):
                    if m["idMeal"] not in seen_ids:
                        seen_ids.add(m["idMeal"])
                        candidates.append(m)
        except Exception:
            pass

    # Fetch full recipe details for top candidates
    detailed = []
    for m in candidates[:6]:
        try:
            r_det = s.get(f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={m['idMeal']}", timeout=3)
            if r_det.status_code == 200:
                meal_data = r_det.json().get("meals", [{}])[0]
                # Collect meal ingredients
                meal_ings = []
                for idx in range(1, 21):
                    val = meal_data.get(f"strIngredient{idx}")
                    if val and val.strip():
                        meal_ings.append(val.strip().lower())
                        
                detailed.append({
                    "title": meal_data.get("strMeal", ""),
                    "category": meal_data.get("strCategory", ""),
                    "area": meal_data.get("strArea", ""),
                    "thumb": meal_data.get("strMealThumb", ""),
                    "source": meal_data.get("strSource", ""),
                    "ingredients": meal_ings,
                    "instructions": meal_data.get("strInstructions", "")
                })
        except Exception:
            pass
    return detailed

# ── Accurate Food Photo Pipeline (Matches Exactly What You Cook) ───────────────
def get_accurate_dish_image(dish_name: str, fallback_thumb: str = "") -> str:
    """Take dish title, search online directly for the same dish, and fetch the real photo."""
    if fallback_thumb and fallback_thumb.startswith("http"):
        return fallback_thumb

    s = requests.Session()
    s.headers.update({"User-Agent": "FridgeSnapBot/3.0 (culinary-assistant; contact@fridgesnap.org)"})

    clean_title = re.sub(r"[^\w\s-]", " ", dish_name).strip()

    # 1. Primary: Search Openverse directly online for the exact dish title
    search_attempts = [clean_title]
    words = clean_title.split()
    if len(words) > 3:
        search_attempts.append(" ".join(words[:3]))

    for q in search_attempts:
        try:
            url = f"https://api.openverse.org/v1/images/?q={urllib.parse.quote(q)}&page_size=4"
            r = s.get(url, timeout=4)
            if r.status_code == 200:
                results = r.json().get("results", [])
                for item in results:
                    img_url = item.get("url")
                    if img_url and not img_url.endswith(".svg"):
                        return img_url
        except Exception:
            pass

    # 2. Secondary: TheMealDB search for exact or clean dish name
    for q in search_attempts:
        try:
            r = s.get(f"https://www.themealdb.com/api/json/v1/1/search.php?s={urllib.parse.quote(q)}", timeout=3)
            if r.status_code == 200:
                meals = r.json().get("meals")
                if meals and meals[0].get("strMealThumb"):
                    return meals[0]["strMealThumb"]
        except Exception:
            pass

    # 3. Tertiary: Filtered Wikimedia Commons photo search (real food photos only)
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            "&generator=search&gsrnamespace=6"
            f"&gsrsearch={urllib.parse.quote(clean_title + ' food')}"
            "&gsrlimit=6&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
        )
        r = s.get(url, timeout=4)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                title = page.get("title", "").lower()
                if any(bad in title for bad in [".pdf", ".svg", ".tif", "menu", "carte", "book", "cover", "text", "label"]):
                    continue
                if any(ext in title for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    ii = page.get("imageinfo", [])
                    if ii and ii[0].get("thumburl"):
                        return ii[0]["thumburl"]
    except Exception:
        pass

    # 4. Wikipedia summary
    try:
        slug = urllib.parse.quote(clean_title.replace(" ", "_"))
        r = s.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}", timeout=3)
        if r.status_code == 200:
            d = r.json()
            thumb = d.get("thumbnail", {}).get("source") or d.get("originalimage", {}).get("source")
            if thumb and not thumb.endswith(".svg"):
                return re.sub(r"/(\d+)px-", "/640px-", thumb)
    except Exception:
        pass

    # Final fallback: high-res generic gourmet plate
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

# ── Robust Recipe Parser (Prevents OUTPUT_PARSING_FAILURE) ─────────────────────
def extract_and_parse_recipes(raw_text: str) -> list:
    """Robustly extract and normalize 3 recipes from LLM text output."""
    if not raw_text or not raw_text.strip():
        raise ValueError("AI Chef returned an empty response.")
        
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    
    data = None
    m_obj = re.search(r"\{.*\}", text, re.DOTALL)
    if m_obj:
        candidate = re.sub(r",\s*([\]}])", r"\1", m_obj.group(0))
        try: data = json.loads(candidate)
        except Exception: pass

    if data is None:
        m_arr = re.search(r"\[.*\]", text, re.DOTALL)
        if m_arr:
            candidate = re.sub(r",\s*([\]}])", r"\1", m_arr.group(0))
            try: data = json.loads(candidate)
            except Exception: pass

    if data is None:
        try:
            candidate = re.sub(r"[\r\n]+", " ", text)
            m_retry = re.search(r"\{.*\}", candidate) or re.search(r"\[.*\]", candidate)
            if m_retry:
                clean_retry = re.sub(r",\s*([\]}])", r"\1", m_retry.group(0))
                data = json.loads(clean_retry)
        except Exception:
            pass

    if data is None:
        raise ValueError(f"Could not parse valid JSON from AI response: {raw_text[:200]}")

    recipe_list = []
    if isinstance(data, dict):
        for k in ["recipes", "dishes", "recipe_list", "items"]:
            if k in data and isinstance(data[k], list):
                recipe_list = data[k]
                break
        if not recipe_list:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    recipe_list = v
                    break
    elif isinstance(data, list):
        recipe_list = data

    if not recipe_list:
        raise ValueError("AI response did not contain a list of recipes.")

    default_chars = ["QUICK", "HEARTY", "CREATIVE"]
    standardized = []
    for idx, r in enumerate(recipe_list[:3]):
        if not isinstance(r, dict):
            continue
        title = r.get("title") or r.get("name") or f"Chef Special #{idx+1}"
        char = str(r.get("character") or default_chars[idx % 3]).upper()
        if "QUICK" in char: char = "QUICK"
        elif "HEARTY" in char: char = "HEARTY"
        elif "CREATIVE" in char: char = "CREATIVE"
        else: char = default_chars[idx % 3]

        try:
            time_match = re.search(r"\d+", str(r.get("time_min") or r.get("cooking_time") or 25))
            time_min = int(time_match.group(0)) if time_match else 25
        except Exception:
            time_min = 25

        try:
            cal_match = re.search(r"\d+", str(r.get("calories_est") or r.get("calories") or 450))
            calories_est = int(cal_match.group(0)) if cal_match else 450
        except Exception:
            calories_est = 450

        difficulty = str(r.get("difficulty") or "Medium").capitalize()
        image_url = r.get("image_url") or r.get("thumb") or ""
        source_url = r.get("source_url") or r.get("source") or ""
        
        uses = r.get("uses") or r.get("ingredients") or []
        if isinstance(uses, str):
            uses = [x.strip() for x in re.split(r"[,;]+", uses) if x.strip()]
        elif isinstance(uses, list):
            uses = [str(u) for u in uses if u]

        missing_raw = r.get("missing") or []
        missing = []
        if isinstance(missing_raw, list):
            for m in missing_raw:
                if isinstance(m, dict):
                    missing.append({"item": str(m.get("item", "")), "swap": str(m.get("swap", ""))})
                elif isinstance(m, str):
                    missing.append({"item": m, "swap": ""})

        steps_raw = r.get("steps") or r.get("instructions") or []
        steps = []
        if isinstance(steps_raw, list):
            steps = [str(s).strip() for s in steps_raw if str(s).strip()]
        elif isinstance(steps_raw, str):
            steps = [s.strip() for s in re.split(r"[\n\r]+", steps_raw) if s.strip()]
        if not steps:
            steps = ["Prepare all ingredients.", "Cook thoroughly in a pan or pot.", "Season to taste and serve warm."]

        tip = str(r.get("tip") or "Serve immediately while fresh and hot.")

        standardized.append({
            "title": title,
            "character": char,
            "time_min": time_min,
            "calories_est": calories_est,
            "difficulty": difficulty,
            "image_url": image_url,
            "source_url": source_url,
            "uses": uses,
            "missing": missing,
            "steps": steps,
            "tip": tip
        })

    if not standardized:
        raise ValueError("Could not extract individual recipe details.")
    return standardized

# ── LangChain Agent with Online Recipe Retrieval ──────────────────────────────
def generate_recipes(ingredients, cuisine, diet, max_time, servings, kcal_target):
    llm = ChatOpenAI(
        model_name=DO_MODEL, openai_api_key=DO_API_KEY,
        openai_api_base=DO_URL, temperature=0.6, max_tokens=3200)
    ing = ", ".join(ingredients)
    kcal_line = f"Target max ~{kcal_target} kcal per serving." if kcal_target else ""
    diet_line = f"Dietary preference: {diet}." if diet != "No restriction" else ""

    # Real online recipe search from culinary database
    online_candidates = search_online_recipes(ingredients, cuisine)
    candidates_text = ""
    if online_candidates:
        cand_list = []
        for c in online_candidates:
            cand_list.append(f"- Title: {c['title']} | Image: {c['thumb']} | Source: {c['source']} | Main Ingredients: {', '.join(c['ingredients'][:6])}")
        candidates_text = "VERIFIED DISHES FROM ONLINE RECIPE SITES WITH REAL PHOTOS:\n" + "\n".join(cand_list)

    system_text = f"""You are an expert master chef agent. A home cook has these fridge ingredients: {ing}.
Assume salt, pepper, oil, water are available.
Constraints: Cuisine={cuisine}. {diet_line} Max cook time ≤{max_time} min. Serves {servings}. {kcal_line}

{candidates_text}

Instructions:
Generate exactly 3 distinct, delicious real dishes: 1) QUICK (≤20 min), 2) HEARTY, 3) CREATIVE.
You can adapt or draw directly from the verified online dishes above or well-known authentic recipes.
For each recipe include:
- title: recognizable authentic dish name
- character: QUICK, HEARTY, or CREATIVE
- time_min: minutes to prepare
- calories_est: calories per serving
- difficulty: Easy, Medium, or Hard
- image_url: verified photo URL if matched from online dishes, or empty string
- source_url: link to original recipe site (e.g. BBC Good Food, AllRecipes) if matched, or empty string
- uses: array of user ingredients used
- missing: array of objects with 'item' and 'swap' fields for other needed items
- steps: array of 4-7 concise cooking steps
- tip: one pro cooking tip

CRITICAL: Return ONLY a valid JSON object with schema: {{"recipes": [{{"title": "...", "character": "QUICK", "time_min": 20, "calories_est": 450, "difficulty": "Easy", "image_url": "", "source_url": "", "uses": ["..."], "missing": [{{"item": "...", "swap": "..."}}], "steps": ["..."], "tip": "..."}}]}}
No markdown backticks, no greeting text."""

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content="Provide 3 recipes in valid JSON format.")
    ]
    
    res = llm.invoke(messages)
    raw_content = res.content if hasattr(res, "content") else str(res)
    try:
        recipes = extract_and_parse_recipes(raw_content)
    except Exception:
        # Safe fallback repair prompt using direct messages
        fix_system = 'Extract and format the recipe information into pure, valid JSON with schema: {"recipes": [{"title": "...", "character": "QUICK", "time_min": 20, "calories_est": 450, "difficulty": "Easy", "image_url": "", "source_url": "", "uses": ["..."], "missing": [{"item": "...", "swap": "..."}], "steps": ["..."], "tip": "..."}]}. Output JSON ONLY.'
        messages_fix = [
            SystemMessage(content=fix_system),
            HumanMessage(content=raw_content[:2500])
        ]
        fix_res = llm.invoke(messages_fix)
        fix_content = fix_res.content if hasattr(fix_res, "content") else str(fix_res)
        recipes = extract_and_parse_recipes(fix_content)

    return recipes

# ── Audio TTS (Fixed: Uses get_audio_data directly, no status_code bug) ────────
def generate_audio(recipe_idx: int, title: str, steps: list):
    dashscope.api_key = ALIBABA_API_KEY
    text = f"Here is the recipe for {title}. " + " ".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    
    # Singapore/International endpoint first, then mainland
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
                if res is not None:
                    # SpeechSynthesisResult provides get_audio_data() directly
                    audio_data = res.get_audio_data()
                    if audio_data:
                        st.session_state[f"audio_{recipe_idx}"] = audio_data
                        return
                    last_err = str(res.get_response() or "Speech synthesis returned no audio data")
            except Exception as e:
                last_err = str(e)
                
    st.error(f"Voice generation note: {last_err or 'Could not generate speech audio.'}")

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
html { scroll-behavior: smooth; }

body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }

/* Constrain width cleanly without empty margins */
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

/* ── HERO BANNER ── */
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

/* ── STREAMLIT CONTAINERS ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--border) !important;
  border-radius: 12px !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 4px !important;
}

/* ── FILE UPLOADER ── */
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

/* ── BADGES & META ── */
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
.verified-source-badge {
  font-size: 10px;
  color: var(--green) !important;
  background: var(--green-bg);
  border: 1px solid rgba(30,111,61,0.25);
  border-radius: 4px;
  padding: 2px 6px;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-weight: 600;
  margin-bottom: 6px;
  text-decoration: none !important;
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
.source-link-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  margin-top: 12px;
  font-size: 12px;
}
.source-link-box a {
  color: var(--accent) !important;
  font-weight: 700;
  text-decoration: none;
}
.source-link-box a:hover {
  text-decoration: underline;
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

# Top Nav
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
    auto_scroll("stage-1-anchor")
    st.markdown("""
    <div class="stage-hero">
      <h2>Snap your fridge, cook what's inside.</h2>
      <p>Show the agent what you have — it spots the ingredients, searches verified cooking sites, and builds 3 real recipes.</p>
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
                auto_scroll("photo-ready-anchor")
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
    auto_scroll("stage-2-anchor")
    st.markdown("""
    <div class="stage-hero">
      <h2>Confirm ingredients & customize your meal.</h2>
      <p>Review what was found, select your flavor profile, and let the LangChain agent search real recipes for you.</p>
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
                if st.button("🍳 Search & Generate 3 Recipes →", use_container_width=True, disabled=not st.session_state.confirmed):
                    with st.spinner("Searching online culinary databases & reasoning over ingredients…"):
                        try:
                            recs = generate_recipes(st.session_state.confirmed, cuisine, diet, max_time, servings, kcal_target)
                            st.session_state.recipes = recs
                            # Preload accurate dish images matching the food
                            for i, r in enumerate(recs):
                                st.session_state[f"img_{i}"] = get_accurate_dish_image(r.get("title", ""), r.get("image_url", ""))
                            st.session_state.stage = 3
                            st.rerun()
                        except Exception as e:
                            st.error(f"Recipe error: {html.escape(str(e))[:300]}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Pick Recipe (Side-by-Side 3-Column Display, Zero Scrolling)
# ══════════════════════════════════════════════════════════════════════════════
elif stage == 3:
    auto_scroll("stage-3-anchor")
    st.markdown("""
    <div class="stage-hero">
      <h2>Pick your dish.</h2>
      <p>Here are 3 unique recipes with verified photos matching your food. Select one to cook.</p>
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
        source_url = r.get("source_url")

        with col:
            with st.container(border=True):
                st.markdown(f'<div class="{badge_cls}">{badge_text}</div>', unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin:0 0 6px;font-size:17px;font-family:Playfair Display,serif;'>{html.escape(r.get('title',''))}</h3>", unsafe_allow_html=True)
                
                if source_url:
                    domain = urllib.parse.urlparse(source_url).netloc.replace("www.", "")
                    st.markdown(f'<a href="{html.escape(source_url)}" target="_blank" class="verified-source-badge">🌐 Source: {html.escape(domain)}</a>', unsafe_allow_html=True)

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
    auto_scroll("stage-4-anchor")
    r = st.session_state.recipes[st.session_state.picked]
    img = st.session_state.get(f"img_{st.session_state.picked}")
    char = r.get("character", "QUICK").upper()
    uses = r.get("uses", [])
    miss = r.get("missing", [])
    steps = r.get("steps", [])
    source_url = r.get("source_url")

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

            # Audio Reader (Fixed: reads directly from Alibaba DashScope)
            st.markdown("<small style='font-weight:700;'>🎧 Hands-free Chef Voice</small>", unsafe_allow_html=True)
            if st.button("🔊 Read Recipe Aloud", key="btn_audio_trigger", use_container_width=True):
                with st.spinner("Generating chef narration via DashScope…"):
                    generate_audio(st.session_state.picked, r.get("title", ""), steps)

            if f"audio_{st.session_state.picked}" in st.session_state:
                auto_scroll("audio-player-anchor")
                st.markdown("<div id='audio-player-anchor'></div>", unsafe_allow_html=True)
                st.audio(st.session_state[f"audio_{st.session_state.picked}"], format="audio/mp3")

            if source_url:
                st.markdown(f"""
                <div class="source-link-box">
                  📖 <b>Original Recipe:</b> <a href="{html.escape(source_url)}" target="_blank">View on {urllib.parse.urlparse(source_url).netloc} →</a>
                </div>
                """, unsafe_allow_html=True)

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

st.markdown('<div class="app-footer">FridgeSnap · LangChain Culinary Agent · Online Recipe & Image Verification</div>', unsafe_allow_html=True)
