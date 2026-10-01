import os
import re
import json
import html
import time
import asyncio
import base64
import tempfile
import requests
import edge_tts
import streamlit as st
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="HypeRepo — AI Marketing Agent", page_icon="⚡", layout="wide")

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)

DO_API_KEY = get_secret("DO_API_KEY")
ALIBABA_API_KEY = get_secret("ALIBABA_API_KEY")
DO_URL = "https://inference.do-ai.run/v1/responses"
DO_MODEL = "openai-gpt-oss-20b"

# ================= BACKEND PIPELINE (the brain) =================
# Voice settings — swap VOICE for any en-US voice name, e.g. "en-US-AvaMultilingualNeural"
VOICE = "en-US-AndrewMultilingualNeural"
VOICE_FALLBACK = "en-US-RogerNeural"

def _clean_spoken(text):
    """Strip URLs/links so the voiceover never reads 'https colon slash slash' aloud."""
    t = re.sub(r"\[([^\]]+)\]\(\s*https?://\S+\s*\)", r"\1", text or "")
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\bwww\.\S+", "", t)
    return re.sub(r"\s+", " ", t).strip()

def _to_ssml(text, voice):
    """Light SSML: a dramatic pause after the hook + breathing room between paragraphs,
    so the delivery has pacing instead of one flat robot run-on."""
    text = _clean_spoken(text)
    paras = [html.escape(" ".join(p.split())) for p in text.split("\n\n") if p.strip()]
    body = '<break time="600ms"/>'.join(paras)
    body = re.sub(r"([.!?])\s+", r'\1<break time="450ms"/>', body, count=1)
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
            f"<voice name='{voice}'><prosody rate='+0%'>{body}</prosody></voice></speak>")

def _synth(text, out_path, voice):
    try:
        ssml = _to_ssml(text, voice)
        async def _main():
            await edge_tts.Communicate(ssml, voice=voice).save(out_path)
        asyncio.run(_main())
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return "TTS finished but produced no audio file"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def tts_to_mp3(text, out_path, voice=VOICE):
    """Returns None on success, or an error string on failure. Never raises,
    so one bad voiceover can't silently kill the whole bundle."""
    if not (text or "").strip():
        return "empty script — nothing to narrate"
    err = _synth(text, out_path, voice)
    if err and voice != VOICE_FALLBACK:
        err = _synth(text, out_path, VOICE_FALLBACK)
    return err

def do_call(prompt, max_tokens, temperature, _tries=3):
    """Call the LLM and return parsed JSON. Retries up to _tries times with
    escalating token limits (the old single retry reused the same limit, so a
    truncated response failed twice identically). Raises RuntimeError with a
    clear message if the model keeps returning garbage — the caller decides
    how to surface it instead of crashing with a bare JSONDecodeError."""
    original = prompt
    last_err = None
    for attempt in range(_tries):
        tok = int(max_tokens * (1.6 ** attempt))
        if attempt > 0:
            time.sleep(2 * attempt)  # brief breather for transient flops
        response = requests.post(DO_URL,
            headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"},
            json={"model": DO_MODEL, "input": prompt, "max_output_tokens": tok,
                  "temperature": temperature, "stream": False}, timeout=180)
        response.raise_for_status()
        res_data = response.json()
        raw_content = ""
        if "choices" in res_data and len(res_data["choices"]) > 0:
            raw_content = res_data["choices"][0].get("message", {}).get("content", "")
        else:
            raw_content = res_data.get("output", "") or res_data.get("text", "")
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict) and block.get("role") == "assistant":
                    sub = block.get("content", [])
                    if isinstance(sub, list) and len(sub) > 0:
                        raw_content = sub[0].get("text", "")
        if not isinstance(raw_content, str):
            raw_content = json.dumps(raw_content)
        clean_text = re.sub(r'```(?:json)?', '', raw_content).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        final_json = match.group(0) if match else clean_text
        try:
            if not final_json.strip():
                raise ValueError(f"empty model output (response keys: {list(res_data.keys())[:6]})")
            return json.loads(final_json)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            if final_json.strip():
                prompt = ("Your previous response was not valid JSON (cut off or malformed). "
                          "Return the COMPLETE object again as valid JSON only, every field, full text, "
                          "no truncation, no markdown fences.\n\nBroken output:\n" + final_json[:6000])
            else:
                prompt = original + "\n\nRespond with valid JSON only, no other text."
    raise RuntimeError(f"AI returned invalid JSON after {_tries} tries ({last_err})")

class AgentState(TypedDict):
    repo_url: str; out_dir: str; readme_text: str; repo_context: str; code_context: str
    concept_brief: Dict; pitch_cards: List[Dict]; social_posts: List[Dict]

CODE_EXTS = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php", ".swift", ".kt")
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".next", "vendor", ".idea", ".vscode"}

# Files that look like code but tell us nothing about what the project DOES
JUNK_PARTS = (".config.", "config.", ".d.ts", ".pyi", ".min.js", ".min.css",
              "_test.", ".test.", ".spec.", "test_", "__tests__", "mock", "fixture")
ENTRY_HINTS = ("main", "index", "app", "cli", "server", "__main__")

def _is_junk(name):
    n = name.lower()
    return n.startswith("builtins") or any(p in n for p in JUNK_PARTS)

def _rank(name):
    n = name.lower()
    return (_is_junk(n), not any(h in n for h in ENTRY_HINTS), n)

def _raw(owner, repo, branch, path):
    try:
        r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", timeout=15)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return ""

def fetch_repo(state: AgentState):
    m = re.search(r"https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)", state["repo_url"])
    if not m:
        raise ValueError("Could not parse GitHub repo URL")
    owner, repo = m.group(1), m.group(2)
    r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md", timeout=30)
    if r.status_code != 200:
        r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md", timeout=30)
    readme = r.text[:4000] if r.status_code == 200 else ""
    meta, tree, manifest = "", "", ""
    code_chunks = []
    def grab(path):
        for br in ("main", "master"):
            t = _raw(owner, repo, br, path)
            if t:
                code_chunks.append(f"--- {path} ---\n{t[:2000]}")
                return True
        return False
    try:
        meta_r = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=20).json()
        meta = (f"{meta_r.get('description', '')} | stars: {meta_r.get('stargazers_count', '?')} "
                f"| lang: {meta_r.get('language', '?')} | topics: {', '.join(meta_r.get('topics', [])[:8])}")
    except Exception:
        pass
    try:
        top = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/", timeout=20).json()
        items = top if isinstance(top, list) else []
        tree = ", ".join(x.get("name", "") for x in items[:30])
        cands = [x["name"] for x in items if x.get("type") == "file" and x["name"].lower().endswith(CODE_EXTS)]
        cands.sort(key=_rank)
        for name in cands[:4]:
            grab(name)
        subdirs = [x["name"] for x in items if x.get("type") == "dir" and x["name"] not in SKIP_DIRS]
        for sd in ["src", "lib", "app", "pkg", "components"]:
            if sd in subdirs:
                try:
                    sub = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{sd}", timeout=20).json()
                    if isinstance(sub, list):
                        sfiles = sorted(
                            (x["name"] for x in sub
                             if x.get("type") == "file" and x["name"].lower().endswith(CODE_EXTS)),
                            key=_rank)[:3]
                        for name in sfiles:
                            if len(code_chunks) >= 7:
                                break
                            grab(f"{sd}/{name}")
                except Exception:
                    pass
                break
    except Exception:
        pass
    for mf in ["package.json", "pyproject.toml", "setup.py", "Cargo.toml", "go.mod"]:
        t = _raw(owner, repo, "main", mf) or _raw(owner, repo, "master", mf)
        if t and len(t) > 50:
            manifest = f"[{mf}]\n{t[:1200]}"
            break
    return {"readme_text": readme,
            "repo_context": f"META: {meta}\nFILES: {tree}\n{manifest}",
            "code_context": "\n\n".join(code_chunks)}


def understand_project(state: AgentState):
    prompt = f"""You are a senior staff engineer doing technical due diligence on an open-source project.
Read the README, repo metadata, and source code excerpts below and explain the project like you truly understand it.

REPO: {state['repo_url']}
{state['repo_context']}

README:
{state['readme_text']}

SOURCE CODE EXCERPTS:
{state['code_context']}

Output ONLY a valid JSON object matching this exact schema:
{{"one_liner": "what it is, in one punchy sentence",
"concept": "2-3 sentences: the core idea, explained simply",
"how_it_works": "3-5 sentences, technically concrete: what the user actually does step by step, and what the code does under the hood",
"key_features": ["concrete feature with a specific detail", "up to 6 total, most impressive first"],
"audience": "who this is for, specifically",
"differentiator": "what makes it different from alternatives — or 'not clear from context' if honestly unknown",
"vibe": "the project's personality/aesthetic in ~5 words"}}
RULES: Only state what the context supports. Be concrete: name real commands, file types, behaviors from the code — never generic filler."""
    brief = do_call(prompt, max_tokens=1500, temperature=0.2)
    # Focused second pass — the main brief often drops these fields, so ask directly.
    try:
        review = do_call(
            "You are a blunt senior engineer reviewing an open-source project. Context:\n"
            + json.dumps(brief, indent=1)[:4000] +
            "\nReturn ONLY valid JSON: {\"strengths\": [\"3-4 concrete strengths grounded in the context\"], "
            "\"weaknesses\": [\"3-4 honest weaknesses, limitations, or missing pieces visible from the context\"]}. "
            "Be specific and technical — e.g. 'has zero tests', 'README lacks a usage example', "
            "'setup needs 5 manual steps'. Never vague filler like 'could be more popular'.",
            max_tokens=600, temperature=0.3)
        brief["strengths"] = [str(s) for s in review.get("strengths", [])][:4]
        brief["weaknesses"] = [str(s) for s in review.get("weaknesses", [])][:4]
    except Exception:
        brief["_review_failed"] = True
    return {"concept_brief": brief}

def draft_strategy(state: AgentState):
    brief_text = json.dumps(state["concept_brief"], indent=1)
    posts_prompt = f"""You are a senior product marketer who writes scroll-stopping launch content for developer tools.
You already understand the project deeply. Concept brief:
{brief_text}
RULES:
- Every claim must come from the brief. NEVER invent features, stats, integrations, or testimonials.
- Concrete nouns only. BANNED: revolutionary, game-changing, cutting-edge, unlock, supercharge, seamless.
- HOOKS name a painful problem or open a curiosity loop. Max 8 words.
- Short SCRIPTS are SPOKEN WORD: contractions, short sentences, 25-35 words, end with a call to action.
- SCRIPTS must never contain URLs, links, or the word "https" — say "link below" instead.
- IMAGE PROMPTS are self-contained prompts for a square 1:1 social promo graphic depicting THIS project's actual subject matter (never generic laptops/robots/tech wallpaper). MUST include the exact on-image headline in "quotes" (max 5 words), describe scene, composition, style, lighting, colors, and end with: "No watermark, no extra text, no garbled letters."
Output ONLY a valid JSON object matching this exact schema:
{{"pitch_cards": [{{"headline": "punchy benefit, max 6 words", "sub": "one concrete sentence with a real feature or proof point"}}, {{"headline": "...", "sub": "..."}}, {{"headline": "...", "sub": "..."}}],
"social_posts": [{{"hook": "scroll-stopper, max 8 words", "script": "25-35 word spoken script: hook, one concrete capability, call to action", "image_prompt": "full image generation prompt per the rules above", "caption": "post caption: hook line, 2-3 value lines, call to action", "hashtags": ["#tag1", "#tag2", "#tag3"]}}, {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}}, {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}}]}}
The 3 posts cover 3 angles IN ORDER: 1) the painful problem, 2) the magic moment of using it, 3) proof + call to action (star the repo)."""
    package = do_call(posts_prompt, max_tokens=4000, temperature=0.3)
    cards = package.get("pitch_cards", [])
    norm_cards = [c if isinstance(c, dict) else {"headline": str(c), "sub": ""} for c in cards]
    return {"pitch_cards": norm_cards, "social_posts": package.get("social_posts", [])}

def generate_assets(state: AgentState):
    out_dir = state["out_dir"]
    posts = state["social_posts"]
    for i, post in enumerate(posts):
        audio_path = os.path.join(out_dir, f"post_{i}.mp3")
        script = _clean_spoken(post.get("script", ""))
        post["script"] = script  # keep displayed script identical to what the voice says
        a_err = tts_to_mp3(script, audio_path)
        if a_err:
            post["audio_path"] = None
            post["audio_error"] = a_err
        else:
            post["audio_path"] = audio_path
            post["audio_error"] = None
        try:
            img_rsp = requests.post(
                "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
                json={"model": "qwen-image-max",
                      "input": {"messages": [{"role": "user", "content": [{"text": post["image_prompt"]}]}]},
                      "parameters": {"size": "1328*1328", "n": 1, "prompt_extend": True, "watermark": False}},
                timeout=180)
            img_data = img_rsp.json()
            if img_rsp.status_code != 200:
                raise RuntimeError(f"Alibaba error {img_rsp.status_code}: {img_data.get('message', img_rsp.text)}")
            img_url = None
            for block in img_data["output"]["choices"][0]["message"]["content"]:
                if isinstance(block, dict) and "image" in block:
                    img_url = block["image"]
                    break
            if not img_url:
                raise RuntimeError(f"No image in response: {img_data}")
            img_path = os.path.join(out_dir, f"post_{i}.png")
            if img_url.startswith("data:"):
                _, b64data = img_url.split(",", 1)
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(b64data))
            else:
                img_res = requests.get(img_url, timeout=60)
                img_res.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(img_res.content)
            post["img_path"] = img_path
            post["img_error"] = None
        except Exception as e:
            post["img_path"] = None
            post["img_error"] = str(e)
    return {"social_posts": posts}

workflow = StateGraph(AgentState)
workflow.add_node("fetch_repo", fetch_repo)
workflow.add_node("understand_project", understand_project)
workflow.add_node("draft_strategy", draft_strategy)
workflow.add_node("generate_assets", generate_assets)
workflow.set_entry_point("fetch_repo")
workflow.add_edge("fetch_repo", "understand_project")
workflow.add_edge("understand_project", "draft_strategy")
workflow.add_edge("draft_strategy", "generate_assets")
workflow.add_edge("generate_assets", END)
video_agent = workflow.compile()

STEPS = [("fetch_repo", "📥", "Reading repo"),
         ("understand_project", "🔬", "Understanding project"),
         ("draft_strategy", "🧠", "Writing copy"),
         ("generate_assets", "🎨", "Images + voiceovers")]

def render_steps(done, active=None):
    parts = []
    for key, icon, label in STEPS:
        cls = "done" if key in done else ("active" if key == active else "todo")
        mark = "✓" if key in done else icon
        parts.append(f'<div class="step {cls}"><div class="dot">{mark}</div>{label}</div>')
    return '<div class="steps">' + "".join(parts) + "</div>"
_CSS = "\n<style>\n@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@400;600;700;800&family=Playfair+Display:ital,wght@1,500;1,600;1,700&display=swap');\n\n  :root {\n    --bg:#fff; --bg2:#F9FAFB; --bg3:#F3F4F6;\n    --border:#E5E7EB; --border2:#D1D5DB;\n    --text:#0A0A0A; --text2:#374151; --text3:#6B7280; --text4:#9CA3AF;\n    --green:#059669;\n  }\n  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}\n  *{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}\n  html{scroll-behavior:smooth;}\n  ::selection{background:#0A0A0A;color:#fff;}\n\n  .stApp{background:var(--bg)!important;min-height:100vh;}\n  #MainMenu,footer,header[data-testid=\"stHeader\"]{display:none!important;}\n  .block-container{max-width:1160px!important;padding:0 clamp(16px,4vw,48px) 100px!important;margin:0 auto!important;}\n  section[data-testid=\"stSidebar\"]{display:none!important;}\n\n  .bg-canvas,.grid-overlay,.orb,.noise{display:none;}\n\n  /* NAV */\n  .nav{position:sticky;top:0;z-index:100;background:rgba(255,255,255,0.92);backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--border);margin:0 clamp(-16px,-4vw,-48px);padding:0 clamp(16px,4vw,48px);}\n  .nav-inner{max-width:1160px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:60px;}\n  .logo{font-family:'Sora',sans-serif!important;font-weight:800;font-size:18px;letter-spacing:-.04em;color:var(--text);display:flex;align-items:center;gap:9px;}\n  .logo-icon{width:30px;height:30px;border-radius:8px;background:var(--text)!important;color:#fff!important;display:flex;align-items:center;justify-content:center;font-size:14px;}\n  .logo-text span{color:var(--text3);}\n  .nav-links{display:flex;align-items:center;gap:28px;}\n  .nav-links a{color:var(--text3);text-decoration:none;font-size:14px;font-weight:500;transition:color .15s;}\n  .nav-links a:hover{color:var(--text);}\n  .nav-badge{background:var(--bg3);border:1px solid var(--border);color:var(--text3);font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:6px;}\n  .nav-cta{background:var(--text)!important;color:#fff!important;text-decoration:none;font-size:13px;font-weight:600;padding:9px 20px;border-radius:8px;transition:opacity .15s;}\n  .nav-cta:hover{opacity:.82;}\n\n  /* HERO */\n  .hero{text-align:center;padding:clamp(80px,11vw,130px) 16px clamp(20px,4vw,40px);position:relative;z-index:2;}\n  .badge{display:inline-flex;align-items:center;gap:7px;font-size:11px;font-weight:600;letter-spacing:.14em;color:var(--text3);background:var(--bg3);border:1px solid var(--border);padding:6px 14px;border-radius:999px;margin-bottom:28px;text-transform:uppercase;}\n  .pulse-dot{width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 0 2px rgba(5,150,105,.2);animation:pulse 2s ease-in-out infinite;}\n  .hero h1{font-family:'Sora',sans-serif!important;font-size:clamp(2.6rem,7vw,5.2rem);font-weight:800;letter-spacing:-.05em;line-height:1.03;color:var(--text);margin:0 0 22px;}\n  .serif-accent{font-family:'Playfair Display',Georgia,serif!important;font-style:italic;font-weight:600;letter-spacing:-.02em;color:var(--text3);}\n  .hero p.sub{font-size:clamp(.95rem,2.5vw,1.15rem);color:var(--text3);max-width:560px;margin:0 auto 8px;line-height:1.75;font-weight:400;}\n\n  /* INPUT */\n  div[data-testid=\"stTextInput\"]{max-width:660px;margin:32px auto 0;position:relative;z-index:2;}\n  div[data-testid=\"stTextInput\"] label{display:none!important;}\n  div[data-testid=\"stTextInput\"] input{border-radius:12px!important;padding:16px 22px!important;font-size:14.5px!important;font-weight:400!important;border:1px solid var(--border2)!important;background:var(--bg)!important;color:var(--text)!important;box-shadow:0 1px 3px rgba(0,0,0,.06)!important;transition:border-color .15s,box-shadow .15s!important;caret-color:var(--text)!important;}\n  div[data-testid=\"stTextInput\"] input::placeholder{color:var(--text4)!important;}\n  div[data-testid=\"stTextInput\"] input:focus{border-color:var(--text)!important;box-shadow:0 0 0 3px rgba(10,10,10,.08)!important;background:var(--bg)!important;}\n\n  /* BUTTON */\n  div[data-testid=\"stButton\"]{margin-top:14px;position:relative;z-index:2;}\n  div[data-testid=\"stButton\"] button{background:var(--text)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:16px 40px!important;font-size:14.5px!important;font-weight:600!important;letter-spacing:-.01em!important;box-shadow:0 1px 3px rgba(0,0,0,.12)!important;transition:opacity .15s,transform .15s!important;width:100%!important;}\n  div[data-testid=\"stButton\"] button p{color:#fff!important;}\n  div[data-testid=\"stButton\"] button:hover{opacity:.86!important;transform:translateY(-1px)!important;box-shadow:0 4px 12px rgba(0,0,0,.15)!important;}\n\n  /* STATS */\n  .stats{display:flex;justify-content:center;align-items:center;margin:64px auto 0;max-width:780px;position:relative;z-index:2;border:1px solid var(--border);border-radius:16px;flex-wrap:wrap;overflow:hidden;background:var(--bg2);}\n  .stat{text-align:center;padding:24px 40px;flex:1;min-width:120px;}\n  .stat+.stat{border-left:1px solid var(--border);}\n  .stat b{display:block;font-size:32px;font-weight:800;letter-spacing:-.04em;color:var(--text);margin-bottom:4px;}\n  .stat span{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text4);}\n\n  /* TICKER */\n  .ticker{margin:64px clamp(-16px,-4vw,-48px) 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:var(--bg2);overflow:hidden;position:relative;z-index:2;}\n  .ticker-track{display:flex;gap:0;width:max-content;animation:tick 32s linear infinite;padding:15px 0;}\n  .ticker:hover .ticker-track{animation-play-state:paused;}\n  .tick{font-size:11px;font-weight:700;letter-spacing:.22em;color:var(--text4);padding:0 28px;white-space:nowrap;text-transform:uppercase;}\n  .tick em{font-style:normal;color:var(--text3);padding-right:28px;}\n  @keyframes tick{to{transform:translateX(-50%);}}\n\n  /* SECTIONS */\n  .section{max-width:1100px;margin:0 auto;padding:clamp(80px,10vw,112px) 0 0;position:relative;z-index:2;}\n  .kicker{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;}\n  .kicker::before{content:'';width:16px;height:1px;background:var(--border2);}\n  .sec-h{font-family:'Sora',sans-serif!important;font-size:clamp(1.8rem,4.5vw,2.9rem);font-weight:800;letter-spacing:-.04em;color:var(--text);margin:0 0 14px;line-height:1.1;}\n  .sec-p{color:var(--text3);font-size:16px;line-height:1.75;max-width:560px;margin:0 0 44px;font-weight:400;}\n\n  /* HOW IT WORKS */\n  .how-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:20px;overflow:hidden;}\n  .how-card{background:var(--bg);padding:36px 30px;transition:background .2s;}\n  .how-card:hover{background:var(--bg2);}\n  .how-card::before{display:none;}\n  .how-num{font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--text4);margin-bottom:18px;}\n  .how-card h3{font-size:16px;font-weight:700;margin:0 0 9px;color:var(--text);letter-spacing:-.01em;}\n  .how-card p{font-size:14px;color:var(--text3);line-height:1.7;margin:0;}\n\n  /* BUNDLE */\n  .bundle-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}\n  .bundle-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px 26px;transition:border-color .2s,box-shadow .2s;position:relative;overflow:hidden;}\n  .bundle-card::after{display:none;}\n  .bundle-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}\n  .bundle-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;background:var(--bg);border:1px solid var(--border);margin-bottom:18px;}\n  .bundle-card h3{font-size:15px;font-weight:700;margin:0 0 7px;color:var(--text);letter-spacing:-.01em;}\n  .bundle-card p{font-size:13.5px;color:var(--text3);line-height:1.65;margin:0;}\n\n  /* STEPS */\n  .steps{display:flex;gap:6px;justify-content:center;margin:40px auto 16px;max-width:1000px;position:relative;z-index:2;flex-wrap:wrap;}\n  .step{display:flex;align-items:center;gap:9px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:10px 16px 10px 10px;font-size:12.5px;font-weight:600;color:var(--text4);transition:all .25s;}\n  .step .dot{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:var(--bg3);font-size:13px;flex-shrink:0;}\n  .step.done{color:var(--green);border-color:rgba(5,150,105,.2);background:rgba(5,150,105,.04);}\n  .step.done .dot{background:rgba(5,150,105,.1);}\n  .step.active{color:var(--text);border-color:var(--border2);background:var(--bg);box-shadow:0 2px 8px rgba(0,0,0,.08);}\n  .step.active .dot{background:var(--text);color:#fff;animation:pulse 1.4s ease-in-out infinite;}\n  @keyframes pulse{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.1);opacity:.8;}}\n\n  /* RESULTS */\n  .sec-title{font-family:'Sora',sans-serif!important;font-size:clamp(1.5rem,3.5vw,2rem);font-weight:800;letter-spacing:-.035em;color:var(--text);margin:72px 0 6px;position:relative;z-index:2;}\n  .sec-sub{color:var(--text3);margin-bottom:24px;position:relative;z-index:2;font-size:14.5px;}\n  .demo-card{background:var(--text);border-radius:20px;padding:clamp(28px,4vw,52px);color:#fff;position:relative;overflow:hidden;z-index:2;box-shadow:0 20px 60px -16px rgba(0,0,0,.3);margin-top:16px;}\n  .demo-card::before,.demo-card::after{display:none;}\n  .demo-kicker{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;font-weight:700;letter-spacing:.16em;color:rgba(255,255,255,.45);margin-bottom:14px;position:relative;z-index:1;border:1px solid rgba(255,255,255,.12);padding:5px 12px;border-radius:999px;}\n  .demo-card h2{font-family:'Sora',sans-serif!important;font-size:clamp(1.4rem,3.5vw,2rem);font-weight:800;margin:0 0 20px;position:relative;z-index:1;letter-spacing:-.03em;color:#fff;}\n  .demo-card p.script{color:rgba(255,255,255,.65);line-height:1.85;font-size:15.5px;position:relative;z-index:1;margin-bottom:14px;font-weight:400;}\n\n  /* PITCH */\n  .pitch-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;position:relative;z-index:2;}\n  .pitch-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px 24px;position:relative;overflow:hidden;animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;transition:border-color .2s,box-shadow .2s;}\n  .pitch-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}\n  .pitch-card::before{display:none;}\n  /* HONEST REVIEW */\n  .sw-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:26px 0 8px;position:relative;z-index:2;}\n  .sw-card{background:var(--bg2);border:1px solid var(--border);border-radius:18px;padding:28px 26px;animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;transition:border-color .2s,box-shadow .2s;}\n  .sw-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}\n  .sw-card.strengths{border-top:3px solid var(--green);}\n  .sw-card.weaknesses{border-top:3px solid #DC2626;}\n  .sw-card h3{font-family:'Sora',sans-serif;font-size:17px;margin:0 0 16px;letter-spacing:-.01em;}\n  .sw-card ul{margin:0;padding:0;list-style:none;display:grid;gap:12px;}\n  .sw-card li{font-size:14.5px;line-height:1.6;color:var(--text2);padding-left:28px;position:relative;}\n  .sw-card.strengths li::before{content:\"✅\";position:absolute;left:0;top:0;}\n  .sw-card.weaknesses li::before{content:\"⚠️\";position:absolute;left:0;top:0;}\n  .pitch-card .icon{font-size:28px;margin-bottom:14px;}\n  .pitch-card h3{font-size:15px;font-weight:700;color:var(--text);margin:0 0 8px;letter-spacing:-.01em;}\n  .pitch-card p{font-size:13.5px;color:var(--text3);line-height:1.65;margin:0;}\n\n  /* SOCIAL */\n  .post-wrap{display:grid;grid-template-columns:280px 1fr;gap:36px;align-items:start;background:var(--bg2);border:1px solid var(--border);border-radius:20px;padding:clamp(24px,4vw,40px);margin-bottom:20px;position:relative;z-index:2;animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;transition:border-color .2s,box-shadow .2s;}\n  .post-wrap:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}\n  .post-num{position:absolute;top:-12px;left:24px;background:var(--text);color:#fff;font-size:10.5px;font-weight:700;letter-spacing:.1em;padding:5px 14px;border-radius:999px;}\n  .post-img-wrap{border-radius:18px;overflow:hidden;border:1px solid var(--border);background:var(--bg3);\n    box-shadow:0 18px 40px -16px rgba(0,0,0,.22);\n    transition:transform .35s cubic-bezier(.22,1,.36,1),box-shadow .35s cubic-bezier(.22,1,.36,1);}\n  .post-img-wrap:hover{transform:translateY(-7px) scale(1.015);box-shadow:0 30px 60px -18px rgba(0,0,0,.3);}\n  .post-img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--bg3);}\n  .hook{font-family:'Sora',sans-serif!important;font-size:clamp(1.2rem,3vw,1.65rem);font-weight:800;color:var(--text);letter-spacing:-.03em;margin:0 0 18px;line-height:1.2;}\n  .vo-label,.cap-label{font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text4);margin:22px 0 7px;display:flex;align-items:center;gap:7px;}\n  .vo-label::after,.cap-label::after{content:'';flex:1;height:1px;background:var(--border);}\n  .vo-script{font-size:14.5px;color:var(--text3);font-style:italic;line-height:1.75;border-left:2px solid var(--border2);padding-left:14px;margin:0 0 8px;}\n  .cap-text{font-size:13.5px;color:var(--text3);line-height:1.7;white-space:pre-line;}\n  .tags{margin-top:14px;display:flex;flex-wrap:wrap;gap:5px;}\n  .tag{background:var(--bg3);color:var(--text3);font-size:11.5px;font-weight:600;padding:4px 10px;border-radius:6px;border:1px solid var(--border);transition:all .15s;}\n  .tag:hover{background:var(--border);color:var(--text);transform:translateY(-1px);}\n\n  /* DOWNLOAD */\n  div[data-testid=\"stDownloadButton\"] button{border-radius:9px!important;font-weight:600!important;border:1px solid var(--border)!important;color:var(--text2)!important;background:var(--bg2)!important;padding:9px 18px!important;font-size:13px!important;width:100%;transition:all .15s;letter-spacing:-.01em!important;}\n  div[data-testid=\"stDownloadButton\"] button:hover{background:var(--bg3)!important;border-color:var(--border2)!important;color:var(--text)!important;transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.06);}\n\n  /* EXPANDER */\n  div[data-testid=\"stExpander\"]{border:1px solid var(--border)!important;border-radius:14px!important;background:var(--bg2)!important;position:relative;z-index:2;}\n\n  /* CTA */\n  .cta-dark{margin:100px auto 0;max-width:1100px;background:var(--text);border-radius:24px;padding:clamp(48px,7vw,84px);text-align:center;position:relative;overflow:hidden;z-index:2;}\n  .cta-dark::before,.cta-dark::after{display:none;}\n  .cta-dark h2{font-family:'Sora',sans-serif!important;color:#fff;font-size:clamp(1.9rem,5vw,3.2rem);font-weight:800;letter-spacing:-.04em;margin:0 0 14px;position:relative;z-index:1;line-height:1.1;}\n  .cta-dark p{color:rgba(255,255,255,.5);font-size:16px;max-width:500px;margin:0 auto;line-height:1.75;position:relative;z-index:1;font-weight:400;}\n\n  /* FOOTER */\n  .footer{margin-top:80px;border-top:1px solid var(--border);padding:36px 0 18px;position:relative;z-index:2;}\n  .footer-inner{max-width:1100px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;}\n  .footer .logo{font-size:16px;}\n  .footer p{color:var(--text4);font-size:13px;margin:0;}\n  .footer p b{color:var(--text3);font-weight:600;}\n\n  /* ANIMATIONS */\n  @keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:none;}}\n  .anim{animation:fadeUp .7s cubic-bezier(.22,1,.36,1) both;}\n  .d1{animation-delay:.08s;}.d2{animation-delay:.16s;}.d3{animation-delay:.24s;}.d4{animation-delay:.32s;}\n\n  /* STREAMLIT */\n  div[data-testid=\"stAudio\"]{border-radius:10px;overflow:hidden;border:1px solid var(--border);}\n  div[data-testid=\"stMarkdownContainer\"]{color:var(--text2)!important;}\n  div[data-testid=\"stMarkdownContainer\"] strong{color:var(--text)!important;}\n  div[data-testid=\"stMarkdownContainer\"] em{color:var(--text3)!important;}\n  .stAlert{border-radius:10px!important;border:1px solid var(--border)!important;background:var(--bg2)!important;}\n\n  /* RESPONSIVE */\n  @media(max-width:900px){\n    .nav{margin:0 -16px;padding:0 16px;}\n    .nav-links{display:none;}\n    .pitch-grid,.how-grid,.bundle-grid,.sw-grid{grid-template-columns:1fr;}\n    .how-grid{gap:0;}\n    .post-wrap{grid-template-columns:1fr;}\n    .stat{padding:18px 20px;}\n    .ticker{margin:56px -16px 0;}\n    .stats{border-radius:14px;}\n  }\n</style>\n<div class=\"bg-canvas\"></div>\n<div class=\"grid-overlay\"></div>\n<div class=\"orb orb-1\"></div>\n<div class=\"orb orb-2\"></div>\n<div class=\"orb orb-3\"></div>\n<div class=\"noise\"></div>\n"
_NAV = "\n<nav class=\"nav\"><div class=\"nav-inner\">\n  <div class=\"logo\">\n    <div class=\"logo-icon\">⚡</div>\n    <div class=\"logo-text\">Hype<span>Repo</span></div>\n  </div>\n  <div class=\"nav-links\">\n    <a href=\"#how\">How it works</a>\n    <a href=\"#bundle\">What you get</a>\n    <span class=\"nav-badge\">AI-Powered</span>\n  </div>\n  <a class=\"nav-cta\" href=\"#top\">Generate Kit ✦</a>\n</div></nav>\n"
_HERO = "\n<div class=\"hero\" id=\"top\">\n  <div class=\"badge anim\"><span class=\"pulse-dot\"></span>AI Marketing Agent &nbsp;·&nbsp; Zero setup</div>\n  <h1 class=\"anim d1\">Turn any repo into<br><span class=\"serif-accent\">a full launch kit.</span></h1>\n  <p class=\"sub anim d2\">Paste a GitHub URL and walk away with pitch cards, social posts with AI visuals, voiceovers, and an honest strengths-vs-weaknesses review — generated from your actual code, not a template.</p>\n</div>\n"
_STATS = "\n<div class=\"stats anim\">\n  <div class=\"stat\"><b>3</b><span>AI visuals</span></div>\n  <div class=\"stat\"><b>3</b><span>Voiceovers</span></div>\n  <div class=\"stat\"><b>3</b><span>Pitch cards</span></div>\n  <div class=\"stat\"><b>1</b><span>Honest review</span></div>\n</div>\n"
_TICKER = "\n<div class=\"ticker\"><div class=\"ticker-track\">\n  <span class=\"tick\"><em>✦</em>PITCH CARDS</span><span class=\"tick\"><em>✦</em>AI VOICEOVERS</span><span class=\"tick\"><em>✦</em>SCROLL-STOPPING VISUALS</span><span class=\"tick\"><em>✦</em>STRENGTHS & WEAKNESSES</span><span class=\"tick\"><em>✦</em>CAPTIONS & HASHTAGS</span><span class=\"tick\"><em>✦</em>ZERO EDITING NEEDED</span>\n  <span class=\"tick\"><em>✦</em>PITCH CARDS</span><span class=\"tick\"><em>✦</em>AI VOICEOVERS</span><span class=\"tick\"><em>✦</em>SCROLL-STOPPING VISUALS</span><span class=\"tick\"><em>✦</em>STRENGTHS & WEAKNESSES</span><span class=\"tick\"><em>✦</em>CAPTIONS & HASHTAGS</span><span class=\"tick\"><em>✦</em>ZERO EDITING NEEDED</span>\n</div></div>\n"
_HOW = "\n<div class=\"section\" id=\"how\">\n  <div class=\"kicker\">HOW IT WORKS</div>\n  <div class=\"sec-h\">Repo link to launch kit<br>in under five minutes.</div>\n  <p class=\"sec-p\">No prompts. No templates. A five-stage AI pipeline reads your code, understands what you built, then writes, designs, and records everything.</p>\n  <div class=\"how-grid\">\n    <div class=\"how-card anim\"><div class=\"how-num\">01</div><h3>🔗 Paste your repo URL</h3><p>Drop any public GitHub URL. The agent fetches your README, repo metadata, and actual source files — not just the docs.</p></div>\n    <div class=\"how-card anim d1\"><div class=\"how-num\">02</div><h3>🧠 Deep code analysis</h3><p>A senior-engineer-grade LLM builds a concept brief: what you built, how it works, who it's for, and what makes it genuinely different.</p></div>\n    <div class=\"how-card anim d2\"><div class=\"how-num\">03</div><h3>🚀 Ship the full kit</h3><p>Pitch cards, social posts with AI visuals and voiceovers, plus an honest strengths-and-weaknesses breakdown. One click to download all.</p></div>\n  </div>\n</div>\n"
_BUNDLE = "\n<div class=\"section\" id=\"bundle\">\n  <div class=\"kicker\">WHAT YOU GET</div>\n  <div class=\"sec-h\">Everything a launch needs.<br>Nothing it doesn't.</div>\n  <p class=\"sec-p\">Every asset is grounded in your actual code — never generic filler. Built from a real concept brief, not a template.</p>\n  <div class=\"bundle-grid\">\n    <div class=\"bundle-card anim\"><div class=\"bundle-icon\">✨</div><h3>Pitch Cards</h3><p>Three razor-sharp angles with concrete proof points — ready for your README or landing page hero.</p></div>\n    <div class=\"bundle-card anim d1\"><div class=\"bundle-icon\">📱</div><h3>Social Posts</h3><p>Problem → magic moment → proof. Scroll-stopping hooks, captions, and hashtags — three distinct angles.</p></div>\n    <div class=\"bundle-card anim d2\"><div class=\"bundle-icon\">🎨</div><h3>AI Visuals</h3><p>Custom 1:1 promo graphics per post, generated from prompts based on your project's real subject matter.</p></div>\n    <div class=\"bundle-card anim d3\"><div class=\"bundle-icon\">🎙️</div><h3>Voiceovers</h3><p>Natural-sounding AI narration for every post — no microphone, no studio needed.</p></div>\n    <div class=\"bundle-card anim d4\"><div class=\"bundle-icon\">⚖️</div><h3>Honest Review</h3><p>Brutally honest strengths and weaknesses, grounded in your actual code — what to brag about, what to fix.</p></div>\n    <div class=\"bundle-card anim\"><div class=\"bundle-icon\">⬇️</div><h3>Instant Download Kit</h3><p>Every image and MP3 is one click away. Take the whole bundle straight to your content scheduler.</p></div>\n  </div>\n</div>\n"
_CTA = "\n<div class=\"cta-dark\">\n  <h2>Your repo deserves more<br>than <span class=\"serif-accent\">a README.</span></h2>\n  <p>Paste a link above and walk away with a complete launch kit — copy, visuals, voiceovers, and an honest review. Powered by real code analysis.</p>\n</div>\n<div class=\"footer\"><div class=\"footer-inner\">\n  <div class=\"logo\">\n    <div class=\"logo-icon\">⚡</div>\n    <div class=\"logo-text\">Hype<span>Repo</span></div>\n  </div>\n  <p>Built with <b>HypeRepo</b> — paste a repo, ship the hype. © 2026</p>\n</div></div>\n"

# ================= DESIGN SYSTEM =================
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown(_NAV, unsafe_allow_html=True)
st.markdown(_HERO, unsafe_allow_html=True)
if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("🔑 Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

st.html('<div class="anim d3">', )
repo_url = st.text_input("repo", placeholder="https://github.com/owner/repo  —  paste any public repo URL")
st.html('</div>')

_l, _c, _r = st.columns([1.2, 2, 1.2])
with _c:
    _btn = st.button("✨ Generate marketing bundle", use_container_width=True)

if _btn:
    if not repo_url.strip() or "github.com" not in repo_url:
        st.error("Please paste a valid GitHub repo URL.")
        st.stop()

    out_dir = tempfile.mkdtemp(prefix="mktg_")
    final, done = {}, []
    steps_ph = st.empty()
    steps_ph.markdown(render_steps(done, active="fetch_repo"), unsafe_allow_html=True)
    failed = None
    try:
        for chunk in video_agent.stream({"repo_url": repo_url.strip(), "out_dir": out_dir}):
            for node, update in chunk.items():
                if update:
                    final.update(update)
                done.append(node)
                nxt = next((k for k, _, _ in STEPS if k not in done), None)
                steps_ph.markdown(render_steps(done, active=nxt), unsafe_allow_html=True)
    except Exception as e:
        failed = e
    steps_ph.markdown(render_steps(done), unsafe_allow_html=True)
    if failed is not None:
        step_label = STEPS[len(done)][2] if len(done) < len(STEPS) else "finishing up"
        st.error(f"⚠️ The run failed during **{step_label}** ({type(failed).__name__}): {html.escape(str(failed))[:250]}")
        st.info("This is usually the AI returning malformed output — not your repo. Hit **Generate** again; it usually works on retry, and nothing was billed beyond this attempt.")
        st.stop()

    m = re.search(r"github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)", repo_url)
    repo_name = html.escape(m.group(2)) if m else "project"

    brief = final.get("concept_brief", {})
    st.markdown('<div class="sec-title anim">⚖️ The Honest Review</div>'
                '<div class="sec-sub anim d1">Grounded in your actual code — what to brag about, and what to fix before launch.</div>',
                unsafe_allow_html=True)
    def _sw(items):
        items = [str(s) for s in (items or []) if str(s).strip()]
        return "".join(f"<li>{html.escape(s)}</li>" for s in items) or "<li>—</li>"
    if brief.get("_review_failed"):
        # The review call itself flopped — say so, don't pretend.
        st.markdown("""<div class="sw-grid anim d1">
      <div class="sw-card"><h3>🧐 Review hiccup</h3><ul><li>The reviewer flopped this run — hit Generate again and it should fill in.</li></ul></div>
    </div>""", unsafe_allow_html=True)
    else:
        # Genuine empty verdict? Leave it empty — no nagging, no invented content.
        st.markdown(f"""<div class="sw-grid anim d1">
      <div class="sw-card strengths"><h3>💪 Strengths</h3><ul>{_sw(brief.get("strengths"))}</ul></div>
      <div class="sw-card weaknesses"><h3>🧐 Weaknesses</h3><ul>{_sw(brief.get("weaknesses"))}</ul></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title anim">✨ Pitch Cards</div>'
                '<div class="sec-sub anim d1">The three strongest angles — ready for your README or landing page.</div>',
                unsafe_allow_html=True)
    icons = ["⚡", "🎯", "💎"]
    cards_html = "".join(
        f"""<div class="pitch-card d{i+1}"><div class="icon">{icons[i % 3]}</div>
            <h3>{html.escape(c.get('headline', ''))}</h3><p>{html.escape(c.get('sub', ''))}</p></div>"""
        for i, c in enumerate(final.get("pitch_cards", [])))
    st.markdown(f'<div class="pitch-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title anim">📱 Social Posts</div>'
                '<div class="sec-sub anim d1">Hook, voiceover, AI visual, caption & hashtags — previewed like real posts.</div>',
                unsafe_allow_html=True)
    for i, post in enumerate(final.get("social_posts", []), 1):
        img_bytes = None
        if post.get("img_path") and os.path.exists(post["img_path"]):
            with open(post["img_path"], "rb") as f:
                img_bytes = f.read()
            img_tag = f'<img class="post-img" src="data:image/png;base64,{base64.b64encode(img_bytes).decode()}" />'
        else:
            img_tag = ('<div class="post-img" style="display:flex;align-items:center;justify-content:center;'
                       'color:#a1a1aa;font-size:13px;padding:20px;text-align:center;aspect-ratio:1/1;">'
                       f'⚠ image failed<br>{html.escape(post.get("img_error", ""))[:120]}</div>')
        hook = html.escape(post.get("hook", ""))
        st.markdown(f"""<div class="post-wrap d{(i % 3) + 1}">
          <div class="post-num">POST {i}</div>
          <div class="post-img-wrap">{img_tag}</div>
          <div>
            <div class="hook">🪝 {hook}</div>
            <div class="vo-label">VOICEOVER SCRIPT</div>
            <p class="vo-script">"{html.escape(post.get('script', ''))}"</p>
            <div class="cap-label">CAPTION</div>
            <p class="cap-text">{html.escape(post.get('caption', ''))}</p>
            <div class="tags">{"".join(f'<span class="tag">{html.escape(t)}</span>' for t in post.get("hashtags", []))}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        a_bytes = None
        if post.get("audio_path") and os.path.exists(post["audio_path"]):
            with open(post["audio_path"], "rb") as f:
                a_bytes = f.read()
            st.audio(a_bytes, format="audio/mpeg")
        elif post.get("audio_error"):
            st.warning(f"🎙️ Voiceover {i} failed: {post['audio_error'][:250]}")
        c1, c2 = st.columns(2)
        with c1:
            if img_bytes:
                st.download_button(f"⬇️ Image {i}", img_bytes, file_name=f"post_{i}.png", key=f"dl_img_{i}")
        with c2:
            if a_bytes:
                st.download_button(f"⬇️ Voiceover {i}", a_bytes, file_name=f"post_{i}.mp3", key=f"dl_aud_{i}")

    # ================= GENERATION REPORT =================
    _rep = []
    for _i, _p in enumerate(final.get("social_posts", []), 1):
        _img_ok = bool(_p.get("img_path") and os.path.exists(_p["img_path"]))
        _aud_ok = bool(_p.get("audio_path") and os.path.exists(_p["audio_path"]))
        _rep.append(f"{'✅' if _img_ok else '❌'} Post {_i} image · {'✅' if _aud_ok else '❌'} Post {_i} voiceover")
    st.markdown("<div class='sec-title anim'>🧾 Generation report</div>"
                "<div class='sec-sub anim d1'>What actually got built — no silent failures.</div>",
                unsafe_allow_html=True)
    for _line in _rep:
        st.markdown(f"- {_line}")

# ================= STATS =================
st.markdown(_STATS, unsafe_allow_html=True)

# ================= TICKER =================
st.markdown(_TICKER, unsafe_allow_html=True)

# ================= HOW IT WORKS =================
st.markdown(_HOW, unsafe_allow_html=True)

# ================= WHAT YOU GET =================
st.markdown(_BUNDLE, unsafe_allow_html=True)

# ================= CTA + FOOTER =================
st.markdown(_CTA, unsafe_allow_html=True)
