import os
import re
import json
import html
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
def tts_to_mp3(text, out_path, voice="en-US-RogerNeural"):
    """Returns None on success, or an error string on failure. Never raises,
    so one bad voiceover can't silently kill the whole bundle."""
    if not (text or "").strip():
        return "empty script — nothing to narrate"
    try:
        async def _main():
            rate = "+0%" if len(text) > 500 else "+5%"
            await edge_tts.Communicate(text, voice=voice, rate=rate).save(out_path)
        asyncio.run(_main())
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return "TTS finished but produced no audio file"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def do_call(prompt, max_tokens, temperature, _retry=True):
    response = requests.post(DO_URL,
        headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"},
        json={"model": DO_MODEL, "input": prompt, "max_output_tokens": max_tokens,
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
        return json.loads(final_json)
    except json.JSONDecodeError as e:
        if not _retry:
            raise e
        return do_call("Your previous response was not valid JSON (cut off or malformed). "
                       "Return the COMPLETE object again as valid JSON only, every field, full text, "
                       "no truncation, no markdown fences.\n\nBroken output:\n" + final_json[:6000],
                       max_tokens, temperature, _retry=False)

class AgentState(TypedDict):
    repo_url: str; out_dir: str; readme_text: str; repo_context: str; code_context: str
    concept_brief: Dict; pitch_cards: List[Dict]; social_posts: List[Dict]; demo_pitch: Dict

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
    return {"concept_brief": do_call(prompt, max_tokens=1500, temperature=0.2)}

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
- IMAGE PROMPTS are self-contained prompts for a square 1:1 social promo graphic depicting THIS project's actual subject matter (never generic laptops/robots/tech wallpaper). MUST include the exact on-image headline in "quotes" (max 5 words), describe scene, composition, style, lighting, colors, and end with: "No watermark, no extra text, no garbled letters."
Output ONLY a valid JSON object matching this exact schema:
{{"pitch_cards": [{{"headline": "punchy benefit, max 6 words", "sub": "one concrete sentence with a real feature or proof point"}}, {{"headline": "...", "sub": "..."}}, {{"headline": "...", "sub": "..."}}],
"social_posts": [{{"hook": "scroll-stopper, max 8 words", "script": "25-35 word spoken script: hook, one concrete capability, call to action", "image_prompt": "full image generation prompt per the rules above", "caption": "post caption: hook line, 2-3 value lines, call to action", "hashtags": ["#tag1", "#tag2", "#tag3"]}}, {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}}, {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}}]}}
The 3 posts cover 3 angles IN ORDER: 1) the painful problem, 2) the magic moment of using it, 3) proof + call to action (star the repo)."""
    package = do_call(posts_prompt, max_tokens=3000, temperature=0.3)
    cards = package.get("pitch_cards", [])
    norm_cards = [c if isinstance(c, dict) else {"headline": str(c), "sub": ""} for c in cards]
    demo_prompt = f"""You are a demo-day pitch coach writing a spoken product demo.
You already understand the project deeply. Concept brief:
{brief_text}
Write a FULL 2-minute spoken demo pitch: 260-300 words, paragraphs separated by blank lines.
Structure IN ORDER:
1) Cold-open hook — a surprising or painful truth (15s)
2) The problem this project kills (25s)
3) Narrated walkthrough — describe using it as if showing the screen, naming real UI elements and behaviors from the brief (50s)
4) The 2-3 strongest features with concrete details from the brief (30s)
5) Who it's for + call to action: star the repo, link below (15s)
RULES: spoken word only — contractions, short sentences, concrete nouns. NO bullet points, NO stage directions, no invented features. BANNED: revolutionary, game-changing, cutting-edge, unlock, supercharge, seamless.
Output ONLY a valid JSON object matching this exact schema:
{{"demo_pitch": {{"title": "title of the 2-minute demo", "script": "..."}}}}"""
    demo_data = do_call(demo_prompt, max_tokens=2000, temperature=0.3)
    return {"pitch_cards": norm_cards, "social_posts": package.get("social_posts", []),
            "demo_pitch": demo_data.get("demo_pitch", {})}

def generate_assets(state: AgentState):
    out_dir = state["out_dir"]
    posts = state["social_posts"]
    for i, post in enumerate(posts):
        audio_path = os.path.join(out_dir, f"post_{i}.mp3")
        a_err = tts_to_mp3(post.get("script", ""), audio_path)
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
    demo = state.get("demo_pitch", {})
    if demo.get("script"):
        demo_path = os.path.join(out_dir, "demo_pitch.mp3")
        d_err = tts_to_mp3(demo["script"], demo_path)
        if d_err:
            demo["audio_path"] = None
            demo["audio_error"] = d_err
        else:
            demo["audio_path"] = demo_path
            demo["audio_error"] = None
    return {"social_posts": posts, "demo_pitch": demo}

def build_dashboard(state: AgentState):
    return {}

workflow = StateGraph(AgentState)
workflow.add_node("fetch_repo", fetch_repo)
workflow.add_node("understand_project", understand_project)
workflow.add_node("draft_strategy", draft_strategy)
workflow.add_node("generate_assets", generate_assets)
workflow.add_node("build_dashboard", build_dashboard)
workflow.set_entry_point("fetch_repo")
workflow.add_edge("fetch_repo", "understand_project")
workflow.add_edge("understand_project", "draft_strategy")
workflow.add_edge("draft_strategy", "generate_assets")
workflow.add_edge("generate_assets", "build_dashboard")
workflow.add_edge("build_dashboard", END)
video_agent = workflow.compile()

STEPS = [("fetch_repo", "📥", "Reading repo"),
         ("understand_project", "🔬", "Understanding project"),
         ("draft_strategy", "🧠", "Writing copy"),
         ("generate_assets", "🎨", "Images + voiceovers"),
         ("build_dashboard", "📊", "Assembling")]

def render_steps(done, active=None):
    parts = []
    for key, icon, label in STEPS:
        cls = "done" if key in done else ("active" if key == active else "todo")
        mark = "✓" if key in done else icon
        parts.append(f'<div class="step {cls}"><div class="dot">{mark}</div>{label}</div>')
    return '<div class="steps">' + "".join(parts) + "</div>"

# ================= DESIGN SYSTEM =================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@400;600;700;800&family=Playfair+Display:ital,wght@1,500;1,600;1,700&display=swap');

  :root {
    --bg:#fff; --bg2:#F9FAFB; --bg3:#F3F4F6;
    --border:#E5E7EB; --border2:#D1D5DB;
    --text:#0A0A0A; --text2:#374151; --text3:#6B7280; --text4:#9CA3AF;
    --green:#059669;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  *{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}
  html{scroll-behavior:smooth;}
  ::selection{background:#0A0A0A;color:#fff;}

  .stApp{background:var(--bg)!important;min-height:100vh;}
  #MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
  .block-container{max-width:1160px!important;padding:0 clamp(16px,4vw,48px) 100px!important;margin:0 auto!important;}
  section[data-testid="stSidebar"]{display:none!important;}

  .bg-canvas,.grid-overlay,.orb,.noise{display:none;}

  /* NAV */
  .nav{position:sticky;top:0;z-index:100;background:rgba(255,255,255,0.92);backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--border);margin:0 clamp(-16px,-4vw,-48px);padding:0 clamp(16px,4vw,48px);}
  .nav-inner{max-width:1160px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:60px;}
  .logo{font-family:'Sora',sans-serif!important;font-weight:800;font-size:18px;letter-spacing:-.04em;color:var(--text);display:flex;align-items:center;gap:9px;}
  .logo-icon{width:30px;height:30px;border-radius:8px;background:var(--text)!important;color:#fff!important;display:flex;align-items:center;justify-content:center;font-size:14px;}
  .logo-text span{color:var(--text3);}
  .nav-links{display:flex;align-items:center;gap:28px;}
  .nav-links a{color:var(--text3);text-decoration:none;font-size:14px;font-weight:500;transition:color .15s;}
  .nav-links a:hover{color:var(--text);}
  .nav-badge{background:var(--bg3);border:1px solid var(--border);color:var(--text3);font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:6px;}
  .nav-cta{background:var(--text)!important;color:#fff!important;text-decoration:none;font-size:13px;font-weight:600;padding:9px 20px;border-radius:8px;transition:opacity .15s;}
  .nav-cta:hover{opacity:.82;}

  /* HERO */
  .hero{text-align:center;padding:clamp(80px,11vw,130px) 16px clamp(20px,4vw,40px);position:relative;z-index:2;}
  .badge{display:inline-flex;align-items:center;gap:7px;font-size:11px;font-weight:600;letter-spacing:.14em;color:var(--text3);background:var(--bg3);border:1px solid var(--border);padding:6px 14px;border-radius:999px;margin-bottom:28px;text-transform:uppercase;}
  .pulse-dot{width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 0 2px rgba(5,150,105,.2);animation:pulse 2s ease-in-out infinite;}
  .hero h1{font-family:'Sora',sans-serif!important;font-size:clamp(2.6rem,7vw,5.2rem);font-weight:800;letter-spacing:-.05em;line-height:1.03;color:var(--text);margin:0 0 22px;}
  .serif-accent{font-family:'Playfair Display',Georgia,serif!important;font-style:italic;font-weight:600;letter-spacing:-.02em;color:var(--text3);}
  .hero p.sub{font-size:clamp(.95rem,2.5vw,1.15rem);color:var(--text3);max-width:560px;margin:0 auto 8px;line-height:1.75;font-weight:400;}

  /* INPUT */
  div[data-testid="stTextInput"]{max-width:660px;margin:32px auto 0;position:relative;z-index:2;}
  div[data-testid="stTextInput"] label{display:none!important;}
  div[data-testid="stTextInput"] input{border-radius:12px!important;padding:16px 22px!important;font-size:14.5px!important;font-weight:400!important;border:1px solid var(--border2)!important;background:var(--bg)!important;color:var(--text)!important;box-shadow:0 1px 3px rgba(0,0,0,.06)!important;transition:border-color .15s,box-shadow .15s!important;caret-color:var(--text)!important;}
  div[data-testid="stTextInput"] input::placeholder{color:var(--text4)!important;}
  div[data-testid="stTextInput"] input:focus{border-color:var(--text)!important;box-shadow:0 0 0 3px rgba(10,10,10,.08)!important;background:var(--bg)!important;}

  /* BUTTON */
  div[data-testid="stButton"]{margin-top:14px;position:relative;z-index:2;}
  div[data-testid="stButton"] button{background:var(--text)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:16px 40px!important;font-size:14.5px!important;font-weight:600!important;letter-spacing:-.01em!important;box-shadow:0 1px 3px rgba(0,0,0,.12)!important;transition:opacity .15s,transform .15s!important;width:100%!important;}
  div[data-testid="stButton"] button p{color:#fff!important;}
  div[data-testid="stButton"] button:hover{opacity:.86!important;transform:translateY(-1px)!important;box-shadow:0 4px 12px rgba(0,0,0,.15)!important;}

  /* STATS */
  .stats{display:flex;justify-content:center;align-items:center;margin:64px auto 0;max-width:780px;position:relative;z-index:2;border:1px solid var(--border);border-radius:16px;flex-wrap:wrap;overflow:hidden;background:var(--bg2);}
  .stat{text-align:center;padding:24px 40px;flex:1;min-width:120px;}
  .stat+.stat{border-left:1px solid var(--border);}
  .stat b{display:block;font-size:32px;font-weight:800;letter-spacing:-.04em;color:var(--text);margin-bottom:4px;}
  .stat span{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text4);}

  /* TICKER */
  .ticker{margin:64px clamp(-16px,-4vw,-48px) 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:var(--bg2);overflow:hidden;position:relative;z-index:2;}
  .ticker-track{display:flex;gap:0;width:max-content;animation:tick 32s linear infinite;padding:15px 0;}
  .ticker:hover .ticker-track{animation-play-state:paused;}
  .tick{font-size:11px;font-weight:700;letter-spacing:.22em;color:var(--text4);padding:0 28px;white-space:nowrap;text-transform:uppercase;}
  .tick em{font-style:normal;color:var(--text3);padding-right:28px;}
  @keyframes tick{to{transform:translateX(-50%);}}

  /* SECTIONS */
  .section{max-width:1100px;margin:0 auto;padding:clamp(80px,10vw,112px) 0 0;position:relative;z-index:2;}
  .kicker{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;}
  .kicker::before{content:'';width:16px;height:1px;background:var(--border2);}
  .sec-h{font-family:'Sora',sans-serif!important;font-size:clamp(1.8rem,4.5vw,2.9rem);font-weight:800;letter-spacing:-.04em;color:var(--text);margin:0 0 14px;line-height:1.1;}
  .sec-p{color:var(--text3);font-size:16px;line-height:1.75;max-width:560px;margin:0 0 44px;font-weight:400;}

  /* HOW IT WORKS */
  .how-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:20px;overflow:hidden;}
  .how-card{background:var(--bg);padding:36px 30px;transition:background .2s;}
  .how-card:hover{background:var(--bg2);}
  .how-card::before{display:none;}
  .how-num{font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--text4);margin-bottom:18px;}
  .how-card h3{font-size:16px;font-weight:700;margin:0 0 9px;color:var(--text);letter-spacing:-.01em;}
  .how-card p{font-size:14px;color:var(--text3);line-height:1.7;margin:0;}

  /* BUNDLE */
  .bundle-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
  .bundle-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px 26px;transition:border-color .2s,box-shadow .2s;position:relative;overflow:hidden;}
  .bundle-card::after{display:none;}
  .bundle-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}
  .bundle-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;background:var(--bg);border:1px solid var(--border);margin-bottom:18px;}
  .bundle-card h3{font-size:15px;font-weight:700;margin:0 0 7px;color:var(--text);letter-spacing:-.01em;}
  .bundle-card p{font-size:13.5px;color:var(--text3);line-height:1.65;margin:0;}

  /* STEPS */
  .steps{display:flex;gap:6px;justify-content:center;margin:40px auto 16px;max-width:1000px;position:relative;z-index:2;flex-wrap:wrap;}
  .step{display:flex;align-items:center;gap:9px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:10px 16px 10px 10px;font-size:12.5px;font-weight:600;color:var(--text4);transition:all .25s;}
  .step .dot{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:var(--bg3);font-size:13px;flex-shrink:0;}
  .step.done{color:var(--green);border-color:rgba(5,150,105,.2);background:rgba(5,150,105,.04);}
  .step.done .dot{background:rgba(5,150,105,.1);}
  .step.active{color:var(--text);border-color:var(--border2);background:var(--bg);box-shadow:0 2px 8px rgba(0,0,0,.08);}
  .step.active .dot{background:var(--text);color:#fff;animation:pulse 1.4s ease-in-out infinite;}
  @keyframes pulse{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.1);opacity:.8;}}

  /* RESULTS */
  .sec-title{font-family:'Sora',sans-serif!important;font-size:clamp(1.5rem,3.5vw,2rem);font-weight:800;letter-spacing:-.035em;color:var(--text);margin:72px 0 6px;position:relative;z-index:2;}
  .sec-sub{color:var(--text3);margin-bottom:24px;position:relative;z-index:2;font-size:14.5px;}
  .demo-card{background:var(--text);border-radius:20px;padding:clamp(28px,4vw,52px);color:#fff;position:relative;overflow:hidden;z-index:2;box-shadow:0 20px 60px -16px rgba(0,0,0,.3);margin-top:16px;}
  .demo-card::before,.demo-card::after{display:none;}
  .demo-kicker{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;font-weight:700;letter-spacing:.16em;color:rgba(255,255,255,.45);margin-bottom:14px;position:relative;z-index:1;border:1px solid rgba(255,255,255,.12);padding:5px 12px;border-radius:999px;}
  .demo-card h2{font-family:'Sora',sans-serif!important;font-size:clamp(1.4rem,3.5vw,2rem);font-weight:800;margin:0 0 20px;position:relative;z-index:1;letter-spacing:-.03em;color:#fff;}
  .demo-card p.script{color:rgba(255,255,255,.65);line-height:1.85;font-size:15.5px;position:relative;z-index:1;margin-bottom:14px;font-weight:400;}

  /* PITCH */
  .pitch-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;position:relative;z-index:2;}
  .pitch-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px 24px;position:relative;overflow:hidden;animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;transition:border-color .2s,box-shadow .2s;}
  .pitch-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}
  .pitch-card::before{display:none;}
  .pitch-card .icon{font-size:28px;margin-bottom:14px;}
  .pitch-card h3{font-size:15px;font-weight:700;color:var(--text);margin:0 0 8px;letter-spacing:-.01em;}
  .pitch-card p{font-size:13.5px;color:var(--text3);line-height:1.65;margin:0;}

  /* SOCIAL */
  .post-wrap{display:grid;grid-template-columns:280px 1fr;gap:36px;align-items:start;background:var(--bg2);border:1px solid var(--border);border-radius:20px;padding:clamp(24px,4vw,40px);margin-bottom:20px;position:relative;z-index:2;animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;transition:border-color .2s,box-shadow .2s;}
  .post-wrap:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.06);}
  .post-num{position:absolute;top:-12px;left:24px;background:var(--text);color:#fff;font-size:10.5px;font-weight:700;letter-spacing:.1em;padding:5px 14px;border-radius:999px;}
  .post-img-wrap{border-radius:18px;overflow:hidden;border:1px solid var(--border);background:var(--bg3);
    box-shadow:0 18px 40px -16px rgba(0,0,0,.22);
    transition:transform .35s cubic-bezier(.22,1,.36,1),box-shadow .35s cubic-bezier(.22,1,.36,1);}
  .post-img-wrap:hover{transform:translateY(-7px) scale(1.015);box-shadow:0 30px 60px -18px rgba(0,0,0,.3);}
  .post-img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--bg3);}
  .hook{font-family:'Sora',sans-serif!important;font-size:clamp(1.2rem,3vw,1.65rem);font-weight:800;color:var(--text);letter-spacing:-.03em;margin:0 0 18px;line-height:1.2;}
  .vo-label,.cap-label{font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text4);margin:22px 0 7px;display:flex;align-items:center;gap:7px;}
  .vo-label::after,.cap-label::after{content:'';flex:1;height:1px;background:var(--border);}
  .vo-script{font-size:14.5px;color:var(--text3);font-style:italic;line-height:1.75;border-left:2px solid var(--border2);padding-left:14px;margin:0 0 8px;}
  .cap-text{font-size:13.5px;color:var(--text3);line-height:1.7;white-space:pre-line;}
  .tags{margin-top:14px;display:flex;flex-wrap:wrap;gap:5px;}
  .tag{background:var(--bg3);color:var(--text3);font-size:11.5px;font-weight:600;padding:4px 10px;border-radius:6px;border:1px solid var(--border);transition:all .15s;}
  .tag:hover{background:var(--border);color:var(--text);transform:translateY(-1px);}

  /* DOWNLOAD */
  div[data-testid="stDownloadButton"] button{border-radius:9px!important;font-weight:600!important;border:1px solid var(--border)!important;color:var(--text2)!important;background:var(--bg2)!important;padding:9px 18px!important;font-size:13px!important;width:100%;transition:all .15s;letter-spacing:-.01em!important;}
  div[data-testid="stDownloadButton"] button:hover{background:var(--bg3)!important;border-color:var(--border2)!important;color:var(--text)!important;transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.06);}

  /* EXPANDER */
  div[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:14px!important;background:var(--bg2)!important;position:relative;z-index:2;}

  /* CTA */
  .cta-dark{margin:100px auto 0;max-width:1100px;background:var(--text);border-radius:24px;padding:clamp(48px,7vw,84px);text-align:center;position:relative;overflow:hidden;z-index:2;}
  .cta-dark::before,.cta-dark::after{display:none;}
  .cta-dark h2{font-family:'Sora',sans-serif!important;color:#fff;font-size:clamp(1.9rem,5vw,3.2rem);font-weight:800;letter-spacing:-.04em;margin:0 0 14px;position:relative;z-index:1;line-height:1.1;}
  .cta-dark p{color:rgba(255,255,255,.5);font-size:16px;max-width:500px;margin:0 auto;line-height:1.75;position:relative;z-index:1;font-weight:400;}

  /* FOOTER */
  .footer{margin-top:80px;border-top:1px solid var(--border);padding:36px 0 18px;position:relative;z-index:2;}
  .footer-inner{max-width:1100px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;}
  .footer .logo{font-size:16px;}
  .footer p{color:var(--text4);font-size:13px;margin:0;}
  .footer p b{color:var(--text3);font-weight:600;}

  /* ANIMATIONS */
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:none;}}
  .anim{animation:fadeUp .7s cubic-bezier(.22,1,.36,1) both;}
  .d1{animation-delay:.08s;}.d2{animation-delay:.16s;}.d3{animation-delay:.24s;}.d4{animation-delay:.32s;}

  /* STREAMLIT */
  div[data-testid="stAudio"]{border-radius:10px;overflow:hidden;border:1px solid var(--border);}
  div[data-testid="stMarkdownContainer"]{color:var(--text2)!important;}
  div[data-testid="stMarkdownContainer"] strong{color:var(--text)!important;}
  div[data-testid="stMarkdownContainer"] em{color:var(--text3)!important;}
  .stAlert{border-radius:10px!important;border:1px solid var(--border)!important;background:var(--bg2)!important;}

  /* RESPONSIVE */
  @media(max-width:900px){
    .nav{margin:0 -16px;padding:0 16px;}
    .nav-links{display:none;}
    .pitch-grid,.how-grid,.bundle-grid{grid-template-columns:1fr;}
    .how-grid{gap:0;}
    .post-wrap{grid-template-columns:1fr;}
    .stat{padding:18px 20px;}
    .ticker{margin:56px -16px 0;}
    .stats{border-radius:14px;}
  }
</style>
<div class="bg-canvas"></div>
<div class="grid-overlay"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
<div class="noise"></div>
""")

# ================= NAV =================
st.html("""
<nav class="nav"><div class="nav-inner">
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div class="logo-text">Hype<span>Repo</span></div>
  </div>
  <div class="nav-links">
    <a href="#how">How it works</a>
    <a href="#bundle">What you get</a>
    <span class="nav-badge">AI-Powered</span>
  </div>
  <a class="nav-cta" href="#top">Generate Kit ✦</a>
</div></nav>
""")

# ================= HERO =================
st.html("""
<div class="hero" id="top">
  <div class="badge anim"><span class="pulse-dot"></span>AI Marketing Agent &nbsp;·&nbsp; Zero setup</div>
  <h1 class="anim d1">Turn any repo into<br><span class="serif-accent">a full launch kit.</span></h1>
  <p class="sub anim d2">Paste a GitHub URL and walk away with pitch cards, social posts with AI visuals, voiceovers, and a 2-minute demo pitch — generated from your actual code, not a template.</p>
</div>
""")

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
    for chunk in video_agent.stream({"repo_url": repo_url.strip(), "out_dir": out_dir}):
        for node, update in chunk.items():
            if update:
                final.update(update)
            done.append(node)
            nxt = next((k for k, _, _ in STEPS if k not in done), None)
            steps_ph.markdown(render_steps(done, active=nxt), unsafe_allow_html=True)
    steps_ph.markdown(render_steps(done), unsafe_allow_html=True)

    m = re.search(r"github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)", repo_url)
    repo_name = html.escape(m.group(2)) if m else "project"

    brief = final.get("concept_brief", {})
    with st.expander("🔍 What the AI understood about this repo"):
        st.markdown(f"**{brief.get('one_liner', '')}**")
        st.write(brief.get("concept", ""))
        feats = brief.get("key_features", [])
        if feats:
            st.markdown("**Key features spotted in the code:**")
            for f in feats:
                st.markdown(f"- {f}")

    demo = final.get("demo_pitch", {})
    st.markdown('<div class="sec-title anim">🎤 The 2-Minute Demo Pitch</div>'
                '<div class="sec-sub anim d1">Voiceover-ready script with studio-quality AI audio.</div>', unsafe_allow_html=True)
    paras = "".join(f"<p class='script'>{html.escape(p.strip())}</p>"
                    for p in demo.get("script", "").split("\n\n") if p.strip())
    st.markdown(f"""<div class="demo-card anim d1"><div class="demo-kicker">★ FEATURED</div>
        <h2>{html.escape(demo.get('title', 'Demo Pitch'))}</h2>{paras}</div>""", unsafe_allow_html=True)
    if demo.get("audio_path") and os.path.exists(demo["audio_path"]):
        with open(demo["audio_path"], "rb") as f:
            demo_bytes = f.read()
        st.audio(demo_bytes, format="audio/mpeg")
        st.download_button("⬇️ Download demo pitch audio", demo_bytes, file_name="demo_pitch.mp3")
    elif demo.get("audio_error"):
        st.warning(f"🎙️ Demo voiceover failed: {demo['audio_error'][:250]}")

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
    _d_ok = bool(demo.get("audio_path") and os.path.exists(demo["audio_path"]))
    _rep.append(f"{'✅' if _d_ok else '❌'} Demo pitch voiceover")
    st.markdown("<div class='sec-title anim'>🧾 Generation report</div>"
                "<div class='sec-sub anim d1'>What actually got built — no silent failures.</div>",
                unsafe_allow_html=True)
    for _line in _rep:
        st.markdown(f"- {_line}")

# ================= STATS =================
st.html("""
<div class="stats anim">
  <div class="stat"><b>3</b><span>AI visuals</span></div>
  <div class="stat"><b>4</b><span>Voiceovers</span></div>
  <div class="stat"><b>3</b><span>Pitch cards</span></div>
  <div class="stat"><b>1</b><span>Demo pitch</span></div>
</div>
""")

# ================= TICKER =================
st.html("""
<div class="ticker"><div class="ticker-track">
  <span class="tick"><em>✦</em>PITCH CARDS</span><span class="tick"><em>✦</em>AI VOICEOVERS</span><span class="tick"><em>✦</em>SCROLL-STOPPING VISUALS</span><span class="tick"><em>✦</em>2-MINUTE DEMO PITCH</span><span class="tick"><em>✦</em>CAPTIONS & HASHTAGS</span><span class="tick"><em>✦</em>ZERO EDITING NEEDED</span>
  <span class="tick"><em>✦</em>PITCH CARDS</span><span class="tick"><em>✦</em>AI VOICEOVERS</span><span class="tick"><em>✦</em>SCROLL-STOPPING VISUALS</span><span class="tick"><em>✦</em>2-MINUTE DEMO PITCH</span><span class="tick"><em>✦</em>CAPTIONS & HASHTAGS</span><span class="tick"><em>✦</em>ZERO EDITING NEEDED</span>
</div></div>
""")

# ================= HOW IT WORKS =================
st.html("""
<div class="section" id="how">
  <div class="kicker">HOW IT WORKS</div>
  <div class="sec-h">Repo link to launch kit<br>in under five minutes.</div>
  <p class="sec-p">No prompts. No templates. A five-stage AI pipeline reads your code, understands what you built, then writes, designs, and records everything.</p>
  <div class="how-grid">
    <div class="how-card anim"><div class="how-num">01</div><h3>🔗 Paste your repo URL</h3><p>Drop any public GitHub URL. The agent fetches your README, repo metadata, and actual source files — not just the docs.</p></div>
    <div class="how-card anim d1"><div class="how-num">02</div><h3>🧠 Deep code analysis</h3><p>A senior-engineer-grade LLM builds a concept brief: what you built, how it works, who it's for, and what makes it genuinely different.</p></div>
    <div class="how-card anim d2"><div class="how-num">03</div><h3>🚀 Ship the full kit</h3><p>Pitch cards, social posts with AI visuals and voiceovers, plus a structured 2-minute demo pitch with studio audio. One click to download all.</p></div>
  </div>
</div>
""")

# ================= BUNDLE =================
st.html("""
<div class="section" id="bundle">
  <div class="kicker">WHAT YOU GET</div>
  <div class="sec-h">Everything a launch needs.<br>Nothing it doesn't.</div>
  <p class="sec-p">Every asset is grounded in your actual code — never generic filler. Built from a real concept brief, not a template.</p>
  <div class="bundle-grid">
    <div class="bundle-card anim"><div class="bundle-icon">✨</div><h3>Pitch Cards</h3><p>Three razor-sharp angles with concrete proof points — ready for your README or landing page hero.</p></div>
    <div class="bundle-card anim d1"><div class="bundle-icon">📱</div><h3>Social Posts</h3><p>Problem → magic moment → proof. Scroll-stopping hooks, captions, and hashtags — three distinct angles.</p></div>
    <div class="bundle-card anim d2"><div class="bundle-icon">🎨</div><h3>AI Visuals</h3><p>Custom 1:1 promo graphics per post, generated from prompts based on your project's real subject matter.</p></div>
    <div class="bundle-card anim d3"><div class="bundle-icon">🎙️</div><h3>Voiceovers</h3><p>Natural-sounding AI narration for every post and the full demo pitch — no microphone, no studio needed.</p></div>
    <div class="bundle-card anim d4"><div class="bundle-icon">🎤</div><h3>2-Min Demo Pitch</h3><p>Cold open → problem → walkthrough → features → CTA. Structured like a real demo day, with full audio.</p></div>
    <div class="bundle-card anim"><div class="bundle-icon">⬇️</div><h3>Instant Download Kit</h3><p>Every image and MP3 is one click away. Take the whole bundle straight to your content scheduler.</p></div>
  </div>
</div>
""")

# ================= DARK CTA + FOOTER =================
st.html("""
<div class="cta-dark">
  <h2>Your repo deserves more<br>than <span class="serif-accent">a README.</span></h2>
  <p>Paste a link above and walk away with a complete launch kit — copy, visuals, voiceovers, and a demo pitch. Powered by real code analysis.</p>
</div>
<div class="footer"><div class="footer-inner">
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div class="logo-text">Hype<span>Repo</span></div>
  </div>
  <p>Built with <b>HypeRepo</b> — paste a repo, ship the hype. © 2026</p>
</div></div>
""")
