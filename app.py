import os
import re
import json
import html
import asyncio
import threading
import tempfile
import requests
import edge_tts
import streamlit as st
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="HypeRepo — AI Marketing Agent", page_icon="🚀", layout="centered")

# ---------- Secrets ----------
def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)

DO_API_KEY = get_secret("DO_API_KEY")
ALIBABA_API_KEY = get_secret("ALIBABA_API_KEY")
DO_URL = "https://inference.do-ai.run/v1/responses"
DO_MODEL = "openai-gpt-oss-20b"

# ---------- DESIGN SYSTEM (st.html so <style> is not stripped) ----------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  * { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; }
  .stApp { background: #FAFAF7; }
  #MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
  .block-container { max-width: 1080px !important; padding-top: 0 !important; padding-bottom: 60px; }

  /* ambient gradient blobs */
  .blob { position: fixed; border-radius: 50%; filter: blur(90px); z-index: 0; pointer-events: none; }
  .blob-1 { width: 480px; height: 480px; background: #DDD6FE; opacity: .55; top: -140px; left: -120px; }
  .blob-2 { width: 420px; height: 420px; background: #FBCFE8; opacity: .5; top: 20%; right: -140px; }
  .blob-3 { width: 380px; height: 380px; background: #C7D2FE; opacity: .4; bottom: -120px; left: 30%; }

  /* hero */
  .hero { text-align: center; padding: 64px 16px 8px; position: relative; z-index: 1; }
  .badge { display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: .18em;
           color: #6D28D9; background: #F3EFFF; border: 1px solid #E2D9FD;
           padding: 8px 18px; border-radius: 999px; margin-bottom: 22px; }
  .hero h1 { font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 800; letter-spacing: -.035em;
             line-height: 1.05; color: #18181B; margin: 0 0 16px; }
  .grad-text { background: linear-gradient(120deg, #6366F1, #A855F7 55%, #EC4899);
               -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero p.sub { font-size: clamp(1rem, 2.4vw, 1.2rem); color: #71717A; max-width: 620px; margin: 0 auto 8px; line-height: 1.6; }

  /* repo input pill */
  div[data-testid="stTextInput"] { max-width: 620px; margin: 26px auto 0; position: relative; z-index: 1; }
  div[data-testid="stTextInput"] label { display: none; }
  div[data-testid="stTextInput"] input { border-radius: 999px !important; padding: 17px 26px !important;
      font-size: 16px !important; border: 2px solid #E9E3D6 !important; background: #fff !important;
      box-shadow: 0 10px 30px -12px rgba(99,102,241,.25); color: #18181B; }
  div[data-testid="stTextInput"] input:focus { border-color: #8B5CF6 !important;
      box-shadow: 0 0 0 4px rgba(139,92,246,.15), 0 10px 30px -12px rgba(99,102,241,.35) !important; }

  /* generate button */
  div[data-testid="stButton"] { text-align: center; margin-top: 18px; position: relative; z-index: 1; }
  div[data-testid="stButton"] button { background: linear-gradient(120deg, #6366F1, #A855F7) !important;
      color: #fff !important; border: none !important; border-radius: 999px !important;
      padding: 15px 54px !important; font-size: 17px !important; font-weight: 700 !important;
      box-shadow: 0 14px 30px -10px rgba(139,92,246,.55); transition: transform .15s ease, box-shadow .15s ease; }
  div[data-testid="stButton"] button:hover { transform: translateY(-2px) scale(1.02);
      box-shadow: 0 20px 40px -10px rgba(139,92,246,.65); }
  div[data-testid="stButton"] button:active { transform: scale(.98); }

  /* pipeline steps */
  .steps { display: flex; gap: 6px; justify-content: center; margin: 34px auto 10px;
           max-width: 860px; position: relative; z-index: 1; flex-wrap: wrap; }
  .step { display: flex; align-items: center; gap: 10px; background: #fff; border: 1.5px solid #ECE7DA;
          border-radius: 999px; padding: 10px 18px 10px 10px; font-size: 13.5px; font-weight: 600; color: #A1A1AA; }
  .step .dot { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center;
               justify-content: center; background: #F4F2EC; font-size: 14px; flex-shrink: 0; }
  .step.done { color: #18181B; border-color: #D9F2E3; background: #F4FBF6; }
  .step.done .dot { background: linear-gradient(135deg, #34D399, #10B981); color: #fff; }
  .step.active { color: #18181B; border-color: #C4B5FD; background: #FAF8FF;
                 box-shadow: 0 0 0 4px rgba(139,92,246,.12); }
  .step.active .dot { background: linear-gradient(135deg, #6366F1, #A855F7); color: #fff; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.15); } }

  /* section titles */
  .sec-title { font-size: clamp(1.5rem, 3.5vw, 2rem); font-weight: 800; letter-spacing: -.02em;
               color: #18181B; margin: 54px 0 6px; position: relative; z-index: 1; }
  .sec-sub { color: #71717A; margin-bottom: 24px; position: relative; z-index: 1; }

  /* demo pitch dark card */
  .demo-card { background: #101014; border-radius: 26px; padding: clamp(22px, 4vw, 40px); color: #fff;
               position: relative; overflow: hidden; z-index: 1;
               box-shadow: 0 30px 60px -20px rgba(0,0,0,.45); margin-top: 18px; }
  .demo-card::before { content: ""; position: absolute; width: 420px; height: 420px; border-radius: 50%;
      background: radial-gradient(circle, rgba(139,92,246,.35), transparent 70%); top: -160px; right: -120px; }
  .demo-card::after { content: ""; position: absolute; width: 340px; height: 340px; border-radius: 50%;
      background: radial-gradient(circle, rgba(236,72,153,.22), transparent 70%); bottom: -140px; left: -100px; }
  .demo-kicker { display: inline-block; font-size: 11.5px; font-weight: 700; letter-spacing: .16em;
                 color: #FBBF24; margin-bottom: 10px; position: relative; z-index: 1; }
  .demo-card h2 { font-size: clamp(1.4rem, 3.4vw, 2rem); font-weight: 800; margin: 0 0 18px;
                  position: relative; z-index: 1; letter-spacing: -.02em; }
  .demo-card p.script { color: #D4D4D8; line-height: 1.75; font-size: 15.5px; position: relative; z-index: 1; }
  .demo-card audio { width: 100%; margin: 6px 0 20px; position: relative; z-index: 1; }

  /* pitch cards grid */
  .pitch-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; position: relative; z-index: 1; }
  .pitch-card { background: #fff; border: 1.5px solid #EFEAE0; border-radius: 20px; padding: 26px 22px;
                box-shadow: 0 12px 30px -14px rgba(0,0,0,.12); position: relative; overflow: hidden;
                animation: fadeUp .6s ease both; }
  .pitch-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 5px;
      background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899); }
  .pitch-card .icon { font-size: 30px; margin-bottom: 12px; }
  .pitch-card h3 { font-size: 17px; font-weight: 800; color: #18181B; margin: 0 0 8px; letter-spacing: -.01em; }
  .pitch-card p { font-size: 14px; color: #71717A; line-height: 1.6; margin: 0; }

  /* social post layout */
  .post-wrap { display: grid; grid-template-columns: 320px 1fr; gap: 34px; align-items: start;
               background: #fff; border: 1.5px solid #EFEAE0; border-radius: 26px;
               padding: clamp(20px, 4vw, 36px); margin-bottom: 26px; position: relative; z-index: 1;
               box-shadow: 0 18px 44px -20px rgba(0,0,0,.14); animation: fadeUp .6s ease both; }
  .post-num { position: absolute; top: -16px; left: 28px; background: linear-gradient(120deg,#6366F1,#A855F7);
              color: #fff; font-size: 12px; font-weight: 800; letter-spacing: .1em;
              padding: 7px 16px; border-radius: 999px; box-shadow: 0 8px 18px -6px rgba(139,92,246,.6); }

  /* phone mockup */
  .phone { width: 280px; margin: 14px auto 0; background: #0c0c0e; border-radius: 46px; padding: 11px;
           box-shadow: 0 34px 60px -18px rgba(0,0,0,.4), 0 0 0 2px #2b2b30; }
  .phone-screen { background: #fff; border-radius: 36px; overflow: hidden; position: relative; }
  .notch { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); width: 100px; height: 24px;
           background: #0c0c0e; border-radius: 999px; z-index: 2; }
  .ig-head { display: flex; align-items: center; gap: 10px; padding: 40px 14px 10px; }
  .ig-avatar { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
               background: linear-gradient(135deg,#6366F1,#EC4899); display: flex; align-items: center;
               justify-content: center; color: #fff; font-size: 16px; font-weight: 800; }
  .ig-user { font-size: 13px; font-weight: 700; color: #18181B; }
  .ig-img { width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; background: #f4f4f5; }
  .ig-actions { display: flex; gap: 14px; padding: 12px 14px 4px; font-size: 20px; }
  .ig-cap { padding: 6px 14px 18px; font-size: 12.5px; color: #3f3f46; line-height: 1.55; }
  .ig-cap b { color: #18181B; }

  /* post details */
  .hook { font-size: clamp(1.2rem, 3vw, 1.6rem); font-weight: 800; color: #18181B;
          letter-spacing: -.02em; margin: 6px 0 14px; line-height: 1.25; }
  .vo-label { font-size: 12px; font-weight: 700; letter-spacing: .12em; color: #8B5CF6; margin: 18px 0 6px; }
  .vo-script { font-size: 15px; color: #3f3f46; font-style: italic; line-height: 1.65;
               border-left: 3px solid #DDD6FE; padding-left: 14px; margin: 0 0 8px; }
  .cap-label { font-size: 12px; font-weight: 700; letter-spacing: .12em; color: #8B5CF6; margin: 18px 0 6px; }
  .cap-text { font-size: 14.5px; color: #52525B; line-height: 1.65; white-space: pre-line; }
  .tags { margin-top: 12px; }
  .tag { display: inline-block; background: #F3EFFF; color: #6D28D9; font-size: 12.5px; font-weight: 700;
         padding: 6px 13px; border-radius: 999px; margin: 0 6px 6px 0; }

  /* download buttons */
  div[data-testid="stDownloadButton"] button { border-radius: 999px !important; font-weight: 700 !important;
      border: 2px solid #E4DEF5 !important; color: #6D28D9 !important; background: #fff !important;
      padding: 9px 22px !important; font-size: 13.5px !important; width: 100%; }
  div[data-testid="stDownloadButton"] button:hover { background: #F5F2FF !important; border-color: #8B5CF6 !important; }

  /* expander */
  div[data-testid="stExpander"] { border: 1.5px solid #EFEAE0 !important; border-radius: 16px !important;
      background: #fff !important; position: relative; z-index: 1; }

  .footer { text-align: center; color: #A1A1AA; font-size: 13px; margin-top: 70px; position: relative; z-index: 1; }
  .footer b { color: #6D28D9; }

  @keyframes fadeUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }
  .anim { animation: fadeUp .7s ease both; }
  .d1 { animation-delay: .08s; } .d2 { animation-delay: .16s; } .d3 { animation-delay: .24s; }

  @media (max-width: 820px) {
    .pitch-grid { grid-template-columns: 1fr; }
    .post-wrap { grid-template-columns: 1fr; }
    .phone { width: 250px; }
    .steps { justify-content: flex-start; }
  }
</style>
<div class="blob blob-1"></div><div class="blob blob-2"></div><div class="blob blob-3"></div>
""")

# ---------- Backend (unchanged brain) ----------
def tts_to_mp3(text, out_path, voice="en-US-RogerNeural"):
    def _run():
        async def _main():
            rate = "+0%" if len(text) > 500 else "+5%"
            await edge_tts.Communicate(text, voice=voice, rate=rate).save(out_path)
        asyncio.run(_main())
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join()

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
        cands.sort(key=lambda n: 0 if any(k in n.lower() for k in ["main", "index", "app", "cli", "server"]) else 1)
        for name in cands[:4]:
            grab(name)
        subdirs = [x["name"] for x in items if x.get("type") == "dir" and x["name"] not in SKIP_DIRS]
        for sd in ["src", "lib", "app", "pkg", "components"]:
            if sd in subdirs:
                try:
                    sub = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{sd}", timeout=20).json()
                    if isinstance(sub, list):
                        sfiles = [x["name"] for x in sub
                                  if x.get("type") == "file" and x["name"].lower().endswith(CODE_EXTS)][:3]
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
        tts_to_mp3(post.get("script", ""), audio_path)
        post["audio_path"] = audio_path
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
            img_res = requests.get(img_url, timeout=60)
            img_res.raise_for_status()
            img_path = os.path.join(out_dir, f"post_{i}.png")
            with open(img_path, "wb") as f:
                f.write(img_res.content)
            post["img_path"] = img_path
        except Exception as e:
            post["img_path"] = None
            post["img_error"] = str(e)
    demo = state.get("demo_pitch", {})
    if demo.get("script"):
        demo_path = os.path.join(out_dir, "demo_pitch.mp3")
        tts_to_mp3(demo["script"], demo_path)
        demo["audio_path"] = demo_path
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

# ---------- UI ----------
st.markdown('<div class="hero"><div class="badge">AI MARKETING AGENT</div>'
           '<h1>Turn any repo into<br><span class="grad-text">a launch campaign.</span></h1>'
           '<p class="sub">Paste a GitHub URL. HypeRepo reads the code, understands the project, '
           'then generates pitch cards, social posts with AI images & voiceovers, and a 2-minute demo pitch.</p></div>',
           unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("🔑 Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

repo_url = st.text_input("repo", placeholder="https://github.com/owner/repo  —  paste any repo URL")

if st.button("✨ Generate marketing bundle"):
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

    # What the AI understood
    brief = final.get("concept_brief", {})
    with st.expander("🔍 What the AI understood about this repo"):
        st.markdown(f"**{brief.get('one_liner', '')}**")
        st.write(brief.get("concept", ""))
        feats = brief.get("key_features", [])
        if feats:
            st.markdown("**Key features spotted in the code:**")
            for f in feats:
                st.markdown(f"- {f}")

    # Demo pitch — dark feature card
    demo = final.get("demo_pitch", {})
    st.markdown('<div class="sec-title anim">🎤 The 2-Minute Demo Pitch</div>'
                '<div class="sec-sub anim d1">Your voiceover-ready script, with full audio.</div>', unsafe_allow_html=True)
    paras = "".join(f"<p class='script'>{html.escape(p.strip())}</p>"
                    for p in demo.get("script", "").split("\n\n") if p.strip())
    st.markdown(f"""<div class="demo-card anim d1"><div class="demo-kicker">★ FEATURED</div>
        <h2>{html.escape(demo.get('title', 'Demo Pitch'))}</h2>
        {paras}</div>""", unsafe_allow_html=True)
    if demo.get("audio_path") and os.path.exists(demo["audio_path"]):
        with open(demo["audio_path"], "rb") as f:
            demo_bytes = f.read()
        st.audio(demo_bytes, format="audio/mpeg")
        st.download_button("⬇️ Download demo pitch audio", demo_bytes, file_name="demo_pitch.mp3")

    # Pitch cards
    st.markdown('<div class="sec-title anim">✨ Pitch Cards</div>'
                '<div class="sec-sub anim d1">The three strongest angles, ready for your README or landing page.</div>',
                unsafe_allow_html=True)
    icons = ["⚡", "🎯", "💎"]
    cards_html = "".join(
        f"""<div class="pitch-card d{i+1}"><div class="icon">{icons[i % 3]}</div>
            <h3>{html.escape(c.get('headline', ''))}</h3><p>{html.escape(c.get('sub', ''))}</p></div>"""
        for i, c in enumerate(final.get("pitch_cards", [])))
    st.markdown(f'<div class="pitch-grid">{cards_html}</div>', unsafe_allow_html=True)

    # Social posts — phone mockups
    st.markdown('<div class="sec-title anim">📱 Social Posts</div>'
                '<div class="sec-sub anim d1">Hook, voiceover, AI visual, caption & hashtags — previewed like real posts.</div>',
                unsafe_allow_html=True)
    for i, post in enumerate(final.get("social_posts", []), 1):
        img_tag = ""
        img_bytes = None
        if post.get("img_path") and os.path.exists(post["img_path"]):
            with open(post["img_path"], "rb") as f:
                img_bytes = f.read()
            import base64 as _b64
            img_tag = f'<img class="ig-img" src="data:image/png;base64,{_b64.b64encode(img_bytes).decode()}" />'
        else:
            img_tag = ('<div class="ig-img" style="display:flex;align-items:center;justify-content:center;'
                       'color:#a1a1aa;font-size:13px;padding:20px;text-align:center;">'
                       f'⚠ image failed<br>{html.escape(post.get("img_error", ""))[:120]}</div>')
        cap_preview = html.escape(post.get("caption", "").split("\n")[0])[:110]
        hook = html.escape(post.get("hook", ""))
        st.markdown(f"""<div class="post-wrap d{(i % 3) + 1}">
          <div class="post-num">POST {i}</div>
          <div><div class="phone"><div class="phone-screen"><div class="notch"></div>
            <div class="ig-head"><div class="ig-avatar">⚡</div><div class="ig-user">{repo_name}</div></div>
            {img_tag}
            <div class="ig-actions"><span>♡</span><span>💬</span><span>↗</span></div>
            <div class="ig-cap"><b>{repo_name}</b> {cap_preview}…</div>
          </div></div></div>
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
        c1, c2 = st.columns(2)
        with c1:
            if img_bytes:
                st.download_button(f"⬇️ Image {i}", img_bytes, file_name=f"post_{i}.png", key=f"dl_img_{i}")
        with c2:
            if a_bytes:
                st.download_button(f"⬇️ Voiceover {i}", a_bytes, file_name=f"post_{i}.mp3", key=f"dl_aud_{i}")

st.markdown('<div class="footer">Built with <b>HypeRepo</b> 🚀 — paste a repo, ship the hype.</div>', unsafe_allow_html=True)
