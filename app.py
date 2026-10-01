import asyncio
import json
import os
import re
import tempfile
import threading
from typing import Dict, List, TypedDict

import edge_tts
import requests
import streamlit as st
from langgraph.graph import END, StateGraph

st.set_page_config(page_title="AI Marketing Agent", page_icon="🚀", layout="centered")


def get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


DO_API_KEY = get_secret("DO_API_KEY")
ALIBABA_API_KEY = get_secret("ALIBABA_API_KEY")
DO_URL = "https://inference.do-ai.run/v1/responses"
DO_MODEL = "openai-gpt-oss-20b"


def tts_to_mp3(text: str, out_path: str, voice: str = "en-US-RogerNeural"):
    async def generate():
        rate = "+0%" if len(text) > 500 else "+5%"
        await edge_tts.Communicate(text, voice=voice, rate=rate).save(out_path)

    def run():
        asyncio.run(generate())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()


def do_call(prompt: str, max_tokens: int, temperature: float, retry: bool = True):
    response = requests.post(
        DO_URL,
        headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"},
        json={"model": DO_MODEL, "input": prompt, "max_output_tokens": max_tokens, "temperature": temperature, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    response_data = response.json()

    if response_data.get("choices"):
        raw_content = response_data["choices"][0].get("message", {}).get("content", "")
    else:
        raw_content = response_data.get("output", "") or response_data.get("text", "")
    if isinstance(raw_content, list):
        for block in raw_content:
            if isinstance(block, dict) and block.get("role") == "assistant":
                content = block.get("content", [])
                if isinstance(content, list) and content:
                    raw_content = content[0].get("text", "")
                    break
    if not isinstance(raw_content, str):
        raw_content = json.dumps(raw_content)

    clean_text = re.sub(r"```(?:json)?", "", raw_content).strip()
    match = re.search(r"\{.*\}", clean_text, re.DOTALL)
    json_text = match.group(0) if match else clean_text
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        if not retry:
            raise
        repair_prompt = (
            "Your previous response was not valid JSON. Return the complete object again as valid JSON only, "
            "with every field and full text, without markdown.\n\nBroken output to fix:\n" + json_text[:6000]
        )
        return do_call(repair_prompt, max_tokens, temperature, retry=False)


class AgentState(TypedDict, total=False):
    repo_url: str
    out_dir: str
    readme_text: str
    repo_context: str
    code_context: str
    concept_brief: Dict
    pitch_cards: List[Dict]
    social_posts: List[Dict]
    demo_pitch: Dict


CODE_EXTENSIONS = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php", ".swift", ".kt")
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".next", "vendor", ".idea", ".vscode"}


def raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    try:
        response = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", timeout=15)
        if response.status_code == 200 and response.text.strip():
            return response.text
    except requests.RequestException:
        pass
    return ""


def fetch_repo(state: AgentState):
    match = re.search(r"https?://github\.com/([\w.-]+)/([\w.-]+)", state["repo_url"])
    if not match:
        raise ValueError("Could not parse GitHub repo URL")
    owner, repo = match.groups()

    readme = ""
    for branch in ("main", "master"):
        response = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md", timeout=30)
        if response.status_code == 200:
            readme = response.text[:4000]
            break

    metadata = ""
    tree = ""
    code_chunks = []

    def grab(path: str):
        for branch in ("main", "master"):
            text = raw_file(owner, repo, branch, path)
            if text:
                code_chunks.append(f"--- {path} ---\n{text[:2000]}")
                return

    try:
        metadata_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=20)
        metadata_data = metadata_response.json()
        metadata = (
            f"META: {metadata_data.get('description', '')} | stars: {metadata_data.get('stargazers_count', '?')} | "
            f"lang: {metadata_data.get('language', '?')} | topics: {', '.join(metadata_data.get('topics', [])[:8])}"
        )
        top_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/", timeout=20)
        items = top_response.json() if top_response.ok else []
        if not isinstance(items, list):
            items = []
        tree = ", ".join(item.get("name", "") for item in items[:30])
        candidates = [
            item["name"] for item in items
            if item.get("type") == "file" and item.get("name", "").lower().endswith(CODE_EXTENSIONS)
        ]
        candidates.sort(key=lambda name: 0 if any(key in name.lower() for key in ("main", "index", "app", "cli", "server")) else 1)
        for name in candidates[:4]:
            grab(name)

        subdirs = [item["name"] for item in items if item.get("type") == "dir" and item.get("name") not in SKIP_DIRS]
        for directory in ("src", "lib", "app", "pkg", "components"):
            if directory not in subdirs:
                continue
            sub_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}", timeout=20)
            subitems = sub_response.json() if sub_response.ok else []
            if isinstance(subitems, list):
                for item in subitems:
                    if len(code_chunks) >= 7:
                        break
                    if item.get("type") == "file" and item.get("name", "").lower().endswith(CODE_EXTENSIONS):
                        grab(f"{directory}/{item['name']}")
            break
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    manifest = ""
    for filename in ("package.json", "pyproject.toml", "setup.py", "Cargo.toml", "go.mod"):
        text = raw_file(owner, repo, "main", filename) or raw_file(owner, repo, "master", filename)
        if len(text) > 50:
            manifest = f"[{filename}]\n{text[:1200]}"
            break

    return {"readme_text": readme, "repo_context": f"{metadata}\nFILES: {tree}\n{manifest}", "code_context": "\n\n".join(code_chunks)}


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
{{
  "one_liner": "what it is, in one punchy sentence",
  "concept": "2-3 sentences: the core idea, explained simply",
  "how_it_works": "3-5 sentences, technically concrete: what the user actually does step by step, and what the code does under the hood",
  "key_features": ["concrete feature with a specific detail", "up to 6 total, most impressive first"],
  "audience": "who this is for, specifically",
  "differentiator": "what makes it different from alternatives — or 'not clear from context' if honestly unknown",
  "vibe": "the project's personality/aesthetic in ~5 words"
}}

RULES:
- Only state what the context supports. If something is a guess, say so inline.
- Be concrete: name real commands, real file types, real behaviors you saw in the code — never generic filler.
"""
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
- IMAGE PROMPTS are self-contained prompts for a square 1:1 social promo graphic depicting THIS project's actual subject matter. MUST include the exact on-image headline in "quotes" (max 5 words), describe scene, composition, style, lighting, colors, and end with: "No watermark, no extra text, no garbled letters."

Output ONLY a valid JSON object matching this exact schema:
{{
  "pitch_cards": [
    {{"headline": "punchy benefit, max 6 words", "sub": "one concrete sentence with a real feature or proof point"}},
    {{"headline": "...", "sub": "..."}},
    {{"headline": "...", "sub": "..."}}
  ],
  "social_posts": [
    {{"hook": "scroll-stopper, max 8 words", "script": "25-35 word spoken script: hook, one concrete capability, call to action", "image_prompt": "full image generation prompt", "caption": "post caption: hook line, 2-3 value lines, call to action", "hashtags": ["#tag1", "#tag2", "#tag3"]}},
    {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}},
    {{"hook": "...", "script": "...", "image_prompt": "...", "caption": "...", "hashtags": ["#tag1", "#tag2", "#tag3"]}}
  ]
}}

The 3 posts cover 3 angles IN ORDER: 1) the painful problem, 2) the magic moment of using it, 3) proof + call to action (star the repo).
"""
    package = do_call(posts_prompt, max_tokens=3000, temperature=0.3)
    normalized_cards = [card if isinstance(card, dict) else {"headline": str(card), "sub": ""} for card in package.get("pitch_cards", [])]

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
{{"demo_pitch": {{"title": "title of the 2-minute demo", "script": "..."}}}}
"""
    demo_data = do_call(demo_prompt, max_tokens=2000, temperature=0.3)
    return {"pitch_cards": normalized_cards, "social_posts": package.get("social_posts", []), "demo_pitch": demo_data.get("demo_pitch", {})}


def generate_assets(state: AgentState):
    posts = state["social_posts"]
    for index, post in enumerate(posts):
        audio_path = os.path.join(state["out_dir"], f"post_{index}.mp3")
        tts_to_mp3(post.get("script", ""), audio_path)
        post["audio_path"] = audio_path
        try:
            image_response = requests.post(
                "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
                json={"model": "qwen-image-max", "input": {"messages": [{"role": "user", "content": [{"text": post["image_prompt"]}]}]}, "parameters": {"size": "1328*1328", "n": 1, "prompt_extend": True, "watermark": False}},
                timeout=180,
            )
            image_data = image_response.json()
            if image_response.status_code != 200:
                raise RuntimeError(f"Alibaba error {image_response.status_code}: {image_data.get('message', image_response.text)}")
            image_url = next((block["image"] for block in image_data["output"]["choices"][0]["message"]["content"] if isinstance(block, dict) and "image" in block), None)
            if not image_url:
                raise RuntimeError(f"No image in response: {image_data}")
            image_result = requests.get(image_url, timeout=60)
            image_result.raise_for_status()
            image_path = os.path.join(state["out_dir"], f"post_{index}.png")
            with open(image_path, "wb") as image_file:
                image_file.write(image_result.content)
            post["img_path"] = image_path
        except Exception as error:
            post["img_path"] = None
            post["img_error"] = str(error)

    demo = state.get("demo_pitch", {})
    if demo.get("script"):
        demo_audio_path = os.path.join(state["out_dir"], "demo_pitch.mp3")
        tts_to_mp3(demo["script"], demo_audio_path)
        demo["audio_path"] = demo_audio_path
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

STEP_LABELS = {
    "fetch_repo": "Reading repository (README + source files)",
    "understand_project": "Understanding the project deeply",
    "draft_strategy": "Drafting pitch cards, posts & demo script",
    "generate_assets": "Generating images & voiceovers (takes a few minutes)",
    "build_dashboard": "Assembling dashboard",
}

st.title("AI Marketing Agent")
st.write("Paste any GitHub repo URL and get pitch cards, social posts with AI images and voiceovers, and a 2-minute demo pitch with audio.")

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("Missing API keys. Add DO_API_KEY and ALIBABA_API_KEY to Streamlit secrets or environment variables.")
    st.stop()

repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")

if st.button("Generate Marketing Bundle", type="primary"):
    if not repo_url.strip() or "github.com" not in repo_url:
        st.error("Please paste a valid GitHub repo URL.")
        st.stop()

    output_dir = tempfile.mkdtemp(prefix="mktg_")
    final = {}
    with st.status("Running marketing agent...", expanded=True) as status:
        for chunk in video_agent.stream({"repo_url": repo_url.strip(), "out_dir": output_dir}):
            for node, update in chunk.items():
                st.write(f"Done: {STEP_LABELS.get(node, node)}")
                if update:
                    final.update(update)
        status.update(label="Done!", state="complete")

    demo = final.get("demo_pitch", {})
    st.header(f"Demo Pitch: {demo.get('title', '2-minute demo')}")
    if demo.get("audio_path") and os.path.exists(demo["audio_path"]):
        with open(demo["audio_path"], "rb") as audio_file:
            demo_bytes = audio_file.read()
        st.audio(demo_bytes, format="audio/mpeg")
        st.download_button("Download demo pitch audio", demo_bytes, file_name="demo_pitch.mp3")
    for paragraph in demo.get("script", "").split("\n\n"):
        if paragraph.strip():
            st.markdown(paragraph.strip())

    st.header("Pitch Cards")
    columns = st.columns(3)
    for card, column in zip(final.get("pitch_cards", []), columns):
        with column:
            st.subheader(card.get("headline", ""))
            st.write(card.get("sub", ""))

    st.header("Social Posts")
    for index, post in enumerate(final.get("social_posts", []), 1):
        st.subheader(f"Post {index}: {post.get('hook', '')}")
        st.markdown(f"**Voiceover:** *{post.get('script', '')}*")
        if post.get("img_path") and os.path.exists(post["img_path"]):
            with open(post["img_path"], "rb") as image_file:
                image_bytes = image_file.read()
            st.image(image_bytes)
            st.download_button(f"Download image {index}", image_bytes, file_name=f"post_{index}.png", key=f"img{index}")
        else:
            st.warning(f"Image failed: {post.get('img_error', 'unknown error')}")
        if post.get("audio_path") and os.path.exists(post["audio_path"]):
            with open(post["audio_path"], "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mpeg")
            st.download_button(f"Download voiceover {index}", audio_bytes, file_name=f"post_{index}.mp3", key=f"aud{index}")
        st.markdown(f"**Caption:**\n{post.get('caption', '')}")
        st.markdown(" ".join(post.get("hashtags", [])))
        st.divider()