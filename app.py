"""HypeRepo — fresh rewrite (2026-10-02).
Paste a GitHub URL, get:
  1. The short version (TL;DR)
  2. A how-it-works diagram (Mermaid flowchart, rendered as a clean image)
  3. A plain-English explanation of the repo
  4. An audio explainer with its script shown separately
  5. 4 Instagram hook images (text-based, B&W) + captions + hashtags
Prompting follows a research-backed playbook: strict JSON blocks, grounded roles,
concreteness rules, pre-assigned non-overlapping angles, one rubric-scored critique pass.
"""
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

# ================= CONFIG =================
DO_API_KEY = st.secrets.get("DO_API_KEY", "")
ALIBABA_API_KEY = st.secrets.get("ALIBABA_API_KEY", "")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")  # optional: raises GitHub API quota 60 -> 5000/hr
DO_MODEL = "openai-gpt-oss-20b"
QWEN_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
QWEN_MODEL = "qwen-image-max"

VOICE = "en-US-AndrewMultilingualNeural"
VOICE_FALLBACK = "en-US-RogerNeural"

STRICT_JSON = """
OUTPUT RULES (follow exactly):
- Return STRICT JSON ONLY. No prose before or after, no markdown fences, no comments.
- Double quotes on all keys and strings. No trailing commas.
- Include every required field. If a field is truly unknown, use "" or [] — never invent a value.
- Your response must start with { and end with }."""

REJECT_WORDS = ("revolutionary", "game-changing", "game changing", "cutting-edge",
                "cutting edge", "seamless", "powerful", "unlock", "elevate",
                "delve", "tapestry", "furthermore")

# ================= LLM =================
DO_CHAT_URL = "https://inference.do-ai.run/v1/chat/completions"
DO_RESP_URL = "https://inference.do-ai.run/v1/responses"  # fallback if chat endpoint rejects us

def _post(url, payload):
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=180)
    r.raise_for_status()
    return r.json()

def do_call(prompt, max_tokens, temperature, label="AI step", _tries=3):
    """Call the LLM and return parsed JSON.

    The real fix (not a retry curtain): the primary path uses the chat-completions
    endpoint with response_format={"type": "json_object"} — constrained decoding,
    so the model cannot return prose, markdown, or empty output. If that endpoint
    rejects the request (400/404/422), we fall back to /v1/responses automatically.
    The few retries that remain are only a safety net for network hiccups and
    truncated responses (which get more tokens on retry)."""
    original = prompt
    last_err = None
    raw_preview = ""
    use_chat = True
    for attempt in range(_tries):
        tok = int(max_tokens * (1.6 ** attempt))
        t = temperature if attempt == 0 else 0.0
        if attempt > 0:
            time.sleep(min(2 * attempt, 8))
        res_data = None
        if use_chat:
            try:
                res_data = _post(DO_CHAT_URL, {
                    "model": DO_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": tok, "temperature": t,
                    "response_format": {"type": "json_object"}, "stream": False})
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code in (400, 404, 422):
                    use_chat = False  # JSON mode unsupported here; fall back below
                else:
                    last_err = e
                    continue
            except (requests.RequestException, ValueError) as e:
                last_err = e
                continue
        if res_data is None:
            try:
                res_data = _post(DO_RESP_URL, {
                    "model": DO_MODEL, "input": prompt, "max_output_tokens": tok,
                    "temperature": t, "stream": False})
            except (requests.RequestException, ValueError) as e:
                last_err = e
                continue
        if not isinstance(res_data, dict):
            last_err = ValueError(f"unexpected API response shape: {type(res_data).__name__}")
            continue
        raw_content = ""
        if "choices" in res_data and len(res_data["choices"]) > 0:
            msg = res_data["choices"][0].get("message") or {}
            raw_content = msg.get("content", "") or ""
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
        raw_preview = raw_content[:200]
        clean_text = re.sub(r'```(?:json)?', '', raw_content).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        final_json = match.group(0) if match else clean_text
        try:
            if not final_json.strip():
                raise ValueError("empty model output")
            return json.loads(final_json)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            if final_json.strip():
                prompt = ("Your previous response was cut off. "
                          "Return the COMPLETE object again as valid JSON only, every field, full text, "
                          "no truncation, no markdown fences.\n\nBroken output:\n" + final_json[:6000])
            else:
                prompt = ("Your last response was empty. Output ONLY the complete JSON object now, "
                          "no other text, no explanations.\n\nTask:\n" + original[:8000])
    raise RuntimeError(f"{label} failed after {_tries} tries ({last_err}). "
                       f"Last model output: {raw_preview[:150]!r}")

# ================= SPEECH =================
def _clean_spoken(text):
    """Strip URLs/links/domains so the voiceover never reads them aloud."""
    t = re.sub(r"\[([^\]]+)\]\(\s*https?://\S+\s*\)", r"\1", text or "")
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\bwww\.\S+", "", t)
    t = re.sub(r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|io|dev|app|net|org|ai|co|me|site|page|link|gg|ly|to|so|xyz|tech)\b\S*", "", t)
    t = re.sub(r"\bwww\.(?=\s|$)", "", t)
    t = re.sub(r"\bhttps?\b", "", t)
    return re.sub(r"\s+", " ", t).strip()

def _to_ssml(text, voice):
    """Light SSML: pause after the hook + breathing room between paragraphs."""
    text = _clean_spoken(text)
    paras = [html.escape(" ".join(p.split())) for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [html.escape(" ".join(text.split()))]
    body = '<break time="600ms"/>'.join(paras)
    first_end = body.find(". ")
    if first_end != -1:
        body = body[:first_end + 1] + '<break time="450ms"/>' + body[first_end + 1:]
    return f'<speak version="1.0" xml:lang="en-US"><voice name="{voice}"><prosody rate="+0%">{body}</prosody></voice></speak>'

def _tts_bytes(ssml_text, voice):
    async def _run():
        # NOTE: edge_tts has no `ssml=` kwarg — the SSML string goes in as `text`
        # (the service sniffs the <speak> tag). Passing ssml= raises TypeError.
        communicate = edge_tts.Communicate(ssml_text, voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)
    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

def make_audio(script):
    """Returns (mp3_bytes, error_str). Never raises."""
    script = _clean_spoken(script)
    if not script:
        return None, "empty script"
    last_err = ""
    for voice in (VOICE, VOICE_FALLBACK):
        try:
            data = _tts_bytes(_to_ssml(script, voice), voice)
            if data:
                return data, ""
            last_err = f"{voice}: empty audio"
        except Exception as e:
            last_err = f"{voice}: {str(e)[:120]}"
    return None, last_err or "TTS failed"

# ================= IMAGES (Alibaba Qwen only) =================
def qwen_image(prompt):
    """Returns (png_bytes, error_str). Qwen only, no fallbacks. Never raises."""
    try:
        resp = requests.post(
            QWEN_URL,
            headers={"Authorization": f"Bearer {ALIBABA_API_KEY}", "Content-Type": "application/json"},
            json={"model": QWEN_MODEL,
                  "input": {"messages": [{"role": "user",
                                           "content": [{"text": prompt}]}]},
                  "parameters": {"size": "1328*1328", "n": 1}},
            timeout=180)
    except Exception as e:
        return None, f"request failed: {str(e)[:150]}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        data = resp.json()
    except Exception:
        return None, "non-JSON response"
    out = (data.get("output") or {})
    img_url = ""
    # shape 1 (observed live): output.choices[0].message.content[].image
    for ch in out.get("choices", []) or []:
        msg = (ch.get("message") or {})
        for item in msg.get("content", []) or []:
            if isinstance(item, dict) and item.get("image"):
                img_url = item["image"]
                break
        if img_url:
            break
    # shape 2 (documented): output.results[0].url
    if not img_url:
        results = out.get("results") or []
        if results:
            img_url = results[0].get("url", "")
    if not img_url:
        return None, f"no image in response: {json.dumps(data)[:200]}"
    url = img_url
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1]), ""
        except Exception as e:
            return None, f"bad data URL: {str(e)[:100]}"
    if url.startswith("http"):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                return r.content, ""
            return None, f"download HTTP {r.status_code}"
        except Exception as e:
            return None, f"download failed: {str(e)[:120]}"
    return None, f"unexpected image value: {url[:120]}"

# ================= DIAGRAM (Mermaid -> image) =================
def mermaid_image_url(code):
    b64 = base64.urlsafe_b64encode(code.encode("utf-8")).decode()
    return f"https://mermaid.ink/img/{b64}?theme=neutral"

def fetch_diagram(code):
    """Returns (png_bytes, error_str). Falls back gracefully; caller shows steps instead."""
    code = re.sub(r'```(?:mermaid)?', '', code or "").strip()
    if "flowchart" not in code and "graph" not in code:
        return None, "model did not return flowchart code"
    try:
        r = requests.get(mermaid_image_url(code), timeout=30)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, ""
        return None, f"render HTTP {r.status_code}"
    except Exception as e:
        return None, f"render failed: {str(e)[:120]}"

# ================= GITHUB =================
CODE_EXTS = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php", ".swift", ".kt")
SKIP_PARTS = ("node_modules", ".git/", "dist/", "build/", "__pycache__", ".next/",
              "vendor/", ".idea/", ".vscode/", ".min.js", ".min.css", ".d.ts", ".pyi",
              "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock")
JUNK_FILES = ("package-lock.json", "yarn.lock")

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def _gh_json(url):
    r = requests.get(url, timeout=25, headers=_gh_headers())
    return r.json() if r.status_code == 200 else None

def _gh_text(url):
    r = requests.get(url, timeout=25, headers={"Accept": "application/vnd.github.raw"})
    return r.text if r.status_code == 200 else ""

def fetch_repo(repo_url):
    """Returns dict with labeled context sections. Raises RuntimeError on failure."""
    m = re.search(r"github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)", repo_url or "")
    if not m:
        raise RuntimeError("not a valid GitHub repo URL")
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    base = f"https://api.github.com/repos/{owner}/{repo}"

    meta = _gh_json(base) or {}
    default_branch = meta.get("default_branch", "main")

    readme = ""
    rd = _gh_json(f"{base}/readme")
    if rd and rd.get("content"):
        try:
            readme = base64.b64decode(rd["content"]).decode("utf-8", "replace")
        except Exception:
            readme = ""

    tree = _gh_json(f"{base}/git/trees/{default_branch}?recursive=1") or {}
    paths = [t["path"] for t in tree.get("tree", [])
             if t.get("type") == "blob"
             and not any(p in t["path"] for p in SKIP_PARTS)
             and os.path.basename(t["path"]) not in JUNK_FILES]
    paths = paths[:80]

    pkg = ""
    if "package.json" in paths:
        pkg = _gh_text(f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/package.json")[:1500]

    # central source files: entry-ish, non-config code
    code_files = [p for p in paths if p.endswith(CODE_EXTS)
                  and not any(j in p for j in (".config.", "config."))]
    code_files.sort(key=lambda p: (p.count("/"), len(p)))
    excerpts = []
    for p in code_files[:4]:
        txt = _gh_text(f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{p}")
        if txt.strip():
            excerpts.append(f"--- {p} ---\n{txt[:2500]}")

    context = (
        f"<repo>{owner}/{repo} — {meta.get('description') or ''} "
        f"(language: {meta.get('language') or '?'}, stars: {meta.get('stargazers_count', 0)})</repo>\n"
        f"<readme>\n{readme[:5000]}\n</readme>\n"
        f"<file_tree>\n" + "\n".join(paths[:60]) + "\n</file_tree>\n"
        f"<package_json>\n{pkg}\n</package_json>\n"
        f"<code>\n" + "\n\n".join(excerpts)[:6000] + "\n</code>"
    )
    stats = {"readme_chars": len(readme), "files": len(paths), "excerpts": len(excerpts)}
    # Never generate marketing from thin air: if GitHub gave us nothing usable,
    # fail loudly instead of letting the model hallucinate a different project.
    if not readme.strip() and not paths:
        raise RuntimeError(
            "GitHub returned no README and no file list for this repo "
            "(API rate-limited, repo private/renamed, or network hiccup). "
            "Tip: add a free GITHUB_TOKEN (no scopes needed) to the app's Secrets "
            "to raise the GitHub API quota from 60 to 5000 requests/hour.")
    return {"owner": owner, "repo": repo, "meta": meta, "context": context, "stats": stats}

# ================= LLM CALLS =================
def comprehend(ctx):
    """Call 1 — deep understanding. temp 0.1 for fidelity."""
    prompt = f"""You are a senior engineer who has read this exact codebase. Base everything ONLY on the repo context below. If the context doesn't support a claim, write "not clear from context" instead of inventing.

RULE: Every claim must name something real from the repo context: a file path, a command, a config key, a function name, or a README section. A sentence with no concrete reference is a failed sentence — rewrite it.

REJECTED — never write these words/phrases: {", ".join(REJECT_WORDS)}. Never write a sentence that would still be true if you swapped in a different repo's name.

Repo context:
{ctx}

Return STRICT JSON with exactly these fields:
{{
  "explainer": "120-180 words. Explain this repo to a smart friend who doesn't code. No jargon; if you must use a technical term, define it in the same sentence.",
  "short_text": "2-3 sentences COMPRESSED FROM your explainer above — same facts, shorter. This is never empty and never 'not clear from context' when the explainer exists.",
  "how_it_works": ["4-6 steps, in order, one sentence each. Each step must cite the file(s) it comes from, e.g. [src/render.ts]. Steps must form a chain: the output of step N is the input of step N+1. If a step can't be tied to a file in the context, drop it — do not bridge gaps with guesses."],
  "audience": "who this is for, specifically (not 'developers' — which developers, doing what). Derive from the repo context, never leave empty.",
  "proof_points": ["3-5 concrete, verifiable facts from the context: numbers, features, file names, commands. Never empty when the context has code."]
}}
{STRICT_JSON}"""
    def _problems(b):
        if not isinstance(b, dict):
            return ["response was not a JSON object"]
        ps = []
        if not str(b.get("explainer", "")).strip():
            ps.append("explainer")
        if not str(b.get("short_text", "")).strip():
            ps.append("short_text")
        hiw = b.get("how_it_works", [])
        if not isinstance(hiw, list) or not [s for s in hiw if str(s).strip()]:
            ps.append("how_it_works")
        if not str(b.get("audience", "")).strip():
            ps.append("audience")
        pp = b.get("proof_points", [])
        if not isinstance(pp, list) or not [s for s in pp if str(s).strip()]:
            ps.append("proof_points")
        return ps

    b = do_call(prompt, max_tokens=1800, temperature=0.1, label="Understanding the project")
    # Validate — never let empty fields slip through silently (they poison
    # every downstream step: empty short_text once made the social kit invent
    # a whole different project). Re-ask for exactly what's missing.
    for _ in range(2):
        missing = _problems(b)
        banned = [w for w in REJECT_WORDS if w in json.dumps(b).lower()]
        if not missing and not banned:
            break
        bits = []
        if missing:
            bits.append("these fields were empty or missing: " + ", ".join(missing)
                        + " — fill every one from the repo context, no exceptions")
        if banned:
            bits.append("you used banned generic-marketing words (" + ", ".join(banned)
                        + ") — rewrite avoiding them completely")
        b = do_call(prompt + "\nYour last response had problems: " + "; ".join(bits)
                    + ". Return the full corrected JSON.",
                    max_tokens=1800, temperature=0.1, label="Understanding the project")
    missing = _problems(b)
    if missing:
        raise RuntimeError(f"the AI left these fields empty after retries: {', '.join(missing)}")
    return b

def make_diagram(brief):
    """Call 2 — mermaid flowchart from the how-it-works steps. temp 0.1."""
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(brief.get("how_it_works", [])))
    prompt = f"""You are a technical diagrammer. Turn these how-it-works steps into a clean flowchart.

Steps:
{steps}

Rules:
- Output valid Mermaid code starting with "flowchart TD", 4-7 nodes max.
- Node labels: 2-5 plain words each, no jargon, no parentheses, no quotes, no special characters.
- Keep every node traceable to the steps above — no invented stages.
- End with one node showing the outcome for the user.

Return STRICT JSON: {{"mermaid": "flowchart TD\\n    A[First thing] --> B[Second thing]"}}
{STRICT_JSON}"""
    code = ""
    for attempt in range(2):
        p = (prompt if attempt == 0
             else prompt + "\nYour last response was not Mermaid code. Output ONLY the flowchart code "
                          "starting with 'flowchart TD' inside the JSON — no explanations, no JSON dumps.")
        out = do_call(p, max_tokens=800, temperature=0.1, label="Drawing the diagram")
        code = re.sub(r'```(?:mermaid)?', '', str(out.get("mermaid", ""))).strip()
        if code.startswith("flowchart") or code.startswith("graph"):
            break
        code = ""
    return code

EXAMPLE_ITEM = ('{"angle": "THE PROBLEM", "hook": "Dinner panic at 7pm again?", '
                '"visual_hint": "empty fridge glowing in a dark kitchen", '
                '"caption": "You open the fridge. Nothing makes sense together.\\nChefBot looks at what you actually have and builds dinner around it — no shopping trip.", '
                '"hashtags": ["#mealprep", "#home cooking", "#foodtech", "#indiehackers"]}')

def make_social(brief):
    """Call 3 — 4 angle-assigned social items. temp 0.7 for creativity."""
    ctx = (f"Project: {brief.get('short_text','')}\nAudience: {brief.get('audience','')}\n"
           f"Proof points: {'; '.join(brief.get('proof_points', []))}")
    prompt = f"""You are a sharp social-media copywriter who hates generic marketing. Everything must be traceable to this project:

{ctx}

REJECTED — never write these: {", ".join(REJECT_WORDS)}. Never "In today's fast-paced world…", never "Are you tired of…?". Never a sentence that would still be true with a different repo's name.

Write exactly 4 items. Each item owns ONE angle — an item may not reuse another item's angle, vocabulary, or proof point:
1. THE PROBLEM — the pain this project kills. Hook = the frustration, named concretely.
2. THE MAGIC MOMENT — the single most impressive thing it does. Hook = the "wait, it does WHAT?" beat.
3. THE PROOF — a concrete detail: a feature, a number, a file, a workflow step from the context.
4. THE HUMAN — who this is for and why they'd care. Hook = direct address to that person.

DISTINCTNESS CHECK (do this before outputting): list the 4 hooks side by side. If any two share a noun phrase, a verb, or the same sentence shape, rewrite the weaker one. No hashtag may appear in more than one caption. No hook may share more than 2 content words with another hook.

FORMAT EXAMPLE (fictional repo — imitate the punch, not the content):
{EXAMPLE_ITEM}

Per item return: "angle", "hook" (5-9 words, curiosity gap, plain words that render cleanly as big poster text), "visual_hint" (5-10 words: the poster's background motif, monochrome-friendly, tied to the project), "caption" (2-4 short lines, human voice, ends with a soft CTA like "link in bio"), "hashtags" (4-6, no repeats across items).

Return STRICT JSON: {{"items": [4 items]}}
{STRICT_JSON}"""
    out = do_call(prompt, max_tokens=2800, temperature=0.7, label="Writing the social kit")
    items = out.get("items", [])
    return [it for it in items if isinstance(it, dict) and it.get("hook")][:4]

def critique_social(brief, items):
    """Call 4 — one rubric-scored critique pass. temp 0.2. Worth the extra call."""
    if not items:
        return items
    ctx = f"Project: {brief.get('short_text','')}\nProof: {'; '.join(brief.get('proof_points', []))}"
    prompt = f"""You are a ruthless social-media editor. Project context: {ctx}

Here are 4 social items as JSON:
{json.dumps(items, indent=1)}

Score EACH item 1-5 on:
(a) the hook stops the scroll,
(b) zero overlap with the other 3 items (angle, words, proof point),
(c) every claim traceable to the project context,
(d) sounds human, not AI.
For any item scoring below 4 on any axis, rewrite ONLY that item (keep its angle, keep the same JSON shape).
Return the full corrected set as STRICT JSON: {{"items": [4 items]}}
{STRICT_JSON}"""
    out = do_call(prompt, max_tokens=2800, temperature=0.2, label="Reviewing the social kit")
    fixed = [it for it in out.get("items", []) if isinstance(it, dict) and it.get("hook")]
    return fixed[:4] if fixed else items

def make_voiceover(brief):
    """Call 5 — spoken explainer script. temp 0.5. Write for the EAR, not the eye."""
    ctx = (f"{brief.get('explainer','')}\nHow it works: "
           + " ".join(brief.get("how_it_works", [])))
    prompt = f"""Write a voiceover script about this project. A text-to-speech voice reads out everything you write, exactly as you write it. So write speech, not text.

Project context:
{ctx}

Write for the EAR, not the eye:
- Plain sentences only. No markdown, no bullets, no headings, no emoji, no symbols, no parentheses.
- No URLs, no domain names (not even github.com/owner/repo), no version numbers as digits — spell out anything that must be spoken.
- Contractions everywhere ("it's", "you'll"). Short sentences. One idea per sentence.
- Vary sentence length for pacing: a short punchy line after two longer ones.
- Read it back mentally: if a sentence feels stiff spoken aloud, rewrite it.
- Banned AI tics: "Here's the thing", "It's not just X, it's Y", "And that matters because".
- 130-160 words total (about 60-90 seconds spoken).
- Structure: hook (1 line) → what it is (2-3 lines) → how it works, simply (3-4 lines) → who it's for + close (2 lines).

Return STRICT JSON: {{"script": "..."}}
{STRICT_JSON}"""
    out = do_call(prompt, max_tokens=900, temperature=0.5, label="Writing the voiceover")
    script = _clean_spoken(out.get("script", ""))
    # hard gate: no URLs/domains may survive into the script
    if re.search(r"https?://|\bwww\.|\b(?:[a-zA-Z0-9-]+\.)+(?:com|io|dev|app|net|org)\b", script):
        out = do_call(prompt + "\nYour last script contained a URL or domain name. Rewrite with zero URLs, zero domains.",
                      max_tokens=900, temperature=0.5, label="Writing the voiceover")
        script = _clean_spoken(out.get("script", ""))
    return script

def hook_image_prompt(hook, visual_hint):
    return (f"Black and white minimalist Instagram poster, square 1:1. Huge bold sans-serif typography, "
            f"perfectly legible, centered, reading exactly: \"{hook}\". "
            f"Background: {visual_hint}, subtle and abstract, monochrome, lots of negative space. "
            f"High contrast studio poster, clean professional design, no watermark, no logo, no extra text.")

# ================= UI =================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#0A0A0A;--paper:#fff;--wash:#F7F7F7;--line:#E8E8E8;--muted:#737373;}
*{font-family:'Inter',-apple-system,'Segoe UI',sans-serif!important;}
.stApp{background:var(--paper)!important;}
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
.block-container{max-width:880px!important;padding:0 20px 80px!important;}
section[data-testid="stSidebar"]{display:none!important;}

/* nav */
.nav{display:flex;align-items:center;justify-content:space-between;padding:18px 0;border-bottom:1px solid var(--line);margin-bottom:8px;}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:17px;letter-spacing:-.02em;}
.brand-mark{width:26px;height:26px;background:var(--ink);color:#fff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;}
.nav-note{font-size:12px;color:var(--muted);letter-spacing:.04em;}

/* hero */
.hero{text-align:center;padding:64px 8px 12px;}
.kicker{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.22em;color:var(--muted);margin-bottom:18px;text-transform:uppercase;}
.hero h1{font-size:clamp(2rem,6vw,3.4rem);font-weight:800;letter-spacing:-.045em;line-height:1.05;margin:0 0 16px;color:var(--ink);}
.hero p{font-size:16px;color:var(--muted);max-width:520px;margin:0 auto;line-height:1.65;}
div[data-testid="stTextInput"]{max-width:560px;margin:30px auto 0;}
div[data-testid="stTextInput"] input{border-radius:10px!important;padding:15px 18px!important;font-size:15px!important;border:1px solid #D4D4D4!important;background:#fff!important;}
div[data-testid="stTextInput"] input:focus{border-color:var(--ink)!important;box-shadow:0 0 0 3px rgba(0,0,0,.07)!important;}
div[data-testid="stButton"]{max-width:560px;margin:12px auto 0;}
div[data-testid="stButton"] button{background:var(--ink)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:15px!important;font-size:15px!important;font-weight:700!important;width:100%!important;transition:opacity .15s;}
div[data-testid="stButton"] button:hover{opacity:.85!important;}

/* steps */
.steps{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:34px 0 8px;}
.step{display:flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600;color:#A3A3A3;border:1px solid var(--line);border-radius:999px;padding:8px 14px;background:#fff;}
.step .n{width:20px;height:20px;border-radius:50%;background:var(--wash);border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:10.5px;font-weight:700;}
.step.done{color:var(--ink);border-color:#D4D4D4;}
.step.done .n{background:var(--ink);color:#fff;border-color:var(--ink);}
.step.active{color:var(--ink);border-color:var(--ink);}
.step.active .n{background:#fff;border-color:var(--ink);animation:pulse 1.2s infinite;}
@keyframes pulse{50%{transform:scale(1.15);}}

/* sections */
.sec{margin-top:64px;}
.sec-kicker{font-size:11px;font-weight:700;letter-spacing:.22em;color:var(--muted);text-transform:uppercase;margin-bottom:10px;}
.sec-h{font-size:clamp(1.4rem,3.5vw,1.9rem);font-weight:800;letter-spacing:-.03em;margin:0 0 20px;color:var(--ink);}
.tldr{background:var(--ink);color:#fff;border-radius:16px;padding:30px 28px;font-size:clamp(1.05rem,2.6vw,1.3rem);line-height:1.6;font-weight:500;letter-spacing:-.01em;}
.explainer{font-size:16.5px;line-height:1.8;color:#262626;max-width:680px;}
.diagram{border:1px solid var(--line);border-radius:14px;background:#fff;padding:12px;}
.diagram img{width:100%;border-radius:8px;}
.stepper{display:flex;flex-direction:column;gap:0;}
.fstep{display:flex;gap:16px;padding:16px 4px;border-bottom:1px solid var(--line);}
.fstep:last-child{border-bottom:none;}
.fstep .fn{width:30px;height:30px;flex-shrink:0;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;}
.fstep p{margin:2px 0 0;font-size:15px;line-height:1.6;color:#262626;}

/* audio */
.audio-card{border:1px solid var(--line);border-radius:14px;padding:22px;background:var(--wash);}
div[data-testid="stAudio"]{margin-bottom:6px;}
.script{font-size:15px;line-height:1.85;color:#404040;border-left:3px solid var(--ink);padding-left:18px;margin:18px 0 0;font-style:italic;}

/* instagram kit */
.post{border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:28px;background:#fff;}
.post-grid{display:grid;grid-template-columns:300px 1fr;}
.post-img{background:var(--wash);}
.post-img img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;}
.post-body{padding:26px 26px 22px;}
.post-angle{font-size:10.5px;font-weight:700;letter-spacing:.2em;color:var(--muted);text-transform:uppercase;margin-bottom:10px;}
.post-hook{font-size:19px;font-weight:800;letter-spacing:-.02em;line-height:1.3;margin:0 0 14px;color:var(--ink);}
.post-cap{font-size:14.5px;line-height:1.7;color:#404040;white-space:pre-line;margin:0 0 14px;}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px;}
.tag{font-size:12px;font-weight:600;color:var(--muted);background:var(--wash);border:1px solid var(--line);padding:4px 10px;border-radius:6px;}
div[data-testid="stDownloadButton"] button{border-radius:8px!important;border:1px solid #D4D4D4!important;background:#fff!important;color:var(--ink)!important;font-weight:600!important;font-size:13px!important;padding:8px 16px!important;}
div[data-testid="stDownloadButton"] button:hover{background:var(--ink)!important;color:#fff!important;}

/* report + footer */
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;}
.report-row{display:flex;justify-content:space-between;font-size:13.5px;padding:8px 0;border-bottom:1px solid var(--line);color:#404040;}
.report-row:last-child{border:none;}
.ok{color:#15803d;font-weight:700;} .bad{color:#b91c1c;font-weight:700;} .info{color:var(--muted);font-weight:600;}
.footer{margin-top:72px;padding-top:24px;border-top:1px solid var(--line);text-align:center;font-size:12.5px;color:var(--muted);}
.stAlert{border-radius:10px!important;}

@media(max-width:700px){
  .post-grid{grid-template-columns:1fr;}
  .hero{padding:44px 4px 8px;}
  .tldr{padding:24px 20px;}
  .post-body{padding:20px;}
  .nav-note{display:none;}
}
</style>
"""

def render_steps(done, active=None):
    labels = ["Reading repo", "Understanding project", "Drawing the diagram",
              "Writing the social kit", "Creating visuals", "Recording audio"]
    keys = ["fetch", "comprehend", "diagram", "social", "visuals", "audio"]
    out = ['<div class="steps">']
    for k, lab in zip(keys, labels):
        cls = "done" if k in done else ("active" if k == active else "")
        mark = "✓" if k in done else str(keys.index(k) + 1)
        out.append(f'<div class="step {cls}"><span class="n">{mark}</span>{lab}</div>')
    out.append('</div>')
    return "".join(out)

# ================= PAGE =================
st.set_page_config(page_title="HypeRepo — paste a repo, get its story", layout="centered")
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="brand"><div class="brand-mark">H</div>HypeRepo</div>'
            '<div class="nav-note">REPO → STORY</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="kicker">AI repo explainer</div>'
            '<h1>Paste a repo.<br>Get its story.</h1>'
            '<p>A clean diagram of how it works, four scroll-stopping visuals with captions, '
            'and a plain-English audio explanation — all generated from the actual code.</p></div>',
            unsafe_allow_html=True)

if not DO_API_KEY or not ALIBABA_API_KEY:
    st.error("Missing API keys — add `DO_API_KEY` and `ALIBABA_API_KEY` in the app's Secrets settings.")
    st.stop()

repo_url = st.text_input("repo", placeholder="https://github.com/owner/repo", label_visibility="collapsed")
go = st.button("Generate the story", use_container_width=True)

if go:
    if not repo_url.strip() or "github.com" not in repo_url:
        st.error("Please paste a valid GitHub repo URL.")
        st.stop()

    steps_ph = st.empty()
    order = ["fetch", "comprehend", "diagram", "social", "visuals", "audio"]
    done = []
    failed = None
    results = {}

    def tick(key):
        done.append(key)
        nxt = next((k for k in order if k not in done), None)
        steps_ph.markdown(render_steps(done, active=nxt), unsafe_allow_html=True)

    # ---- fatal section: everything downstream needs the repo + the brief ----
    current = "Reading the repo"
    try:
        steps_ph.markdown(render_steps(done, active="fetch"), unsafe_allow_html=True)
        repo = fetch_repo(repo_url.strip())
        tick("fetch")

        current = "Understanding the project"
        brief = comprehend(repo["context"])
        results["brief"] = brief
        tick("comprehend")
    except Exception as e:
        failed = (current, e)

    # ---- soft sections: a failure degrades one section, never the run ----
    warnings = []
    if failed is None:
        try:
            mermaid = make_diagram(brief)
            d_bytes, d_err = fetch_diagram(mermaid)
            if d_err:
                warnings.append(f"Diagram render: {d_err}")
        except Exception as e:
            mermaid, d_bytes, d_err = "", None, str(e)[:150]
            warnings.append(f"Diagram: {str(e)[:150]}")
        results.update(mermaid=mermaid, diagram=d_bytes, diagram_err=d_err)
        tick("diagram")

        try:
            items = make_social(brief)
            items = critique_social(brief, items)
        except Exception as e:
            items = []
            warnings.append(f"Social kit: {str(e)[:150]}")
        results["items"] = items
        tick("social")

        images = []
        for idx, it in enumerate(items):
            if idx:
                time.sleep(4)  # breathing room: Alibaba throttles rapid-fire requests
            png, err = qwen_image(hook_image_prompt(it.get("hook", ""), it.get("visual_hint", "abstract minimal shapes")))
            images.append({"png": png, "err": err})
        results["images"] = images
        results["stats"] = repo.get("stats", {})
        tick("visuals")

        try:
            script = make_voiceover(brief)
        except Exception as e:
            script = ""
            warnings.append(f"Voiceover script: {str(e)[:150]}")
        if script:
            audio, a_err = make_audio(script)
            if a_err:
                warnings.append(f"Audio: {a_err}")
        else:
            audio, a_err = None, "script unavailable"
        results.update(script=script, audio=audio, audio_err=a_err)
        tick("audio")

    steps_ph.markdown(render_steps(done), unsafe_allow_html=True)
    if failed is not None:
        step_label, e = failed
        st.error(f"The run stopped while {step_label.lower()}.")
        st.write(f"**What happened:** {html.escape(str(e)[:400])}")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")
        st.stop()

    brief, items = results["brief"], results["items"]
    repo_name = re.sub(r"\W+", "_", repo["repo"])[:30]

    # 1 — short version
    st.markdown('<div class="sec"><div class="sec-kicker">The short version</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="tldr">{html.escape(brief.get("short_text", ""))}</div></div>', unsafe_allow_html=True)

    # 2 — diagram
    st.markdown('<div class="sec"><div class="sec-kicker">How it works</div>'
                '<div class="sec-h">The whole app, one diagram.</div>', unsafe_allow_html=True)
    if results["diagram"]:
        st.markdown('<div class="diagram">', unsafe_allow_html=True)
        st.image(results["diagram"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("Download diagram (PNG)", results["diagram"],
                           file_name=f"{repo_name}_how_it_works.png", mime="image/png")
    else:
        st.warning(f"Couldn't render the diagram ({html.escape(results['diagram_err'][:120])}) — here are the steps instead:")
        steps_html = "".join(
            f'<div class="fstep"><div class="fn">{i+1}</div><p>{html.escape(re.sub(r"\[[^\]]+\]", "", s).strip())}</p></div>'
            for i, s in enumerate(brief.get("how_it_works", [])))
        st.markdown(f'<div class="stepper">{steps_html}</div>', unsafe_allow_html=True)
    with st.expander("Diagram source (Mermaid)"):
        st.code(results["mermaid"], language="mermaid")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3 — explainer
    st.markdown('<div class="sec"><div class="sec-kicker">The explanation</div>'
                '<div class="sec-h">What this repo actually is.</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="explainer">{html.escape(brief.get("explainer", ""))}</p></div>', unsafe_allow_html=True)

    # 4 — audio
    st.markdown('<div class="sec"><div class="sec-kicker">Listen</div>'
                '<div class="sec-h">The 60-second version, out loud.</div>', unsafe_allow_html=True)
    st.markdown('<div class="audio-card">', unsafe_allow_html=True)
    if results["audio"]:
        st.audio(results["audio"], format="audio/mp3")
        st.download_button("Download audio (MP3)", results["audio"],
                           file_name=f"{repo_name}_explainer.mp3", mime="audio/mp3")
    else:
        st.warning(f"Audio failed: {html.escape(results['audio_err'][:150])}")
    if results["script"]:
        st.markdown(f'<p class="script">"{html.escape(results["script"])}"</p>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 5 — instagram kit
    st.markdown('<div class="sec"><div class="sec-kicker">Instagram kit</div>'
                '<div class="sec-h">Four hooks. Four captions. Zero repetition.</div>', unsafe_allow_html=True)
    if not items:
        st.warning("The social kit couldn't be generated this run — everything else on this page is fine.")
    for i, it in enumerate(items):
        img = results["images"][i] if i < len(results["images"]) else {"png": None, "err": "missing"}
        tags = "".join(f'<span class="tag">#{html.escape(str(t).lstrip("#"))}</span>' for t in it.get("hashtags", []))
        if img["png"]:
            st.markdown('<div class="diagram" style="margin-bottom:14px;">', unsafe_allow_html=True)
            st.image(img["png"])
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="post"><div class="post-body">'
            f'<div class="post-angle">{html.escape(str(it.get("angle", "")))}</div>'
            f'<p class="post-hook">{html.escape(str(it.get("hook", "")))}</p>'
            f'<p class="post-cap">{html.escape(str(it.get("caption", "")))}</p>'
            f'<div class="tags">{tags}</div>'
            f'</div></div>', unsafe_allow_html=True)
        if img["png"]:
            st.download_button(f"Download visual {i+1} (PNG)", img["png"],
                               file_name=f"{repo_name}_hook_{i+1}.png", mime="image/png",
                               key=f"dl_img_{i}")
        else:
            st.warning(f"Visual {i+1} failed: {html.escape(str(img['err'])[:150])}")
    st.markdown('</div>', unsafe_allow_html=True)

    # report
    with st.expander("Generation report"):
        rows = []
        stt = results.get("stats", {})
        rows.append(("Repo context",
                     f"{stt.get('readme_chars', 0)} readme chars, "
                     f"{stt.get('files', 0)} files, {stt.get('excerpts', 0)} code excerpts"))
        rows.append(("Diagram", "OK" if results["diagram"] else f"FAILED — {results['diagram_err'][:100]}"))
        for i, im in enumerate(results["images"]):
            rows.append((f"Visual {i+1}", "OK" if im["png"] else f"FAILED — {str(im['err'])[:100]}"))
        rows.append(("Audio", "OK" if results["audio"] else f"FAILED — {results['audio_err'][:100]}"))
        for label, status in rows:
            cls = "ok" if status == "OK" else ("info" if label == "Repo context" else "bad")
            st.markdown(f'<div class="report-row"><span>{html.escape(label)}</span>'
                        f'<span class="{cls}">{html.escape(status)}</span></div>', unsafe_allow_html=True)
        for w in warnings:
            st.markdown(f'<div class="report-row"><span>Notice</span>'
                        f'<span class="bad">{html.escape(w)}</span></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">HypeRepo — paste a repo, get its story. Built from real code, not templates.</div>',
            unsafe_allow_html=True)
