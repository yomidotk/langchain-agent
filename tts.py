"""Read answers aloud with Alibaba Qwen-TTS (Singapore / international endpoint)."""

import hashlib
import io
import re
import wave
from pathlib import Path

import httpx

URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
CACHE = Path(__file__).parent / "audio_cache"  # generated audio is saved here, so replays are free and instant
MAX_CHARS = 3000  # reading very long answers aloud gets slow and costs more
CHUNK = 450  # the API accepts at most ~600 characters per request

VOICES = ["Cherry", "Serena", "Ethan", "Chelsie", "Momo", "Vivian"]

# A style uses the "instruct" model, which can be told *how* to speak (Chinese/English instructions only).
STYLES = {
    "Default voice": None,
    "Friendly teacher": "Warm, upbeat and encouraging, like a friendly teacher explaining clearly at a natural pace.",
    "Calm and clear": "Calm, soft and clear, at a slightly slower pace, easy to follow.",
    "Energetic coach": "Energetic, enthusiastic and motivating, with a lively pace and rising intonation.",
}


def clean_for_speech(text: str) -> str:
    """Remove markdown, links, emojis and code so the voice doesn't read symbols out loud."""
    text = re.sub(r"```.*?```", " (code example skipped) ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)  # thread_id -> thread id
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~|>#]+", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]", "", text)
    text = re.sub(r"\s*\n+\s*", ". ", text)  # line breaks become pauses
    text = re.sub(r"([.!?:;,])\.\s", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunks(text: str, limit: int = CHUNK) -> list[str]:
    """Split on sentence ends, packing sentences into pieces of at most `limit` characters."""
    out, current = [], ""
    for sentence in re.split(r"(?<=[.!?。])\s+", text):
        while len(sentence) > limit:  # one giant sentence: cut at the last space
            cut = sentence.rfind(" ", 0, limit) or limit
            piece, sentence = sentence[:cut], sentence[cut:].strip()
            if current:
                out.append(current)
                current = ""
            out.append(piece)
        if len(current) + len(sentence) + 1 > limit and current:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def _request_audio_url(chunk: str, api_key: str, voice: str, instruction: str | None) -> str:
    body = {
        "model": "qwen3-tts-instruct-flash" if instruction else "qwen3-tts-flash",
        "input": {"text": chunk, "voice": voice, "language_type": "Auto"},
    }
    if instruction:
        body["input"]["instructions"] = instruction
        body["input"]["optimize_instructions"] = True
    r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200:
        raise RuntimeError(f"Alibaba TTS error {r.status_code}: {data.get('message') or r.text[:200]}")
    return data["output"]["audio"]["url"]


def _download(url: str) -> bytes:
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


def _join_wavs(parts: list[bytes]) -> bytes:
    if len(parts) == 1:
        return parts[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        for i, part in enumerate(parts):
            with wave.open(io.BytesIO(part), "rb") as reader:
                if i == 0:
                    writer.setparams(reader.getparams())
                writer.writeframes(reader.readframes(reader.getnframes()))
    return out.getvalue()


def synthesize(text: str, api_key: str, voice: str = "Cherry", style: str | None = None) -> bytes:
    """Return WAV bytes for `text`. Results are cached on disk."""
    speech = clean_for_speech(text)[:MAX_CHARS]
    if not speech:
        raise ValueError("There is nothing to read aloud.")
    instruction = STYLES.get(style)
    key = hashlib.md5(f"{voice}|{instruction}|{speech}".encode()).hexdigest()
    path = CACHE / f"{key}.wav"
    if path.exists():
        return path.read_bytes()
    parts = [_download(_request_audio_url(c, api_key, voice, instruction)) for c in _chunks(speech)]
    wav = _join_wavs(parts)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(wav)
    return wav
