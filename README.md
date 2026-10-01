# AI Marketing Agent

A Streamlit app that reads a public GitHub repository, uses a DigitalOcean model to create marketing copy, generates voiceovers with Edge TTS, and creates social images through Alibaba Qwen Image.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Replace the placeholder values in `.streamlit/secrets.toml`, then start the app:

```bash
streamlit run app.py
```
