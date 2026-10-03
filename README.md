# Studeno

A personal study companion built with Streamlit. Studeno supports contextual AI chat, image questions, saved notes, optional voice playback, and self-paced flashcard practice.

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
