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

# ================= DESIGN SYSTEM =================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@400;600;700;800&family=Playfair+Display:ital,wght@1,500;1,600;1,700&display=swap');

  /* ─── TOKENS ─── */
  :root {
    --cream:   #FAF7F2;
    --cream2:  #F4EFE6;
    --parchm:  #EDE5D8;
    --sand:    #D9CEBA;
    --caramel: #C4A882;
    --amber:   #A07850;
    --walnut:  #6B4F35;
    --espresso:#3D2B1A;
    --ink:     #1C1208;
    --accent:  #7C5C38;
    --purple:  #7C6FCD;
    --purple2: #A390E4;
    --green:   #4A7C59;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
  html { scroll-behavior: smooth; }
  ::selection { background: rgba(160,120,80,0.25); color: var(--walnut); }

  /* ─── BASE ─── */
  .stApp { background: var(--cream) !important; min-height: 100vh; }
  #MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
  .block-container { max-width: 1200px !important; padding: 0 clamp(16px, 4vw, 48px) 80px !important; margin: 0 auto !important; }
  section[data-testid="stSidebar"] { display: none !important; }

  /* ─── BACKGROUND ─── */
  .bg-canvas { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
  .bg-canvas::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 70% 55% at 8% 0%,   rgba(196,168,130,0.35) 0%, transparent 60%),
      radial-gradient(ellipse 55% 45% at 92% 8%,  rgba(160,120,80,0.22)  0%, transparent 55%),
      radial-gradient(ellipse 60% 50% at 50% 105%, rgba(124,95,56,0.18)   0%, transparent 60%),
      var(--cream);
  }
  .grid-overlay {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: linear-gradient(rgba(160,120,80,0.06) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(160,120,80,0.06) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
  }
  .orb { position: fixed; border-radius: 50%; filter: blur(130px); pointer-events: none; z-index: 0; }
  .orb-1 { width: 640px; height: 640px; background: radial-gradient(circle, rgba(196,168,130,0.55), transparent 70%); top: -220px; left: -160px; animation: orbFloat 22s ease-in-out infinite alternate; }
  .orb-2 { width: 520px; height: 520px; background: radial-gradient(circle, rgba(160,120,80,0.38),  transparent 70%); top: 28%; right: -190px; animation: orbFloat 28s ease-in-out infinite alternate-reverse; }
  .orb-3 { width: 440px; height: 440px; background: radial-gradient(circle, rgba(217,206,186,0.5),  transparent 70%); bottom: -110px; left: 28%; animation: orbFloat 19s ease-in-out infinite alternate; }
  @keyframes orbFloat {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(40px,60px) scale(1.1); }
  }

  /* ─── NOISE OVERLAY ─── */
  .noise { position: fixed; inset: 0; z-index: 1; pointer-events: none; opacity: 0.06;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }

  /* ─── NAV ─── */
  .nav {
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    background: rgba(250,247,242,0.82);
    border-bottom: 1px solid rgba(160,120,80,0.14);
    margin: 0 clamp(-16px, -4vw, -48px);
    padding: 0 clamp(16px, 4vw, 48px);
    box-shadow: 0 1px 0 rgba(160,120,80,0.08);
  }
  .nav-inner { max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; height: 68px; }
  .logo {
    font-family: 'Sora', sans-serif !important; font-weight: 800; font-size: 20px;
    letter-spacing: -.04em; color: var(--espresso); display: flex; align-items: center; gap: 10px;
  }
  .logo-icon {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--amber), var(--walnut));
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; box-shadow: 0 4px 16px rgba(107,79,53,0.35);
  }
  .logo-text span { background: linear-gradient(120deg, var(--amber), var(--walnut)); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .nav-links { display: flex; align-items: center; gap: 32px; }
  .nav-links a {
    color: var(--walnut); text-decoration: none; font-size: 14px; font-weight: 500;
    transition: color .2s; letter-spacing: -.01em; opacity: 0.7;
  }
  .nav-links a:hover { color: var(--espresso); opacity: 1; }
  .nav-badge {
    background: rgba(160,120,80,0.1); border: 1px solid rgba(160,120,80,0.25);
    color: var(--amber); font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 6px;
    letter-spacing: .02em;
  }
  .nav-cta {
    background: linear-gradient(135deg, var(--amber), var(--walnut)) !important;
    color: #fff !important; text-decoration: none; font-size: 13.5px; font-weight: 700;
    padding: 10px 24px; border-radius: 10px;
    box-shadow: 0 4px 20px rgba(107,79,53,0.3);
    transition: transform .15s, box-shadow .15s;
    letter-spacing: -.01em;
  }
  .nav-cta:hover { transform: translateY(-1px); box-shadow: 0 8px 28px rgba(107,79,53,0.45); }

  /* ─── HERO ─── */
  .hero { text-align: center; padding: clamp(80px,12vw,140px) 16px clamp(20px,4vw,40px); position: relative; z-index: 2; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 700;
    letter-spacing: .18em; color: var(--amber);
    background: rgba(160,120,80,0.1); border: 1px solid rgba(160,120,80,0.25);
    padding: 8px 18px; border-radius: 999px; margin-bottom: 32px;
    backdrop-filter: blur(8px); text-transform: uppercase;
  }
  .pulse-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--amber);
    box-shadow: 0 0 8px rgba(160,120,80,0.7); animation: pulse 1.6s ease-in-out infinite;
  }
  .hero h1 {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(2.8rem,7vw,5.5rem); font-weight: 800; letter-spacing: -.05em;
    line-height: 1.02; color: var(--ink); margin: 0 0 24px;
  }
  .serif-accent {
    font-family: 'Playfair Display', Georgia, serif !important; font-style: italic; font-weight: 600;
    letter-spacing: -.02em;
    background: linear-gradient(115deg, var(--amber) 0%, var(--walnut) 45%, var(--caramel) 80%, #C87941 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shimmer 6s linear infinite;
  }
  @keyframes shimmer { to { background-position: 200% center; } }
  .hero p.sub {
    font-size: clamp(1rem,2.5vw,1.2rem); color: var(--walnut);
    max-width: 580px; margin: 0 auto 8px; line-height: 1.75; font-weight: 400; opacity: 0.75;
  }

  /* ─── INPUT ─── */
  div[data-testid="stTextInput"] { max-width: 680px; margin: 36px auto 0; position: relative; z-index: 2; }
  div[data-testid="stTextInput"] label { display: none !important; }
  div[data-testid="stTextInput"] input {
    border-radius: 999px !important; padding: 20px 30px !important;
    font-size: 15px !important; font-weight: 500 !important;
    border: 1.5px solid rgba(160,120,80,0.22) !important;
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(16px) !important;
    color: var(--espresso) !important;
    box-shadow: 0 2px 16px rgba(160,120,80,0.1), inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transition: border-color .25s, box-shadow .25s, background .25s !important;
    caret-color: var(--amber) !important;
  }
  div[data-testid="stTextInput"] input::placeholder { color: rgba(107,79,53,0.35) !important; }
  div[data-testid="stTextInput"] input:focus {
    border-color: var(--caramel) !important;
    background: #fff !important;
    box-shadow: 0 0 0 5px rgba(160,120,80,0.12), 0 8px 32px rgba(160,120,80,0.18) !important;
  }

  /* ─── BUTTON ─── */
  div[data-testid="stButton"] { margin-top: 0; position: relative; z-index: 2; }
  div[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--amber) 0%, var(--walnut) 60%, var(--espresso) 100%) !important;
    color: #fff !important; border: none !important; border-radius: 999px !important;
    padding: 18px 72px !important; font-size: 16px !important; font-weight: 700 !important;
    letter-spacing: -.01em !important;
    box-shadow: 0 8px 36px rgba(107,79,53,0.45), 0 2px 8px rgba(107,79,53,0.3) !important;
    transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s ease !important;
    position: relative; overflow: hidden;
  }
  div[data-testid="stButton"] button::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.18) 0%, transparent 55%);
    border-radius: 14px;
  }
  div[data-testid="stButton"] button::after {
    content: ''; position: absolute; top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg); transition: left .8s ease;
  }
  div[data-testid="stButton"] button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 16px 52px rgba(107,79,53,0.55), 0 4px 14px rgba(107,79,53,0.3) !important;
  }
  div[data-testid="stButton"] button:hover::after { left: 150% !important; }

  /* ─── STATS BAR ─── */
  .stats {
    display: flex; justify-content: center; align-items: center; gap: 0;
    margin: 60px auto 0; max-width: 820px; position: relative; z-index: 2;
    background: rgba(255,255,255,0.6); border: 1px solid rgba(160,120,80,0.16);
    border-radius: 20px; backdrop-filter: blur(16px); flex-wrap: wrap; overflow: hidden;
    box-shadow: 0 4px 32px rgba(160,120,80,0.12);
  }
  .stat { text-align: center; padding: 28px 48px; flex: 1; min-width: 120px; position: relative; }
  .stat + .stat { border-left: 1px solid rgba(160,120,80,0.12); }
  .stat b {
    display: block; font-size: 36px; font-weight: 800; letter-spacing: -.04em;
    background: linear-gradient(135deg, var(--espresso) 0%, var(--walnut) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 4px;
  }
  .stat span { font-size: 11.5px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--caramel); }

  /* ─── TICKER ─── */
  .ticker {
    margin: 70px clamp(-16px,-4vw,-48px) 0;
    border-top: 1px solid rgba(160,120,80,0.12); border-bottom: 1px solid rgba(160,120,80,0.12);
    background: rgba(237,229,216,0.5); backdrop-filter: blur(10px);
    overflow: hidden; position: relative; z-index: 2;
  }
  .ticker-track { display: flex; gap: 0; width: max-content; animation: tick 30s linear infinite; padding: 18px 0; }
  .ticker:hover .ticker-track { animation-play-state: paused; }
  .tick { font-size: 12px; font-weight: 700; letter-spacing: .25em; color: rgba(107,79,53,0.35); padding: 0 32px; white-space: nowrap; text-transform: uppercase; }
  .tick em { font-style: normal; color: var(--amber); padding-right: 32px; }
  @keyframes tick { to { transform: translateX(-50%); } }

  /* ─── SECTION SCAFFOLDING ─── */
  .section { max-width: 1100px; margin: 0 auto; padding: clamp(80px,10vw,120px) 0 0; position: relative; z-index: 2; }
  .kicker {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: var(--amber); margin-bottom: 18px;
  }
  .kicker::before { content: ''; width: 20px; height: 1px; background: var(--caramel); }
  .sec-h {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(1.9rem,4.5vw,3rem); font-weight: 800; letter-spacing: -.04em;
    color: var(--ink); margin: 0 0 16px; line-height: 1.1;
  }
  .sec-p { color: var(--walnut); font-size: 17px; line-height: 1.7; max-width: 580px; margin: 0 0 48px; font-weight: 400; opacity: 0.8; }

  /* ─── HOW IT WORKS ─── */
  .how-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }
  .how-card {
    background: rgba(255,255,255,0.72); border: 1px solid rgba(160,120,80,0.14);
    border-radius: 24px; padding: 36px 30px; position: relative; overflow: hidden;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1), border-color .3s, box-shadow .3s;
    backdrop-filter: blur(12px); box-shadow: 0 2px 20px rgba(160,120,80,0.08);
  }
  .how-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(160,120,80,0.4), transparent);
  }
  .how-card:hover {
    transform: translateY(-6px);
    border-color: rgba(160,120,80,0.3);
    box-shadow: 0 20px 56px -16px rgba(107,79,53,0.22), 0 0 0 1px rgba(160,120,80,0.18);
  }
  .how-num {
    font-size: 72px; font-weight: 900; letter-spacing: -.06em; line-height: 1;
    background: linear-gradient(180deg, rgba(196,168,130,0.7) 0%, rgba(196,168,130,0.1) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 20px;
  }
  .how-card h3 { font-size: 17px; font-weight: 700; margin: 0 0 10px; color: var(--espresso); letter-spacing: -.01em; }
  .how-card p { font-size: 14.5px; color: var(--walnut); line-height: 1.7; margin: 0; opacity: 0.8; }

  /* ─── BUNDLE GRID ─── */
  .bundle-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }
  .bundle-card {
    background: rgba(255,255,255,0.6); border: 1px solid rgba(160,120,80,0.12);
    border-radius: 22px; padding: 30px 28px; backdrop-filter: blur(12px);
    transition: transform .3s cubic-bezier(.34,1.56,.64,1), border-color .3s, box-shadow .3s;
    position: relative; overflow: hidden; box-shadow: 0 2px 16px rgba(160,120,80,0.07);
  }
  .bundle-card::after {
    content: ''; position: absolute; inset: 0; border-radius: 22px;
    background: radial-gradient(circle at 0% 0%, rgba(196,168,130,0.12), transparent 60%);
    opacity: 0; transition: opacity .3s;
  }
  .bundle-card:hover {
    transform: translateY(-5px);
    border-color: rgba(160,120,80,0.25);
    box-shadow: 0 18px 48px -16px rgba(107,79,53,0.2);
  }
  .bundle-card:hover::after { opacity: 1; }
  .bundle-icon {
    width: 52px; height: 52px; border-radius: 16px; display: flex; align-items: center; justify-content: center;
    font-size: 24px; background: rgba(160,120,80,0.1); border: 1px solid rgba(160,120,80,0.2);
    margin-bottom: 20px; position: relative; z-index: 1;
  }
  .bundle-card h3 { font-size: 16px; font-weight: 700; margin: 0 0 8px; color: var(--espresso); letter-spacing: -.01em; position: relative; z-index: 1; }
  .bundle-card p { font-size: 14px; color: var(--walnut); line-height: 1.65; margin: 0; position: relative; z-index: 1; opacity: 0.8; }

  /* ─── PIPELINE STEPS ─── */
  .steps {
    display: flex; gap: 8px; justify-content: center; margin: 40px auto 16px;
    max-width: 1000px; position: relative; z-index: 2; flex-wrap: wrap;
  }
  .step {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.65); border: 1px solid rgba(160,120,80,0.15);
    border-radius: 12px; padding: 12px 18px 12px 12px;
    font-size: 13px; font-weight: 600; color: rgba(107,79,53,0.45);
    backdrop-filter: blur(8px); transition: all .3s;
  }
  .step .dot {
    width: 32px; height: 32px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(160,120,80,0.08); font-size: 14px; flex-shrink: 0;
  }
  .step.done { color: var(--green); border-color: rgba(74,124,89,0.2); background: rgba(74,124,89,0.06); }
  .step.done .dot { background: rgba(74,124,89,0.12); color: var(--green); }
  .step.active {
    color: var(--espresso); border-color: rgba(160,120,80,0.4);
    background: rgba(255,255,255,0.9);
    box-shadow: 0 4px 20px rgba(160,120,80,0.2);
  }
  .step.active .dot {
    background: linear-gradient(135deg, var(--amber), var(--walnut)); color: #fff;
    animation: pulse 1.2s ease-in-out infinite; box-shadow: 0 0 14px rgba(160,120,80,0.5);
  }
  @keyframes pulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.12); opacity: 0.85; } }

  /* ─── RESULTS ─── */
  .sec-title {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(1.6rem,3.5vw,2.2rem); font-weight: 800; letter-spacing: -.035em;
    color: var(--ink); margin: 72px 0 8px; position: relative; z-index: 2;
  }
  .sec-sub { color: var(--walnut); margin-bottom: 28px; position: relative; z-index: 2; font-size: 15px; opacity: 0.75; }
  .demo-card {
    background: var(--espresso); border-radius: 28px;
    padding: clamp(28px,4vw,52px); color: #fff;
    position: relative; overflow: hidden; z-index: 2;
    box-shadow: 0 32px 72px -24px rgba(61,43,26,0.5);
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.07);
  }
  .demo-card::before {
    content: ''; position: absolute; width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(196,168,130,0.2), transparent 70%);
    top: -200px; right: -150px; animation: orbFloat 14s ease-in-out infinite alternate;
  }
  .demo-card::after {
    content: ''; position: absolute; width: 400px; height: 400px; border-radius: 50%;
    background: radial-gradient(circle, rgba(160,120,80,0.15), transparent 70%);
    bottom: -160px; left: -120px;
  }
  .demo-kicker {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 700; letter-spacing: .18em;
    color: var(--caramel); margin-bottom: 16px; position: relative; z-index: 1;
    background: rgba(196,168,130,0.12); border: 1px solid rgba(196,168,130,0.25);
    padding: 6px 14px; border-radius: 999px;
  }
  .demo-card h2 {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(1.5rem,3.5vw,2.2rem); font-weight: 800; margin: 0 0 24px;
    position: relative; z-index: 1; letter-spacing: -.03em; color: var(--cream);
  }
  .demo-card p.script {
    color: rgba(250,247,242,0.72); line-height: 1.85; font-size: 16px;
    position: relative; z-index: 1; margin-bottom: 16px; font-weight: 400;
  }

  /* ─── PITCH GRID ─── */
  .pitch-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; position: relative; z-index: 2; }
  .pitch-card {
    background: rgba(255,255,255,0.75); border: 1px solid rgba(160,120,80,0.14);
    border-radius: 22px; padding: 32px 28px;
    position: relative; overflow: hidden;
    animation: fadeUp .6s cubic-bezier(.22,1,.36,1) both;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1), box-shadow .3s, border-color .3s;
    backdrop-filter: blur(10px); box-shadow: 0 2px 18px rgba(160,120,80,0.08);
  }
  .pitch-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 56px -16px rgba(107,79,53,0.22);
    border-color: rgba(160,120,80,0.28);
  }
  .pitch-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--caramel), var(--amber), var(--walnut));
  }
  .pitch-card .icon { font-size: 32px; margin-bottom: 16px; }
  .pitch-card h3 { font-size: 16px; font-weight: 700; color: var(--espresso); margin: 0 0 10px; letter-spacing: -.01em; }
  .pitch-card p { font-size: 14px; color: var(--walnut); line-height: 1.65; margin: 0; opacity: 0.8; }

  /* ─── SOCIAL POSTS ─── */
  .post-wrap {
    display: grid; grid-template-columns: 300px 1fr; gap: 40px; align-items: start;
    background: rgba(255,255,255,0.7); border: 1px solid rgba(160,120,80,0.14);
    border-radius: 28px;
    padding: clamp(24px,4vw,44px); margin-bottom: 28px; position: relative; z-index: 2;
    box-shadow: 0 8px 40px rgba(107,79,53,0.12);
    animation: fadeUp .6s cubic-bezier(.22,1,.36,1) both;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1), box-shadow .3s, border-color .3s;
    backdrop-filter: blur(16px);
  }
  .post-wrap:hover {
    transform: translateY(-4px);
    box-shadow: 0 24px 60px -16px rgba(107,79,53,0.22);
    border-color: rgba(160,120,80,0.25);
  }
  .post-num {
    position: absolute; top: -14px; left: 28px;
    background: linear-gradient(135deg, var(--amber), var(--walnut));
    color: #fff; font-size: 11px; font-weight: 800; letter-spacing: .12em;
    padding: 6px 16px; border-radius: 999px;
    box-shadow: 0 4px 18px rgba(107,79,53,0.4);
  }
  .phone {
    width: 260px; margin: 16px auto 0; background: var(--espresso); border-radius: 48px; padding: 10px;
    box-shadow: 0 36px 64px -18px rgba(61,43,26,0.55), 0 0 0 1px rgba(255,255,255,0.1);
    transition: transform .35s cubic-bezier(.34,1.56,.64,1);
  }
  .phone:hover { transform: rotate(-1.5deg) scale(1.02); }
  .phone-screen { background: #fff; border-radius: 40px; overflow: hidden; position: relative; }
  .notch {
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    width: 90px; height: 22px; background: var(--espresso); border-radius: 999px; z-index: 2;
  }
  .ig-head { display: flex; align-items: center; gap: 10px; padding: 40px 14px 10px; }
  .ig-avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, var(--amber), var(--walnut));
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 14px; font-weight: 800;
  }
  .ig-user { font-size: 12.5px; font-weight: 700; color: #18181B; }
  .ig-img { width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; background: #f4f4f5; }
  .ig-actions { display: flex; gap: 14px; padding: 10px 14px 4px; font-size: 19px; }
  .ig-cap { padding: 4px 14px 16px; font-size: 12px; color: #3f3f46; line-height: 1.55; }
  .ig-cap b { color: #18181B; }

  /* ─── POST CONTENT AREA ─── */
  .hook {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(1.3rem,3vw,1.8rem); font-weight: 800;
    color: var(--ink); letter-spacing: -.03em; margin: 0 0 20px; line-height: 1.2;
  }
  .vo-label { font-size: 10px; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: var(--amber); margin: 24px 0 8px;
    display: flex; align-items: center; gap: 8px; }
  .vo-label::after { content: ''; flex: 1; height: 1px; background: rgba(160,120,80,0.2); }
  .cap-label { font-size: 10px; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: var(--amber); margin: 24px 0 8px;
    display: flex; align-items: center; gap: 8px; }
  .cap-label::after { content: ''; flex: 1; height: 1px; background: rgba(160,120,80,0.2); }
  .vo-script {
    font-size: 15px; color: var(--walnut); font-style: italic; line-height: 1.75;
    border-left: 2px solid rgba(160,120,80,0.35); padding-left: 16px; margin: 0 0 8px; opacity: 0.85;
  }
  .cap-text { font-size: 14px; color: var(--walnut); line-height: 1.7; white-space: pre-line; opacity: 0.85; }
  .tags { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 6px; }
  .tag {
    background: rgba(160,120,80,0.09); color: var(--amber);
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 8px;
    border: 1px solid rgba(160,120,80,0.2);
    transition: all .2s;
  }
  .tag:hover { background: rgba(160,120,80,0.16); border-color: rgba(160,120,80,0.4); transform: translateY(-1px); }

  /* ─── DOWNLOAD BUTTONS ─── */
  div[data-testid="stDownloadButton"] button {
    border-radius: 10px !important; font-weight: 600 !important;
    border: 1px solid rgba(160,120,80,0.22) !important;
    color: var(--walnut) !important;
    background: rgba(255,255,255,0.7) !important;
    padding: 10px 20px !important; font-size: 13px !important; width: 100%;
    transition: all .2s; letter-spacing: -.01em !important;
  }
  div[data-testid="stDownloadButton"] button:hover {
    background: rgba(160,120,80,0.08) !important;
    border-color: var(--caramel) !important;
    color: var(--espresso) !important;
    transform: translateY(-1px);
  }

  /* ─── EXPANDER ─── */
  div[data-testid="stExpander"] {
    border: 1px solid rgba(160,120,80,0.16) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(10px) !important;
    position: relative; z-index: 2;
  }

  /* ─── CTA SECTION ─── */
  .cta-dark {
    margin: 100px auto 0; max-width: 1100px;
    background: var(--espresso);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 32px; padding: clamp(48px,7vw,90px);
    text-align: center; position: relative; overflow: hidden; z-index: 2;
  }
  .cta-dark::before {
    content: ''; position: absolute; inset: -1px; border-radius: 32px;
    background: linear-gradient(135deg, rgba(196,168,130,0.5), rgba(160,120,80,0.3), rgba(107,79,53,0.2), rgba(196,168,130,0.5));
    background-size: 300% 300%; animation: borderflow 8s linear infinite;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
  }
  .cta-dark::after {
    content: ''; position: absolute; width: 600px; height: 600px; border-radius: 50%;
    background: radial-gradient(circle, rgba(196,168,130,0.18), transparent 70%);
    top: 50%; left: 50%; transform: translate(-50%,-50%); pointer-events: none;
  }
  @keyframes borderflow { to { background-position: 300% 0; } }
  .cta-dark h2 {
    font-family: 'Sora', sans-serif !important;
    color: var(--cream); font-size: clamp(2rem,5vw,3.5rem); font-weight: 800; letter-spacing: -.04em;
    margin: 0 0 16px; position: relative; z-index: 1; line-height: 1.1;
  }
  .cta-dark p {
    color: rgba(250,247,242,0.55); font-size: 17px; max-width: 520px; margin: 0 auto;
    line-height: 1.7; position: relative; z-index: 1; font-weight: 400;
  }

  /* ─── FOOTER ─── */
  .footer {
    margin-top: 80px; border-top: 1px solid rgba(160,120,80,0.12);
    padding: 40px 0 20px; position: relative; z-index: 2;
  }
  .footer-inner {
    max-width: 1100px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 16px;
  }
  .footer .logo { font-size: 17px; }
  .footer p { color: rgba(107,79,53,0.45); font-size: 13px; margin: 0; }
  .footer p b { color: var(--amber); }

  /* ─── ANIMATIONS ─── */
  @keyframes fadeUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: none; } }
  .anim { animation: fadeUp .9s cubic-bezier(.22,1,.36,1) both; }
  .d1 { animation-delay: .1s; } .d2 { animation-delay: .22s; } .d3 { animation-delay: .34s; } .d4 { animation-delay: .46s; }

  /* ─── STREAMLIT OVERRIDES ─── */
  div[data-testid="stAudio"] { border-radius: 12px; overflow: hidden; }
  div[data-testid="stMarkdownContainer"] { color: var(--walnut) !important; }
  div[data-testid="stMarkdownContainer"] strong { color: var(--espresso) !important; }
  div[data-testid="stMarkdownContainer"] em { color: var(--amber) !important; }
  .stAlert { border-radius: 14px !important; border: 1px solid rgba(160,120,80,0.18) !important; }

  /* ─── RESPONSIVE ─── */
  @media (max-width: 900px) {
    .nav { margin: 0 -16px; padding: 0 16px; }
    .nav-links { display: none; }
    .pitch-grid, .how-grid, .bundle-grid { grid-template-columns: 1fr; }
    .post-wrap { grid-template-columns: 1fr; }
    .phone { width: 240px; }
    .stat { padding: 20px 24px; }
    .ticker { margin: 60px -16px 0; }
    .stats { border-radius: 16px; }
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

# ================= BACKEND (unchanged brain) =================
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

# ================= DARK CTA + FOOTER =================
st.html("""
<div class="cta-dark">
  <h2>Your repo deserves more<br>than <span class="serif-accent">a README.</span></h2>
  <p>Paste a link above and walk away with a complete launch kit — copy, visuals, voiceovers, and a demo pitch. Powered by real code analysis.</p>
</div>
<div class="footer"><div class="footer-inner">
  <div class="logo">
    <div class="logo-icon" style="display:inline-flex;width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#6366F1,#A855F7);align-items:center;justify-content:center;font-size:13px;margin-right:8px;vertical-align:middle;">⚡</div>
    <span style="vertical-align:middle;">Hype<span style="background:linear-gradient(120deg,#818CF8,#C084FC);-webkit-background-clip:text;background-clip:text;color:transparent;">Repo</span></span>
  </div>
  <p>Built with <b>HypeRepo</b> — paste a repo, ship the hype. © 2026</p>
</div></div>
""")
