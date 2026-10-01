import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace CSS
start_marker = '# ================= DESIGN SYSTEM =================\n'
end_marker = '# ================= NAV =================\n'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print('Failed to find CSS markers')
    sys.exit(1)

new_css = '''st.html("""
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
  .phone{width:240px;margin:10px auto 0;background:#18181B;border-radius:44px;padding:9px;box-shadow:0 24px 48px -12px rgba(0,0,0,.3),0 0 0 1px rgba(255,255,255,.06);transition:transform .3s cubic-bezier(.34,1.56,.64,1);}
  .phone:hover{transform:rotate(-1.5deg) scale(1.02);}
  .phone-screen{background:#fff;border-radius:38px;overflow:hidden;position:relative;}
  .notch{position:absolute;top:9px;left:50%;transform:translateX(-50%);width:80px;height:20px;background:#18181B;border-radius:999px;z-index:2;}
  .ig-head{display:flex;align-items:center;gap:9px;padding:36px 12px 9px;}
  .ig-avatar{width:28px;height:28px;border-radius:50%;flex-shrink:0;background:#18181B;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:800;}
  .ig-user{font-size:12px;font-weight:700;color:#18181B;}
  .ig-img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--bg3);}
  .ig-actions{display:flex;gap:12px;padding:9px 12px 4px;font-size:17px;}
  .ig-cap{padding:4px 12px 14px;font-size:11.5px;color:#374151;line-height:1.55;}
  .ig-cap b{color:#18181B;}
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
    .phone{width:220px;}
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
\n'''

content = content[:start_idx + len(start_marker)] + new_css + content[end_idx:]

# 2. Fix footer logo
footer_target = '''<div class="logo-icon" style="display:inline-flex;width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#6366F1,#A855F7);align-items:center;justify-content:center;font-size:13px;margin-right:8px;vertical-align:middle;">⚡</div>
    <span style="vertical-align:middle;">Hype<span style="background:linear-gradient(120deg,#818CF8,#C084FC);-webkit-background-clip:text;background-clip:text;color:transparent;">Repo</span></span>'''

footer_replacement = '''<div class="logo-icon">⚡</div>
    <div class="logo-text">Hype<span>Repo</span></div>'''

content = content.replace(footer_target, footer_replacement)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
