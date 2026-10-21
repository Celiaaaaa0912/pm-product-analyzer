import streamlit as st
from openai import OpenAI
import os
import json
import html
import io
import re
import urllib.parse
import PyPDF2
import docx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ================================================================ setup

def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY")

API_KEY = get_api_key()
MODEL = "deepseek-chat"

def get_client():
    return OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def parse_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())

# ================================================================ analysis prompt

SYSTEM_PROMPT = """You are a skeptical senior product manager with 15+ years of experience taking early-stage
products from idea to launch. Your job is to find the assumptions the idea depends on, rank the riskiest ones,
and lay out a concrete, cheap validation plan the founder can start this week.

Rules:
- Be critical and specific. No generic advice. Use concrete numbers, names, channels, and sample sizes.
- In assumption, why_risky, pass_signal, details, and decision_rule, wrap the single most important phrase
  in **double asterisks** so it stands out. Bold only one short phrase per field.
- You have no live market data. For market size, show a bottom-up calculation (users x price x frequency)
  and say how to verify each input.
- Pick exactly 3 riskiest assumptions, most dangerous first. Classify each as Desirability (do people want it),
  Viability (will it make money), Feasibility (can it be built), or Usability (can people use it).
- Validation plan: 3-4 steps ordered from cheapest/fastest to most expensive. Test desirability before
  viability, and both before building anything expensive. Each step names the method, sample size,
  exactly where to recruit participants, the deliverable, and a decision rule (continue / change / stop).
- Competitors: 4-6 real companies or products when possible (direct, indirect, and substitutes). Give typical
  price, key features, and target user. Choose two axes for a positioning map that matter most to buyers,
  and place each competitor and the user's product on 1-10 scales.
- Business model canvas: 2-4 specific bullet points per block.
- Pain points: 4-6 specific pains, each with who has it, how they cope today, how often it happens,
  how severe it is, which feature addresses it, and which validation step will confirm it.
- Cost model: classify the product as Hardware, Software, or Connected (hardware plus an app or cloud service).
  For Hardware or Connected, list a bill of materials (BOM) of 6-12 main components with rough unit costs in USD.
  For Software or Connected, list monthly cost-to-serve items per user (hosting, AI or API calls, support,
  payment fees). Use plain numbers without currency symbols in numeric fields.
- The score (0-100) is a secondary reference only. Above 70 should be rare.
{mode_rules}

Return ONLY valid json in exactly this structure:
{{
  "one_line_summary": "what the idea is, in plain words",
  "assumed_target_user": "who this is for, as specifically as possible",
  "stage": "Idea | Prototype | MVP | Launched",
  "confidence": "Low | Medium | High",
  "confidence_reason": "two sentences: what is known and what is not",
  "score": 0,
  "riskiest_assumptions": [
    {{
      "assumption": "the belief the idea depends on",
      "risk_type": "Desirability | Viability | Feasibility | Usability",
      "why_risky": "what happens if it is wrong",
      "test_method": "a concrete test",
      "test_time": "e.g. 3 days",
      "sample": "e.g. 8 interviews",
      "pass_signal": "the result that means the assumption holds"
    }}
  ],
  "validation_plan": [
    {{
      "step": "short name",
      "timing": "e.g. Week 1",
      "goal": "which assumption this tests and why now",
      "method": "Interviews | Survey | Landing page | Prototype test | Concierge test | Pilot | Desk research",
      "details": "what exactly to do",
      "sample_size": "e.g. 8-10 people",
      "where_to_recruit": "specific places, e.g. r/SkincareAddiction, campus clubs, LinkedIn groups",
      "deliverable": "what you will have at the end",
      "decision_rule": "continue if..., change if..., stop if..."
    }}
  ],
  "what_would_change_my_mind": {{
    "raise_confidence": ["specific evidence"],
    "lower_confidence": ["specific evidence"]
  }},
  "market": {{
    "tam": {{"value": "e.g. $2B", "calculation": "bottom-up formula", "how_to_verify": "..."}},
    "sam": {{"value": "...", "calculation": "...", "how_to_verify": "..."}},
    "som": {{"value": "...", "calculation": "...", "how_to_verify": "..."}},
    "trends": ["..."]
  }},
  "positioning_axes": {{"x_label": "e.g. Price (low to high)", "y_label": "e.g. Routine guidance (none to full)"}},
  "your_position": {{"x": 5, "y": 5}},
  "competitors": [
    {{
      "name": "...", "type": "Direct | Indirect | Substitute", "price": "typical price",
      "key_features": "...", "target_user": "...", "strength": "...", "weakness": "...", "x": 5, "y": 5
    }}
  ],
  "user_pain_points": {{
    "jtbd": ["When I..., I want to..., so I can..."],
    "pains": [
      {{
        "pain": "...", "who": "...", "current_workaround": "how they cope today",
        "frequency": "Daily | Weekly | Monthly | Rarely", "severity": "High | Medium | Low",
        "addressed_by": "which feature", "validate_in": "e.g. Step 1"
      }}
    ]
  }},
  "business_model": {{
    "key_partners": ["..."], "key_activities": ["..."], "key_resources": ["..."],
    "value_propositions": ["..."], "customer_relationships": ["..."], "channels": ["..."],
    "customer_segments": ["..."], "cost_structure": ["..."], "revenue_streams": ["..."]
  }},
  "cost_model": {{
    "product_type": "Hardware | Software | Connected",
    "price_per_unit": 0,
    "subscription_price_per_month": 0,
    "bom": [{{"component": "...", "qty": 1, "unit_cost": 0, "note": "..."}}],
    "manufacturing_and_logistics_per_unit": 0,
    "cost_to_serve": [{{"item": "...", "monthly_cost_per_user": 0, "note": "..."}}],
    "customer_acquisition_cost": 0,
    "expected_lifetime_months": 24,
    "biggest_unknown": "..."
  }}
}}"""

B2C_RULES = "- Treat this as a consumer product: the user and the buyer are usually the same person."
B2B_RULES = """- Treat this as a B2B product. Separate the BUYER (who pays and signs) from the USER (who uses it daily).
- At least one riskiest assumption must be about the buyer: budget owner, procurement, switching cost, or sales cycle.
- Consider compliance, security review, and integration with existing systems."""

def analyze_product(product_input, mode, previous=None):
    system_prompt = SYSTEM_PROMPT.format(mode_rules=B2B_RULES if mode == "B2B" else B2C_RULES)
    user_msg = "Analyze this product concept and return json only:\n\n" + product_input
    if previous:
        user_msg += ("\n\nThis is a revised version. The previous riskiest assumptions were:\n- "
                     + "\n- ".join(a.get("assumption", "") for a in previous.get("riskiest_assumptions", []))
                     + "\nRe-rank them for the revised idea.")
    response = get_client().chat.completions.create(
        model=MODEL, max_tokens=7000, temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_msg}])
    text = response.choices[0].message.content
    try:
        return parse_json(text)
    except Exception:
        return {"error": text}

# ================================================================ test kit

KIT_PROMPT = """You are a senior UX researcher and market research analyst. Write ready-to-use research materials
for a founder who will run this validation step tomorrow. Be concrete and specific to the product.
Rules:
- Interview questions ask about past behavior and real situations, never "would you use this?" or leading questions.
- Surveys: start with 2-3 screener questions, use clear single-topic questions, include answer options.
  If price is being tested, include the four Van Westendorp price sensitivity questions.
  End with a short "How to analyze the results" section.
- Recruiting posts are short, honest about time needed, and say what participants get.
- Landing page copy: headline, subheadline, 3 benefits, call to action, price line, and what metric to track.
Write in clear English using Markdown headings. Only include the sections requested."""

KIT_OPTIONS = {
    "Screener questions": "Screener questions to find the right participants",
    "Interview guide": "A 30-minute interview guide with intro script, 10-12 questions, and probes",
    "Survey": "A survey of 12-15 questions",
    "Recruiting post": "A recruiting post for the places named in the plan",
    "Landing page copy": "Landing page copy for a smoke test",
    "Prototype test script": "A usability test script with 3-5 tasks and success criteria",
}

def default_kit(method):
    m = (method or "").lower()
    if "interview" in m:
        return ["Screener questions", "Interview guide", "Recruiting post"]
    if "survey" in m:
        return ["Survey", "Recruiting post"]
    if "landing" in m:
        return ["Landing page copy", "Recruiting post"]
    if "prototype" in m or "usability" in m or "concierge" in m or "pilot" in m:
        return ["Screener questions", "Prototype test script", "Recruiting post"]
    return ["Screener questions", "Interview guide"]

def generate_kit(result, step, parts):
    brief = {
        "product": result.get("one_line_summary"),
        "target_user": result.get("assumed_target_user"),
        "riskiest_assumptions": [a.get("assumption") for a in result.get("riskiest_assumptions", [])],
        "validation_step": step,
        "sections_requested": [KIT_OPTIONS[p] for p in parts],
    }
    response = get_client().chat.completions.create(
        model=MODEL, max_tokens=5000, temperature=0.4,
        messages=[{"role": "system", "content": KIT_PROMPT},
                  {"role": "user", "content": json.dumps(brief, ensure_ascii=False, indent=2)}])
    return response.choices[0].message.content

# ================================================================ PRD / MRD

MRD_TEMPLATE = """# Market Requirements Document: [Product name]
## 1. Executive summary
## 2. Market problem
## 3. Target market and segments
## 4. Market size (TAM / SAM / SOM, with calculation)
## 5. Customer needs and jobs-to-be-done
## 6. Competitive landscape
## 7. Positioning and differentiation
## 8. Pricing and business model
## 9. Go-to-market approach
## 10. Success criteria
## 11. Assumptions and open questions
## 12. Validation plan"""

PRD_TEMPLATE = """# Product Requirements Document: [Product name]
## 1. Overview (problem, solution, why now)
## 2. Goals and non-goals
## 3. Target users and personas
## 4. User stories and key use cases
## 5. Requirements (table: ID, requirement, priority P0/P1/P2, rationale, acceptance criteria)
## 6. MVP scope (in scope / deferred, with reasons)
## 7. User flow
## 8. Success metrics (North Star, supporting metrics, guardrail metrics)
## 9. Dependencies and technical considerations
## 10. Risks and mitigations
## 11. Assumptions and open questions
## 12. Release plan and milestones"""

DOC_PROMPT = """You are a senior product manager writing a {doc_type} for an early-stage product.
Write it from the analysis and inputs provided. Be specific and concrete; no filler.
Rules:
- Follow the template's section structure exactly, in the same order, using Markdown headings.
- Never invent research findings, quotes, or user counts. Where real research is needed but not provided,
  write a clear placeholder in square brackets, e.g. [Add findings from 8 user interviews].
- Mark any statement that depends on an untested assumption with "(to validate)".
- Requirements: prioritize as P0 / P1 / P2 with a one-line rationale and testable acceptance criteria,
  and keep P0 to the smallest set that tests the riskiest assumptions.
- Metrics: one North Star metric, 2-3 supporting metrics, 1-2 guardrail metrics, each with a definition.
- If test results are provided, use them, and move any assumption they confirm out of open questions.
- Use Markdown tables where the template asks for a table. Keep tables to 5 columns or fewer with short cells.
- Define each persona in the personas section with a name and a one-line description, and mark it (to validate).
- Write user stories as a bulleted list: "As a [persona], I want [action], so that [benefit]." Do not use a table.
Template:
{template}"""

def generate_doc(result, idea, doc_type, template, test_results):
    payload = {
        "original_description": idea,
        "analysis": result,
        "test_results_so_far": test_results or "None yet",
    }
    response = get_client().chat.completions.create(
        model=MODEL, max_tokens=7000, temperature=0.3,
        messages=[{"role": "system", "content": DOC_PROMPT.format(doc_type=doc_type, template=template)},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    return response.choices[0].message.content

def add_runs(par, text):
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part:
            par.add_run(part)

def markdown_to_docx(md):
    d = docx.Document()
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block
                    if not re.match(r"^\|?\s*:?-{2,}", r.replace("|", "", 1))]
            if rows:
                t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows))
                t.style = "Table Grid"
                for ri, r in enumerate(rows):
                    for ci, c in enumerate(r):
                        cell = t.cell(ri, ci)
                        cell.text = ""
                        add_runs(cell.paragraphs[0], c)
                        if ri == 0:
                            for run in cell.paragraphs[0].runs:
                                run.bold = True
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            d.add_heading(m.group(2).strip(), level=min(len(m.group(1)), 3) if len(m.group(1)) > 1 else 0)
        elif re.match(r"^\s*[-*]\s+", line):
            add_runs(d.add_paragraph(style="List Bullet"), re.sub(r"^\s*[-*]\s+", "", line))
        elif re.match(r"^\s*\d+\.\s+", line):
            add_runs(d.add_paragraph(style="List Number"), re.sub(r"^\s*\d+\.\s+", "", line))
        elif line.strip() and not re.match(r"^-{3,}$", line.strip()):
            add_runs(d.add_paragraph(), line.strip())
        i += 1
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()

def read_template(file):
    name = file.name.lower()
    if name.endswith(".docx"):
        d = docx.Document(file)
        out = []
        for p in d.paragraphs:
            if not p.text.strip():
                continue
            if p.style.name.lower().startswith("heading") or p.style.name.lower() == "title":
                out.append("## " + p.text.strip())
            else:
                out.append(p.text.strip())
        return "\n".join(out)
    if name.endswith(".pdf"):
        return extract_text_from_file(file)
    return file.read().decode("utf-8", errors="ignore")

# ================================================================ display helpers

TYPE_INFO = {
    "Desirability": ("Do people want it?", "#1a6eb5"),
    "Viability":    ("Will it make money?", "#1d9e75"),
    "Feasibility":  ("Can it be built?",   "#8a63d2"),
    "Usability":    ("Can people use it?", "#d9822b"),
}

CSS = """
<style>
.pa-table {width:100%; border-collapse:collapse; font-size:14px; line-height:1.55; table-layout:auto;}
.pa-table th {text-align:left; font-weight:600; font-size:13px; color:#4b5563; background:#f6f7f9;
              padding:10px 12px; border-bottom:1px solid #e3e5e8;}
.pa-table td {vertical-align:top; padding:12px; border-bottom:1px solid #eceef1; color:#1f2937;
              word-break:normal; overflow-wrap:anywhere;}
.pa-badge {display:inline-block; border-radius:6px; padding:2px 9px; font-size:12px; font-weight:600;
           color:#fff; white-space:nowrap;}
.pa-card {border:1px solid #e3e5e8; border-radius:12px; padding:16px 18px; margin:0 0 14px 0; background:#fff;}
.pa-label {font-size:12px; font-weight:600; color:#6b7280; margin:10px 0 2px 0;}
.pa-text {font-size:14px; line-height:1.6; color:#1f2937;}
.pa-muted {font-size:13px; color:#6b7280; line-height:1.5;}
.pa-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px;}
.pa-type {border:1px solid #e3e5e8; border-radius:12px; padding:12px 14px;}
.pa-note {background:#f6f7f9; border-radius:8px; padding:10px 14px; font-size:13px; color:#4b5563; margin:4px 0 14px 0;}
.pa-table b, .pa-text b {font-weight:650; color:#111827;}
div[data-testid="stMarkdownContainer"] table {font-size:13px; width:100%;}
div[data-testid="stMarkdownContainer"] th, div[data-testid="stMarkdownContainer"] td {white-space:normal; vertical-align:top;}
div[data-testid="stMarkdownContainer"] td:first-child {white-space:nowrap;}
</style>
"""

def rich(text):
    """Escape, protect $ from being read as math, and turn **x** into bold."""
    t = html.escape(str(text if text is not None else ""))
    t = t.replace("$", "&#36;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)

def md_safe(text):
    """For plain st.markdown output: keep dollar signs from turning into math."""
    return str(text or "").replace("$", "\\$")

def show(html_str):
    st.markdown(html_str, unsafe_allow_html=True)

def type_badge(t):
    key = next((k for k in TYPE_INFO if k.lower() in str(t).lower()), None)
    if not key:
        return rich(t)
    return f'<span class="pa-badge" style="background:{TYPE_INFO[key][1]};">{key}</span>'

def note(text):
    show(f'<div class="pa-note">{rich(text)}</div>')

def bullets(items):
    return "".join(f'<div style="margin:5px 0;">• {rich(x.get("point") or x.get("claim") if isinstance(x, dict) else x)}</div>'
                   for x in (items or []))

def search_url(name):
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(str(name))

def to_num(v, default=5.0):
    try:
        return max(0.0, min(10.0, float(v)))
    except Exception:
        return default

def beta_pdf(x, a, b):
    from math import lgamma
    log_b = lgamma(a) + lgamma(b) - lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - log_b)

# ================================================================ page

st.set_page_config(page_title="PM Product Analyzer", page_icon="📊", layout="wide")
show(CSS)
st.title("📊 PM Product Analyzer")
st.caption("Enter a product concept. Get its riskiest assumptions, a step-by-step validation plan, "
           "ready-to-use research materials, an MRD or PRD draft, and a market, competitor, and business model overview.")

SECTIONS = ["🎯 Riskiest Assumptions", "🧪 Validation Plan", "🏆 Competitors", "📈 Market Size",
            "👤 User Pain Points", "💰 Business Model", "📄 MRD / PRD"]

for k, v in {"history": [], "kits": {}, "doc": None, "nav": SECTIONS[0], "run_id": 0}.items():
    st.session_state.setdefault(k, v)

if not API_KEY:
    st.error("DEEPSEEK_API_KEY is not set. In Streamlit Cloud, open Manage app → Settings → Secrets, "
             "add DEEPSEEK_API_KEY = \"sk-...\", save, then reboot the app.")

with st.expander("📋 Suggested input format"):
    st.markdown("""
Fill in as many fields as possible for a more accurate analysis:

```
Product name: [name]
Target users: [who are they, age range, role]
Core problem: [what pain point are you solving]
Solution: [how your product solves it]
Market: [which market/industry]
Monetization: [how you plan to make money]
Stage: [idea / MVP / prototype]
```
""")

st.subheader("Enter Product Information")
mode = st.radio("Business type", ["B2C", "B2B"], horizontal=True,
                help="B2B separates the buyer (who pays) from the user (who uses it).")

col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area(
        "Text Description",
        placeholder="Product name: ...\nTarget users: ...\nCore problem: ...\nSolution: ...\nMarket: ...\nMonetization: ...",
        height=220, key="text_input")
with col2:
    uploaded_file = st.file_uploader("Or upload a PDF / DOCX", type=["pdf", "docx"],
                                     key=f"uploader_{st.session_state.get('uploader_key', 0)}")
    file_text = ""
    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        st.success(f"File read successfully — {len(file_text)} characters extracted.")

final_input = (text_input or "") + ("\n\n" + file_text if file_text else "")

def clear_inputs():
    st.session_state["text_input"] = ""
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state["history"] = []
    st.session_state["kits"] = {}
    st.session_state["doc"] = None
    st.session_state["nav"] = SECTIONS[0]

disabled = not final_input.strip() or not API_KEY
if st.session_state["history"]:
    st.caption("Changed the same idea? Choose Update to compare with the last result. "
               "Working on a different product? Choose New idea.")
    b1, b2, b3, _ = st.columns([1.6, 1.6, 1, 4])
    with b1:
        update = st.button("Update this idea", type="primary", disabled=disabled)
    with b2:
        new_idea = st.button("Analyze as a new idea", disabled=disabled)
    with b3:
        st.button("Clear", type="secondary", on_click=clear_inputs)
    if new_idea:
        st.session_state["history"] = []
    analyze = update or new_idea
else:
    b1, b2, _ = st.columns([1.5, 1, 6])
    with b1:
        analyze = st.button("Analyze Product", type="primary", disabled=disabled)
    with b2:
        st.button("Clear", type="secondary", on_click=clear_inputs)

if analyze:
    previous = st.session_state["history"][-1]["result"] if st.session_state["history"] else None
    with st.spinner("Analyzing your product concept... this may take 30-60 seconds."):
        try:
            result = analyze_product(final_input, mode, previous)
        except Exception as e:
            result = {"error": f"Request failed: {e}"}
    if "error" in result:
        st.error("Analysis failed. Click the button again. Details:")
        st.code(result["error"])
    else:
        st.session_state["history"].append({"idea": final_input, "mode": mode, "result": result})
        st.session_state["kits"] = {}
        st.session_state["run_id"] += 1
        st.session_state["doc"] = None
        st.session_state["nav"] = SECTIONS[0]

if not st.session_state["history"]:
    st.stop()

# ================================================================ results

result = st.session_state["history"][-1]["result"]
prev = st.session_state["history"][-2]["result"] if len(st.session_state["history"]) > 1 else None

if prev:
    st.success("Analysis updated. The Validation Plan section compares it with your previous description.")
else:
    st.success("Analysis complete!")
st.divider()

# A stateful menu instead of st.tabs, so clicking a button never sends you back to the first section.
if hasattr(st, "segmented_control"):
    st.segmented_control("Section", SECTIONS, key="nav", label_visibility="collapsed")
else:
    st.radio("Section", SECTIONS, key="nav", horizontal=True, label_visibility="collapsed")
section = st.session_state.get("nav") or SECTIONS[0]
st.write("")

# ---------------- riskiest assumptions
if section == SECTIONS[0]:
    show(f'<h3 style="margin-bottom:4px;">{rich(result.get("one_line_summary", ""))}</h3>'
         f'<div class="pa-muted">Assumed target user: {rich(result.get("assumed_target_user", ""))}'
         f' &nbsp;|&nbsp; Stage: {rich(result.get("stage", "—"))}</div>')
    st.write("")
    risks = result.get("riskiest_assumptions", [])
    present = {k for k in TYPE_INFO for a in risks if k.lower() in str(a.get("risk_type", "")).lower()}
    cards = []
    for k, (q, color) in TYPE_INFO.items():
        n = sum(1 for a in risks if k.lower() in str(a.get("risk_type", "")).lower())
        style = f"border-left:5px solid {color};" + ("" if k in present else "opacity:0.45;")
        cards.append(f'<div class="pa-type" style="{style}"><div style="font-weight:650;color:{color};">{k}</div>'
                     f'<div class="pa-text">{q}</div><div class="pa-muted">{n} risky assumption{"s" if n != 1 else ""}</div></div>')
    show('<div class="pa-text" style="margin-bottom:8px;">Every idea has to pass four checks. '
         'This is where your idea is weakest right now:</div><div class="pa-grid">' + "".join(cards) + "</div>")
    st.write("")
    st.markdown("#### The three riskiest assumptions, most dangerous first")
    rows = "".join(
        f"<tr><td><b>{i}</b></td><td>{type_badge(a.get('risk_type'))}</td><td>{rich(a.get('assumption'))}</td>"
        f"<td>{rich(a.get('why_risky'))}</td><td>{rich(a.get('test_method'))}"
        f"<div class='pa-muted' style='margin-top:6px;'>{rich(a.get('test_time'))} · {rich(a.get('sample'))}</div></td>"
        f"<td>{rich(a.get('pass_signal'))}</td></tr>"
        for i, a in enumerate(risks, 1))
    show('<table class="pa-table"><tr><th style="width:3%">#</th><th style="width:10%">Type</th>'
         '<th style="width:22%">Assumption</th><th style="width:22%">If this is wrong</th>'
         '<th style="width:23%">Cheapest test</th><th style="width:20%">It holds if</th></tr>' + rows + "</table>")

# ---------------- validation plan
elif section == SECTIONS[1]:
    conf = str(result.get("confidence", "")).split("|")[0].strip().title()
    color = {"High": "#1d9e75", "Medium": "#d9822b"}.get(conf, "#c0392b")
    try:
        score = int(float(result.get("score", 0)))
    except Exception:
        score = 0
    show(f'<div class="pa-card" style="border-left:6px solid {color};">'
         f'<div style="font-size:20px;font-weight:700;color:{color};">Confidence: {rich(conf)}</div>'
         f'<div class="pa-text" style="margin-top:6px;">{rich(result.get("confidence_reason", ""))}</div>'
         f'<div class="pa-muted" style="margin-top:8px;">Reference score {score} / 100. Confidence shows how much is '
         f'already known, not how good the idea is. The plan below is how you raise it.</div></div>')
    if prev:
        note(f"Compared with your previous description: confidence {prev.get('confidence', '?')} → {conf}, "
             f"score {prev.get('score', '?')} → {score}")

    st.markdown("#### Your validation plan")
    st.caption("Cheapest and fastest tests first. Move to the next step only if the decision rule says continue.")
    plan = result.get("validation_plan", [])
    for i, s_ in enumerate(plan, 1):
        show(f'''<div class="pa-card">
<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
  <span style="font-size:17px;font-weight:700;">Step {i}: {rich(s_.get("step"))}</span>
  <span class="pa-badge" style="background:#4b5563;">{rich(s_.get("method"))}</span>
  <span class="pa-muted">{rich(s_.get("timing"))}</span></div>
<div class="pa-label">Goal</div><div class="pa-text">{rich(s_.get("goal"))}</div>
<div class="pa-label">What to do</div><div class="pa-text">{rich(s_.get("details"))}</div>
<div class="pa-grid" style="margin-top:4px;">
  <div><div class="pa-label">Sample size</div><div class="pa-text">{rich(s_.get("sample_size"))}</div></div>
  <div><div class="pa-label">Where to recruit</div><div class="pa-text">{rich(s_.get("where_to_recruit"))}</div></div>
  <div><div class="pa-label">You'll end up with</div><div class="pa-text">{rich(s_.get("deliverable"))}</div></div>
</div>
<div class="pa-label">Decision rule</div><div class="pa-text">{rich(s_.get("decision_rule"))}</div>
</div>''')
        parts = default_kit(s_.get("method"))
        if st.button(f"Write materials for step {i}: {', '.join(parts).lower()}", key=f"kitbtn_{i}"):
            with st.spinner("Writing your research materials... about 30 seconds."):
                try:
                    st.session_state["kits"][i] = generate_kit(result, s_, parts)
                except Exception as e:
                    st.error(f"Request failed: {e}")
        if i in st.session_state["kits"]:
            with st.expander(f"Materials for step {i}", expanded=True):
                st.markdown(md_safe(st.session_state["kits"][i]))
                st.download_button("Download materials", st.session_state["kits"][i],
                                   file_name=f"step_{i}_materials.md", mime="text/markdown", key=f"kitdl_{i}")
        st.write("")

    w = result.get("what_would_change_my_mind", {})
    show('<div class="pa-grid">'
         f'<div class="pa-card"><div style="font-weight:650;color:#1d9e75;">Results that would raise confidence</div>'
         f'<div class="pa-text">{bullets(w.get("raise_confidence"))}</div></div>'
         f'<div class="pa-card"><div style="font-weight:650;color:#c0392b;">Results that would lower confidence</div>'
         f'<div class="pa-text">{bullets(w.get("lower_confidence"))}</div></div></div>')

    st.markdown("#### Tested two versions? Find out which one won")
    st.caption("For example, two landing pages with different prices. Enter how many people saw each version "
               "and how many signed up. Works even with small numbers.")
    with st.form("ab_form"):
        f1, f2, f3, f4 = st.columns(4)
        va = f1.number_input("Version A: visitors", 1, 1_000_000, 250)
        ca = f2.number_input("Version A: sign-ups", 0, 1_000_000, 12)
        vb = f3.number_input("Version B: visitors", 1, 1_000_000, 250)
        cb = f4.number_input("Version B: sign-ups", 0, 1_000_000, 20)
        go_ab = st.form_submit_button("Compare")
    if go_ab:
        if ca > va or cb > vb:
            st.error("Sign-ups can't be more than visitors.")
        else:
            rng = np.random.default_rng(7)
            sa = rng.beta(1 + ca, 1 + va - ca, 40000)
            sb = rng.beta(1 + cb, 1 + vb - cb, 40000)
            p_b = float((sb > sa).mean())
            lo_a, hi_a = np.percentile(sa, [2.5, 97.5]) * 100
            lo_b, hi_b = np.percentile(sb, [2.5, 97.5]) * 100
            leader, p_lead = ("B", p_b) if p_b >= 0.5 else ("A", 1 - p_b)
            if p_lead >= 0.95:
                verdict, box = f"Version {leader} is very likely better. Go with {leader}.", st.success
            elif p_lead >= 0.80:
                verdict, box = f"Leaning toward version {leader}. Collect a bit more data before deciding.", st.warning
            else:
                verdict, box = "No clear winner yet. Keep testing, or test a bigger difference.", st.info
            box(f"**{p_lead:.0%} chance version {leader} converts better.** {verdict}")
            m1, m2 = st.columns(2)
            m1.metric("Version A conversion", f"{ca / va:.1%}", help=f"Likely range {lo_a:.1f}%–{hi_a:.1f}%")
            m2.metric("Version B conversion", f"{cb / vb:.1%}", help=f"Likely range {lo_b:.1f}%–{hi_b:.1f}%")
            x = np.linspace(0.0005, min(0.9995, max(hi_a, hi_b) / 100 * 1.6), 400)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x * 100, y=beta_pdf(x, 1 + ca, 1 + va - ca), name="Version A",
                                     fill="tozeroy", line=dict(color="#3b8bd4")))
            fig.add_trace(go.Scatter(x=x * 100, y=beta_pdf(x, 1 + cb, 1 + vb - cb), name="Version B",
                                     fill="tozeroy", line=dict(color="#f0a500")))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(showticklabels=False),
                              xaxis=dict(title="Conversion rate (%)"), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each curve shows where that version's true conversion rate probably sits. "
                       "The less the curves overlap, the clearer the winner. Method: Bayesian A/B test with Beta posteriors.")

    if prev:
        with st.expander("Compare riskiest assumptions with your previous description"):
            pa, pb = st.columns(2)
            pa.markdown("**Previous**\n\n" + "\n".join(f"- {md_safe(x.get('assumption', ''))}" for x in prev.get("riskiest_assumptions", [])))
            pb.markdown("**Now**\n\n" + "\n".join(f"- {md_safe(x.get('assumption', ''))}" for x in result.get("riskiest_assumptions", [])))

# ---------------- competitors
elif section == SECTIONS[2]:
    comps = result.get("competitors", [])
    if comps:
        st.markdown("#### Competitor comparison")
        note("Prices and features come from the AI's general knowledge, not live data. Use Search to check current details.")
        rows = "".join(
            f"<tr><td><b>{rich(c.get('name'))}</b><div class='pa-muted'>{rich(c.get('type'))}</div></td>"
            f"<td>{rich(c.get('price'))}</td><td>{rich(c.get('key_features'))}</td><td>{rich(c.get('target_user'))}</td>"
            f"<td>{rich(c.get('strength'))}</td><td>{rich(c.get('weakness'))}</td>"
            f"<td><a href='{search_url(c.get('name', ''))}' target='_blank'>Search</a></td></tr>" for c in comps)
        show('<table class="pa-table"><tr><th style="width:14%">Product</th><th style="width:9%">Price</th>'
             '<th style="width:20%">Key features</th><th style="width:15%">Target user</th>'
             '<th style="width:17%">Strength</th><th style="width:17%">Weakness</th><th style="width:8%">Look up</th></tr>'
             + rows + "</table>")
        axes, you = result.get("positioning_axes", {}), result.get("your_position", {})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[to_num(c.get("x")) for c in comps], y=[to_num(c.get("y")) for c in comps],
                                 mode="markers+text", text=[c.get("name", "") for c in comps], textposition="top center",
                                 marker=dict(size=13, color="#3b8bd4")))
        fig.add_trace(go.Scatter(x=[to_num(you.get("x"))], y=[to_num(you.get("y"))], mode="markers+text",
                                 text=["Your product"], textposition="top center",
                                 marker=dict(size=18, color="#f0a500", symbol="star")))
        fig.add_hline(y=5, line_color="#ccc"); fig.add_vline(x=5, line_color="#ccc")
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), showlegend=False,
                          xaxis=dict(title=axes.get("x_label", ""), range=[0, 10.5]),
                          yaxis=dict(title=axes.get("y_label", ""), range=[0, 10.5]))
        st.write("")
        st.markdown("#### Positioning map")
        st.caption("Positions are the AI's estimate. Use it to spot open space, then confirm with users.")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No competitor data available.")

# ---------------- market
elif section == SECTIONS[3]:
    m = result.get("market", {})
    st.markdown("#### Market Size Overview")
    note("No live market data is used. These are bottom-up estimates: check the calculation and verify each input.")
    cols = st.columns(3)
    specs = [("TAM", "Total Addressable Market", "tam", "#1a6eb5"),
             ("SAM", "Serviceable Addressable Market", "sam", "#1d9e75"),
             ("SOM", "Serviceable Obtainable Market", "som", "#3b8bd4")]
    for col, (abbr, full, key, bg) in zip(cols, specs):
        item = m.get(key, {}) if isinstance(m.get(key), dict) else {"value": m.get(key)}
        with col:
            show(f'''<div style="background:{bg};border-radius:16px;padding:26px 16px;text-align:center;color:white;">
<div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">{abbr}</div>
<div style="font-size:11px;opacity:0.7;margin-bottom:8px;">{full}</div>
<div style="font-size:28px;font-weight:700;">{rich(item.get("value", "N/A"))}</div></div>
<div class="pa-label">How it's calculated</div><div class="pa-text">{rich(item.get("calculation", "—"))}</div>
<div class="pa-label">How to verify</div><div class="pa-muted">{rich(item.get("how_to_verify", "—"))}</div>''')
    st.write("")
    st.markdown("#### Key Market Trends")
    show(f'<div class="pa-text">{bullets(m.get("trends"))}</div>')

# ---------------- user pain
elif section == SECTIONS[4]:
    p = result.get("user_pain_points", {})
    st.markdown("#### Jobs-to-be-done")
    show(f'<div class="pa-card"><div class="pa-text">{bullets(p.get("jtbd"))}</div></div>')
    pains = p.get("pains") or [{"pain": x} for x in p.get("frustrations", [])]
    order = {"high": 0, "medium": 1, "low": 2}
    pains = sorted(pains, key=lambda x: order.get(str(x.get("severity", "")).lower(), 3))
    sev_color = {"high": "#c0392b", "medium": "#d9822b", "low": "#6b7280"}

    def sev_badge(v):
        c = sev_color.get(str(v).lower(), "#6b7280")
        return f'<span class="pa-badge" style="background:{c};">{rich(v)}</span>' if v else ""

    st.markdown("#### Pain points, most severe first")
    rows = "".join(
        f"<tr><td>{rich(x.get('pain'))}</td><td>{rich(x.get('who'))}</td><td>{rich(x.get('current_workaround'))}</td>"
        f"<td>{rich(x.get('frequency'))}</td><td>{sev_badge(x.get('severity'))}</td>"
        f"<td>{rich(x.get('addressed_by'))}</td><td>{rich(x.get('validate_in'))}</td></tr>" for x in pains)
    show('<table class="pa-table"><tr><th style="width:24%">Pain</th><th style="width:13%">Who</th>'
         '<th style="width:20%">How they cope today</th><th style="width:9%">How often</th>'
         '<th style="width:8%">Severity</th><th style="width:16%">Addressed by</th><th style="width:10%">Confirm in</th></tr>'
         + rows + "</table>")
    note("These are hypotheses, not research findings. A frequent, severe pain with a clumsy workaround is the "
         "strongest signal. Confirm the top two in your Step 1 interviews.")

# ---------------- business model
elif section == SECTIONS[5]:
    b = result.get("business_model", {})
    st.markdown("#### Business Model Canvas")
    cell_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#fff;height:100%;box-sizing:border-box;"
    label_style = "font-size:12px;font-weight:600;color:#666;margin-bottom:6px;"
    val_style = "font-size:13px;color:#222;line-height:1.55;overflow-wrap:break-word;"
    bot_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#f8f8f8;box-sizing:border-box;"

    def cell(icon, label, key, style=cell_style):
        v = b.get(key, [])
        v = [v] if isinstance(v, (str, dict)) else v
        return (f'<div style="{style}"><div style="{label_style}">{icon} {label}</div>'
                f'<div style="{val_style}">{bullets(v) or "—"}</div></div>')

    show(f"""
<table style="width:100%;border-collapse:separate;border-spacing:6px;table-layout:fixed;">
<tr>
  <td rowspan="2" style="width:20%;vertical-align:top;">{cell("🔗","Key Partners","key_partners")}</td>
  <td style="width:20%;vertical-align:top;">{cell("✅","Key Activities","key_activities")}</td>
  <td rowspan="2" style="width:20%;vertical-align:top;">{cell("🎁","Value Propositions","value_propositions")}</td>
  <td style="width:20%;vertical-align:top;">{cell("❤️","Customer Relationships","customer_relationships")}</td>
  <td rowspan="2" style="width:20%;vertical-align:top;">{cell("👥","Customer Segments","customer_segments")}</td>
</tr>
<tr>
  <td style="vertical-align:top;">{cell("🏗️","Key Resources","key_resources")}</td>
  <td style="vertical-align:top;">{cell("🚚","Channels","channels")}</td>
</tr>
<tr>
  <td colspan="2" style="vertical-align:top;">{cell("🏷️","Cost Structure","cost_structure", bot_style)}</td>
  <td></td>
  <td colspan="2" style="vertical-align:top;">{cell("💵","Revenue Streams","revenue_streams", bot_style)}</td>
</tr>
</table>""")
    cm = result.get("cost_model", {}) or {}

    def num(v, d=0.0):
        try:
            return float(str(v).replace("$", "").replace(",", ""))
        except Exception:
            return d

    ptype = str(cm.get("product_type", "")).split("|")[0].strip().title() or "Connected"
    st.write("")
    st.markdown(f"#### Cost model and unit economics · {rich(ptype)} product")
    note("The AI drafts rough numbers. Edit any cell or input below and every result updates. "
         "Hardware products use a bill of materials (BOM); software uses monthly cost to serve each user; "
         "connected products need both.")
    i1, i2, i3, i4 = st.columns(4)
    price = i1.number_input("One-time price (USD)", 0.0, 1e7, num(cm.get("price_per_unit")), 10.0, key=f"ue_price_{st.session_state['run_id']}")
    sub = i2.number_input("Subscription per month (USD)", 0.0, 1e5, num(cm.get("subscription_price_per_month")), 1.0, key=f"ue_sub_{st.session_state['run_id']}")
    cac = i3.number_input("Cost to acquire a customer (USD)", 0.0, 1e6, num(cm.get("customer_acquisition_cost")), 5.0, key=f"ue_cac_{st.session_state['run_id']}")
    life = i4.number_input("Customer lifetime (months)", 1.0, 240.0, max(1.0, num(cm.get("expected_lifetime_months"), 24)), 1.0, key=f"ue_life_{st.session_state['run_id']}")

    cogs, serve = 0.0, 0.0
    bom = cm.get("bom") or []
    if bom or ptype in ("Hardware", "Connected"):
        st.markdown("**Bill of materials (per unit)**")
        bom_df = pd.DataFrame([{"Component": r.get("component", ""), "Qty": num(r.get("qty"), 1),
                                "Unit cost (USD)": num(r.get("unit_cost")), "Note": r.get("note", "")} for r in bom]
                              or [{"Component": "", "Qty": 1.0, "Unit cost (USD)": 0.0, "Note": ""}])
        bom_edit = st.data_editor(bom_df, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"bom_editor_{st.session_state['run_id']}")
        bom_total = float((pd.to_numeric(bom_edit["Qty"], errors="coerce").fillna(0)
                           * pd.to_numeric(bom_edit["Unit cost (USD)"], errors="coerce").fillna(0)).sum())
        mfg = st.number_input("Assembly, packaging, and shipping per unit (USD)", 0.0, 1e6,
                              num(cm.get("manufacturing_and_logistics_per_unit")), 5.0, key=f"ue_mfg_{st.session_state['run_id']}")
        cogs = bom_total + mfg
        st.caption(f"Components {bom_total:,.2f} + assembly and shipping {mfg:,.2f} = cost per unit {cogs:,.2f} USD")

    serve_items = cm.get("cost_to_serve") or []
    if serve_items or ptype in ("Software", "Connected"):
        st.markdown("**Monthly cost to serve one user**")
        sv_df = pd.DataFrame([{"Item": r.get("item", ""), "Monthly cost per user (USD)": num(r.get("monthly_cost_per_user")),
                               "Note": r.get("note", "")} for r in serve_items]
                             or [{"Item": "", "Monthly cost per user (USD)": 0.0, "Note": ""}])
        sv_edit = st.data_editor(sv_df, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"serve_editor_{st.session_state['run_id']}")
        serve = float(pd.to_numeric(sv_edit["Monthly cost per user (USD)"], errors="coerce").fillna(0).sum())

    upfront = price - cogs
    monthly = sub - serve
    lifetime_profit = upfront + monthly * life
    ratio = lifetime_profit / cac if cac > 0 else None
    if cac <= upfront:
        payback = "At purchase"
    elif monthly > 0:
        payback = f"{(cac - upfront) / monthly:.1f} months"
    else:
        payback = "Never"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cost per unit", f"{cogs:,.0f}" if cogs else "—")
    m2.metric("Hardware margin", f"{upfront / price:.0%}" if price > 0 else "—")
    m3.metric("Cost to serve / month", f"{serve:,.2f}" if serve else "—")
    m4.metric("Lifetime profit per customer", f"{lifetime_profit:,.0f}")
    m5.metric("Lifetime profit : CAC", f"{ratio:.1f} : 1" if ratio is not None else "—", help="Payback: " + payback)
    st.caption(f"Payback on acquisition cost: {payback}. All amounts in USD.")

    flags = []
    if price > 0 and upfront / price < 0.4:
        flags.append("Hardware margin is under 40%, a level many consumer hardware companies try to stay above.")
    if ratio is not None and ratio < 3:
        flags.append("Lifetime profit is less than 3x the cost to acquire a customer, a common minimum for a healthy business.")
    if serve > 0 and sub == 0:
        flags.append(f"You pay about {serve * life:,.0f} USD in running costs per customer over {life:.0f} months "
                     "but earn nothing after the sale. Consider a subscription or lower ongoing costs.")
    for f in flags:
        st.warning(f)
    if cm.get("biggest_unknown"):
        note(f"Biggest unknown: {cm.get('biggest_unknown')}")

# ---------------- MRD / PRD
elif section == SECTIONS[6]:
    st.markdown("#### Write an MRD or PRD")
    st.write("Turn this analysis into a first draft you can edit. "
             "Anything that depends on an untested assumption is marked (to validate).")
    dtype = st.radio("Document", ["MRD", "PRD"], horizontal=True,
                     captions=["Market Requirements: who the market is and what it needs",
                               "Product Requirements: what to build first and how to measure it"])
    if dtype == "PRD" and str(result.get("confidence", "")).lower().startswith("low"):
        st.info("Confidence is still low, so many requirements will be marked (to validate). "
                "Many teams write the MRD first and the PRD after the first tests.")
    test_results = st.text_area("Test results so far (optional)", height=100,
                                placeholder="e.g. 7 of 9 interviewees track product expiry manually; "
                                            "survey: acceptable price range $199-$299")
    tpl_file = st.file_uploader("Use your own template (optional): DOCX, MD, TXT, or PDF",
                                type=["docx", "md", "txt", "pdf"], key="tpl_upload")
    template = MRD_TEMPLATE if dtype == "MRD" else PRD_TEMPLATE
    if tpl_file:
        try:
            template = read_template(tpl_file)
            st.success("Your template will be used.")
        except Exception as e:
            st.error(f"Could not read the template: {e}")
    with st.expander("See the template structure"):
        show('<div class="pa-text">' + "".join(
            f'<div style="margin:3px 0;">{rich(line.lstrip("# ").strip())}</div>'
            for line in template.splitlines() if line.strip()) + "</div>")
    if st.button(f"Write {dtype} draft", type="primary"):
        with st.spinner(f"Writing your {dtype}... about 40-60 seconds."):
            try:
                st.session_state["doc"] = {"type": dtype, "text": generate_doc(
                    result, st.session_state["history"][-1]["idea"], dtype, template, test_results)}
            except Exception as e:
                st.error(f"Request failed: {e}")
    doc = st.session_state.get("doc")
    if doc:
        with st.container(border=True):
            st.markdown(md_safe(doc["text"]))
        c1, c2, _ = st.columns([1.3, 1.3, 5])
        try:
            c1.download_button(f"Download {doc['type']} (Word)", markdown_to_docx(doc["text"]),
                               file_name=f"{doc['type']}_draft.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception:
            c1.caption("Word export failed for this draft. Use Markdown instead.")
        c2.download_button(f"Download {doc['type']} (Markdown)", doc["text"],
                           file_name=f"{doc['type']}_draft.md", mime="text/markdown")