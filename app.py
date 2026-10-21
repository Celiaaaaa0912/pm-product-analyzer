import streamlit as st
from openai import OpenAI
import os
import json
import html
import csv
import datetime
import PyPDF2
import docx

# ---------------------------------------------------------------- setup

def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY")

API_KEY = get_api_key()
TEMPERATURE = 0.2

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You are a skeptical senior product manager with 15+ years of experience reviewing early-stage products.
Your main job is NOT to decide whether the idea is good. Your job is to find the assumptions the idea depends on,
judge which are most likely to be wrong, and tell the founder the cheapest way to test them.

Rules:
- Be critical. Most early ideas rest on at least one unsupported assumption. Say so plainly.
- Label the basis of every claim:
    "Evidence"   = stated in the user's input, or widely documented public knowledge
    "Inference"  = reasoned from evidence, could be wrong
    "Assumption" = no evidence; must be tested
- You have no live market data. Any number about market size, growth, or pricing must be labeled
  "Assumption" unless the user provided it, and must include how_to_verify (a specific source or method).
- Choose exactly 3 riskiest assumptions, ordered from most to least dangerous. "Dangerous" means:
  if it is wrong, the idea fails, and there is little evidence for it yet.
- Each test must be doable by 1-2 people in under 2 weeks and have a clear pass signal.
- Confidence reflects how much is already known, not how exciting the idea is:
    "Low" = key assumptions untested; "Medium" = some support; "High" = core assumptions supported.
- The score (0-100) is a secondary reference only. Above 70 should be rare.
{mode_rules}

Return ONLY valid json in exactly this structure:
{{
  "one_line_summary": "what the idea is, in plain words",
  "assumed_target_user": "who this is for, as specifically as possible",
  "confidence": "Low | Medium | High",
  "confidence_reason": "one or two sentences",
  "score": 0,
  "riskiest_assumptions": [
    {{
      "assumption": "the belief the idea depends on",
      "why_risky": "what happens if it is wrong",
      "basis": "Evidence | Inference | Assumption",
      "cheapest_test": "a concrete test",
      "test_type": "Interview | Survey | Landing page | Prototype | Desk research | Pilot",
      "pass_signal": "what result would mean the assumption holds"
    }}
  ],
  "what_would_change_my_mind": {{
    "raise_confidence": ["specific evidence that would make this more promising"],
    "lower_confidence": ["specific evidence that would make this less promising"]
  }},
  "next_step": "the single most useful thing to do this week",
  "market": {{
    "tam": {{"value": "e.g. $15B", "basis": "...", "how_to_verify": "..."}},
    "sam": {{"value": "...", "basis": "...", "how_to_verify": "..."}},
    "som": {{"value": "...", "basis": "...", "how_to_verify": "..."}},
    "trends": [{{"claim": "...", "basis": "..."}}]
  }},
  "competitors": [
    {{"name": "...", "strengths": "...", "weaknesses": "...", "basis": "..."}}
  ],
  "user_pain_points": {{
    "jtbd": [{{"claim": "...", "basis": "..."}}],
    "frustrations": [{{"claim": "...", "basis": "..."}}]
  }},
  "business_model": {{
    "key_partners": {{"text": "...", "basis": "..."}},
    "key_activities": {{"text": "...", "basis": "..."}},
    "key_resources": {{"text": "...", "basis": "..."}},
    "value_propositions": {{"text": "...", "basis": "..."}},
    "customer_relationships": {{"text": "...", "basis": "..."}},
    "channels": {{"text": "...", "basis": "..."}},
    "customer_segments": {{"text": "...", "basis": "..."}},
    "cost_structure": {{"text": "...", "basis": "..."}},
    "revenue_streams": {{"text": "...", "basis": "..."}}
  }}
}}"""

B2C_RULES = "- Treat this as a consumer product: the user and the buyer are usually the same person."
B2B_RULES = """- Treat this as a B2B product. Separate the BUYER (who pays and signs) from the USER (who uses it daily).
- At least one riskiest assumption must be about the buyer: budget owner, procurement, switching cost, or sales cycle.
- Consider compliance, security review, and integration with existing systems."""

def analyze_product(product_input, mode):
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    system_prompt = SYSTEM_PROMPT.format(mode_rules=B2B_RULES if mode == "B2B" else B2C_RULES)
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=4000,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Analyze this product concept and return json only:\n\n" + product_input}
        ]
    )
    result_text = response.choices[0].message.content
    try:
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return {"error": result_text}

# ---------------------------------------------------------------- display helpers

BASIS_COLOR = {"Evidence": "#1d9e75", "Inference": "#f0a500", "Assumption": "#e03c3c"}
CONF_COLOR = {"High": "#1d9e75", "Medium": "#f0a500", "Low": "#e03c3c"}

def esc(x):
    return html.escape(str(x if x is not None else ""))

def clean_basis(b):
    b = str(b or "").split("|")[0].strip().title()
    return b if b in BASIS_COLOR else "Unlabeled"

def pill(b):
    b = clean_basis(b)
    color = BASIS_COLOR.get(b, "#999")
    return (f'<span style="background:{color};color:white;border-radius:10px;padding:1px 8px;'
            f'font-size:11px;font-weight:600;margin-left:6px;white-space:nowrap;">{b}</span>')

def claim_line(item):
    if isinstance(item, dict):
        return f'<div style="margin:4px 0;">• {esc(item.get("claim", ""))}{pill(item.get("basis"))}</div>'
    return f'<div style="margin:4px 0;">• {esc(item)}</div>'

def legend():
    st.markdown("Labels: " + pill("Evidence") + pill("Inference") + pill("Assumption"), unsafe_allow_html=True)

def to_markdown(r, idea, mode):
    out = [f"# Assumption review: {r.get('one_line_summary','')}", "",
           f"Mode: {mode} | Confidence: {r.get('confidence')} | Reference score: {r.get('score')}/100", "",
           f"Target user: {r.get('assumed_target_user','')}", "",
           f"Why this confidence: {r.get('confidence_reason','')}", "", "## Riskiest assumptions"]
    for i, a in enumerate(r.get("riskiest_assumptions", []), 1):
        out += [f"### {i}. {a.get('assumption')}", f"- Why risky: {a.get('why_risky')}",
                f"- Basis: {a.get('basis')}", f"- Cheapest test ({a.get('test_type')}): {a.get('cheapest_test')}",
                f"- Pass signal: {a.get('pass_signal')}", ""]
    w = r.get("what_would_change_my_mind", {})
    out += ["## What would change this", "Raise: " + "; ".join(w.get("raise_confidence", [])), "",
            "Lower: " + "; ".join(w.get("lower_confidence", [])), "",
            f"## Next step\n{r.get('next_step','')}", "", "---", "Original input:", "", idea]
    return "\n".join(out)

def log_feedback(answer, summary):
    # Resets when Streamlit Cloud restarts. Switch to Google Sheets for real user testing.
    try:
        with open("feedback.csv", "a", newline="") as f:
            csv.writer(f).writerow([datetime.datetime.now().isoformat(), answer, summary])
    except Exception:
        pass

# ---------------------------------------------------------------- page

st.set_page_config(page_title="PM Product Analyzer", page_icon="📊", layout="wide")
st.title("📊 PM Product Analyzer")
st.caption("Enter a product concept to find its riskiest assumptions, the cheapest way to test each one, "
           "and a market, competitor, and business model overview.")

if "history" not in st.session_state:
    st.session_state["history"] = []

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
        height=220,
        key="text_input"
    )
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

st.caption("Tip: edit your description and click Analyze again to compare versions.")
col_btn1, col_btn2, col_rest = st.columns([1.5, 1, 6])
with col_btn1:
    analyze = st.button("Analyze Product", type="primary", disabled=not final_input.strip() or not API_KEY)
with col_btn2:
    st.button("Clear", type="secondary", on_click=clear_inputs)

if analyze:
    with st.spinner("Analyzing your product concept... this may take 20-40 seconds."):
        try:
            result = analyze_product(final_input, mode)
        except Exception as e:
            result = {"error": f"Request failed: {e}"}
    if "error" in result:
        st.error("Analysis failed. Details:")
        st.code(result["error"])
    else:
        st.session_state["history"].append({"idea": final_input, "mode": mode, "result": result})

if not st.session_state["history"]:
    st.stop()

# ---------------------------------------------------------------- results

last = st.session_state["history"][-1]
result = last["result"]
prev = st.session_state["history"][-2]["result"] if len(st.session_state["history"]) > 1 else None

st.success(f"Analysis complete! (version {len(st.session_state['history'])})")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Riskiest Assumptions", "📈 Market Size", "🏆 Competitors",
    "👤 User Pain Points", "💰 Business Model", "🚦 Confidence & Next Step"
])

with tab1:
    st.markdown(f"### {esc(result.get('one_line_summary', ''))}")
    st.caption(f"Assumed target user: {result.get('assumed_target_user', '')}")
    st.write("If any of these turn out to be wrong, the idea is in trouble. Test them in this order.")
    for i, a in enumerate(result.get("riskiest_assumptions", []), 1):
        color = BASIS_COLOR.get(clean_basis(a.get("basis")), "#999")
        with st.expander(f"{i}. {a.get('assumption', '')}", expanded=(i == 1)):
            st.markdown(f"""<div style="background:#f5f5f5;border-radius:8px;padding:2px 8px;margin-bottom:10px;">
<div style="background:{color};height:8px;border-radius:6px;width:100%;"></div></div>""", unsafe_allow_html=True)
            st.markdown(f"**Current basis:** {pill(a.get('basis'))}", unsafe_allow_html=True)
            st.write(f"**If wrong:** {a.get('why_risky', '')}")
            st.write(f"**Cheapest test ({a.get('test_type', '')}):** {a.get('cheapest_test', '')}")
            st.write(f"**It holds if:** {a.get('pass_signal', '')}")

with tab2:
    m = result.get("market", {})
    st.markdown("### Market Size Overview")
    st.caption("No live market data is used. Treat these numbers as starting estimates and verify them.")
    cols = st.columns(3)
    specs = [("TAM", "Total Addressable Market", "tam", "#1a6eb5"),
             ("SAM", "Serviceable Addressable Market", "sam", "#1d9e75"),
             ("SOM", "Serviceable Obtainable Market", "som", "#3b8bd4")]
    for col, (abbr, full, key, bg) in zip(cols, specs):
        item = m.get(key, {})
        if not isinstance(item, dict):
            item = {"value": item}
        with col:
            st.markdown(f"""<div style="background:{bg};border-radius:16px;padding:32px 16px;text-align:center;color:white;">
<div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">{abbr}</div>
<div style="font-size:11px;opacity:0.7;margin-bottom:8px;">{full}</div>
<div style="font-size:28px;font-weight:700;">{esc(item.get("value", "N/A"))}</div>
<div style="margin-top:8px;">{pill(item.get("basis"))}</div></div>""", unsafe_allow_html=True)
            if item.get("how_to_verify"):
                st.caption(f"Verify: {item.get('how_to_verify')}")
    st.markdown("### Key Market Trends")
    st.markdown("".join(claim_line(t) for t in m.get("trends", [])), unsafe_allow_html=True)

with tab3:
    competitors = result.get("competitors", [])
    if competitors:
        for c in competitors:
            with st.expander(c.get("name", "Competitor")):
                st.markdown(pill(c.get("basis")), unsafe_allow_html=True)
                st.write(f"**Strengths:** {c.get('strengths', '')}")
                st.write(f"**Weaknesses:** {c.get('weaknesses', '')}")
    else:
        st.info("No competitor data available.")

with tab4:
    p = result.get("user_pain_points", {})
    legend()
    st.write("**Jobs-to-be-done:**")
    st.markdown("".join(claim_line(j) for j in p.get("jtbd", [])), unsafe_allow_html=True)
    st.write("**Frustrations with existing solutions:**")
    st.markdown("".join(claim_line(f) for f in p.get("frustrations", [])), unsafe_allow_html=True)

with tab5:
    b = result.get("business_model", {})
    st.markdown("### Business Model Canvas")
    legend()
    cell_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#fff;height:100%;box-sizing:border-box;"
    label_style = "font-size:11px;font-weight:600;color:#666;margin-bottom:6px;"
    val_style = "font-size:12px;color:#222;line-height:1.5;overflow-wrap:break-word;word-break:break-word;"
    bot_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#f8f8f8;box-sizing:border-box;"

    def cell(icon, label, key, style=cell_style):
        v = b.get(key, "—")
        text, basis = (v.get("text", "—"), v.get("basis")) if isinstance(v, dict) else (v, None)
        tag = pill(basis) if basis else ""
        return (f'<div style="{style}"><div style="{label_style}">{icon} {label}</div>'
                f'<div style="{val_style}">{esc(text)}</div><div style="margin-top:6px;">{tag}</div></div>')

    st.markdown(f"""
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
</table>
""", unsafe_allow_html=True)

with tab6:
    conf = str(result.get("confidence", "")).split("|")[0].strip().title()
    try:
        score = int(float(result.get("score", 0)))
    except Exception:
        score = 0
    banner = f"## Confidence: {conf}　　Reference score: {score} / 100"
    if conf == "High":
        st.success(banner)
    elif conf == "Medium":
        st.warning(banner)
    else:
        st.error(banner)
    st.progress(max(0, min(score, 100)) / 100)
    st.caption("Confidence shows how much is already known. The score is a reference only.")
    st.write(result.get("confidence_reason", ""))

    if prev:
        try:
            prev_score = int(float(prev.get("score", 0)))
        except Exception:
            prev_score = 0
        st.info(f"Compared with the previous version: confidence {prev.get('confidence', '?')} → {conf}, "
                f"score {prev_score} → {score}")

    st.markdown("---")
    w = result.get("what_would_change_my_mind", {})
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Would raise confidence:**")
        for x in w.get("raise_confidence", []):
            st.write(f"- {x}")
    with cb:
        st.markdown("**Would lower confidence:**")
        for x in w.get("lower_confidence", []):
            st.write(f"- {x}")
    st.markdown("---")
    st.markdown("### Next Step This Week")
    st.write(result.get("next_step", ""))

    if prev:
        with st.expander("Compare riskiest assumptions with the previous version"):
            pa, pb = st.columns(2)
            with pa:
                st.markdown("**Previous**")
                for x in prev.get("riskiest_assumptions", []):
                    st.write(f"- {x.get('assumption', '')}")
            with pb:
                st.markdown("**Now**")
                for x in result.get("riskiest_assumptions", []):
                    st.write(f"- {x.get('assumption', '')}")

st.divider()
d1, d2, d3 = st.columns([1.5, 1.5, 5])
with d1:
    st.download_button("Download Report (Markdown)", to_markdown(result, last["idea"], last["mode"]),
                       file_name="pm_assumption_review.md", mime="text/markdown")
with d2:
    st.download_button("Download Full Report (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="pm_analysis_report.json", mime="application/json")

fb = st.radio("Will you run one of these tests in the next two weeks?", ["Yes", "Maybe", "No"],
              index=None, horizontal=True, key=f"fb_{len(st.session_state['history'])}")
if fb:
    log_feedback(fb, result.get("one_line_summary", ""))
    st.caption("Thanks — recorded.")