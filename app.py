"""
Product Analyzer v2 — Assumption Stress-Tester
Deploy as a SEPARATE Streamlit app so v1 stays live.

Secrets (Streamlit Cloud > App settings > Secrets):
    DEEPSEEK_API_KEY = "sk-..."
Streamlit Cloud exposes root-level secrets as environment variables, so os.environ works.
"""

import csv, datetime, io, json, os, re
import streamlit as st
from openai import OpenAI

MODEL = "deepseek-chat"
TEMPERATURE = 0.2  # low temperature = more consistent results

# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You are a skeptical senior product manager reviewing an early-stage product idea.
Your job is NOT to decide whether the idea is good. Your job is to find the assumptions the idea
depends on, judge which are most likely to be wrong, and tell the founder the cheapest way to test them.

Rules:
- Be critical. Most early ideas rest on at least one unsupported assumption. Say so plainly.
- You have no live market data. Never present a market size or statistic as fact. If you mention a
  number, label its basis as "Assumption" or "Inference" and say what source would confirm it.
- Label every claim's basis:
    "Evidence"   = stated in the user's input or widely documented public knowledge
    "Inference"  = reasoned from evidence, could be wrong
    "Assumption" = no evidence; must be tested
- Choose exactly 3 riskiest assumptions, ordered from most to least dangerous. "Dangerous" means:
  if this is wrong, the idea fails, and there is little evidence for it yet.
- Each test must be cheap (doable in under 2 weeks by 1-2 people) and have a clear pass signal.
- Confidence is about how much is already known, not how exciting the idea is:
    "Low" = key assumptions untested; "Medium" = some support; "High" = core assumptions supported.
- The score (0-100) is a secondary reference only. Above 70 should be rare.
{mode_rules}

Return ONLY valid json in exactly this shape:
{{
  "one_line_summary": "what the idea is, in plain words",
  "assumed_target_user": "who you think this is for, as specifically as possible",
  "confidence": "Low | Medium | High",
  "confidence_reason": "one or two sentences",
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
  "dimensions": {{
    "market":         [{{"claim": "...", "basis": "Evidence | Inference | Assumption"}}],
    "competition":    [{{"claim": "...", "basis": "..."}}],
    "user_pain":      [{{"claim": "...", "basis": "..."}}],
    "business_model": [{{"claim": "...", "basis": "..."}}],
    "feasibility":    [{{"claim": "...", "basis": "..."}}]
  }},
  "score": 0,
  "next_step": "the single most useful thing to do this week"
}}"""

B2C_RULES = "- Treat this as a consumer product: the user and the buyer are usually the same person."
B2B_RULES = """- Treat this as a B2B product. Separate the BUYER (who pays and signs) from the USER (who uses it daily).
- At least one riskiest assumption must be about the buyer: budget owner, procurement, switching cost,
  or sales cycle length.
- Consider compliance, security review, and integration with existing systems under feasibility."""

DIM_LABELS = {"market": "Market", "competition": "Competition", "user_pain": "User pain",
              "business_model": "Business model", "feasibility": "Feasibility"}
BASIS_ICON = {"Evidence": "🟢", "Inference": "🟡", "Assumption": "🔴"}

# ---------------------------------------------------------------- helpers

def read_upload(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")

def analyze(idea, mode):
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    system = SYSTEM_PROMPT.format(mode_rules=B2B_RULES if mode == "B2B" else B2C_RULES)
    resp = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=4000,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Product idea:\n{idea}"}])
    raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
    return json.loads(raw)

def basis_tag(b):
    b = (b or "").split("|")[0].strip()
    return f"{BASIS_ICON.get(b, '⚪')} {b or 'Unlabeled'}"

def to_markdown(r, idea, mode):
    out = [f"# Assumption review: {r.get('one_line_summary','')}", "",
           f"Mode: {mode} | Confidence: {r.get('confidence')} | Reference score: {r.get('score')}/100", "",
           f"**Target user:** {r.get('assumed_target_user','')}", "",
           f"**Why this confidence:** {r.get('confidence_reason','')}", "", "## Riskiest assumptions"]
    for i, a in enumerate(r.get("riskiest_assumptions", []), 1):
        out += [f"### {i}. {a.get('assumption')}", f"- Why risky: {a.get('why_risky')}",
                f"- Basis: {a.get('basis')}", f"- Cheapest test ({a.get('test_type')}): {a.get('cheapest_test')}",
                f"- Pass signal: {a.get('pass_signal')}", ""]
    w = r.get("what_would_change_my_mind", {})
    out += ["## What would change this", "**Raise:** " + "; ".join(w.get("raise_confidence", [])),
            "", "**Lower:** " + "; ".join(w.get("lower_confidence", [])), "",
            f"## Next step\n{r.get('next_step','')}", "", "---", "Original input:", "", idea]
    return "\n".join(out)

def log_feedback(answer, summary):
    # Ephemeral on Streamlit Cloud (resets on restart). Swap for Google Sheets when you run a real test.
    with open("feedback.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.datetime.now().isoformat(), answer, summary])

# ---------------------------------------------------------------- UI

st.set_page_config(page_title="Product Analyzer", page_icon="🧪", layout="centered")
st.title("Product Analyzer")
st.write("Find the assumptions your idea depends on, and the cheapest way to test each one.")

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {idea, mode, result}

last = st.session_state.history[-1] if st.session_state.history else None

mode = st.radio("Who pays for it?", ["B2C", "B2B"], horizontal=True,
                captions=["The user pays", "A business pays, its people use it"],
                index=0 if not last else ["B2C", "B2B"].index(last["mode"]))
with st.expander("Suggested input format"):
    st.markdown("""
```
Product name: [name]
Target users: [who they are, age range, role]
Core problem: [the pain point you're solving]
Solution: [how the product solves it]
Market: [market or industry]
Monetization: [how it makes money]
Stage: [idea / MVP / prototype]
```
""")
upload = st.file_uploader("Upload a PRD or pitch doc (optional)", type=["pdf", "docx", "txt", "md"])
idea = st.text_area("Describe the idea", height=160,
                    value=last["idea"] if last else "",
                    placeholder="What it is, who it's for, what problem it solves, and how it makes money.")

label = "Re-run with my changes" if last else "Find the riskiest assumptions"
if st.button(label, type="primary"):
    text = (read_upload(upload) + "\n\n" + idea) if upload else idea
    if len(text.strip()) < 40:
        st.error("Add a bit more detail: who it's for and what problem it solves.")
    else:
        with st.spinner("Reviewing assumptions…"):
            try:
                st.session_state.history.append({"idea": text, "mode": mode, "result": analyze(text, mode)})
                st.rerun()
            except json.JSONDecodeError:
                st.error("The model returned an unreadable response. Run it again.")
            except Exception as e:
                st.error(f"Request failed: {e}")

if not last:
    st.stop()

r = last["result"]
prev = st.session_state.history[-2]["result"] if len(st.session_state.history) > 1 else None

st.divider()
st.subheader(r.get("one_line_summary", ""))
c1, c2, c3 = st.columns(3)
c1.metric("Confidence", r.get("confidence", "?"),
          delta=f"was {prev.get('confidence')}" if prev and prev.get("confidence") != r.get("confidence") else None,
          delta_color="off")
c2.metric("Reference score", r.get("score", "?"),
          delta=(r.get("score", 0) - prev.get("score", 0)) if prev else None)
c3.metric("Version", len(st.session_state.history))
st.caption(f"Assumed target user: {r.get('assumed_target_user','')}")
st.write(r.get("confidence_reason", ""))

st.markdown("### The three riskiest assumptions")
for i, a in enumerate(r.get("riskiest_assumptions", []), 1):
    with st.container(border=True):
        st.markdown(f"**{i}. {a.get('assumption')}**  \n{basis_tag(a.get('basis'))}")
        st.write(f"If wrong: {a.get('why_risky')}")
        st.markdown(f"**Test it ({a.get('test_type')}):** {a.get('cheapest_test')}  \n"
                    f"**It holds if:** {a.get('pass_signal')}")

w = r.get("what_would_change_my_mind", {})
col_a, col_b = st.columns(2)
col_a.markdown("**Would raise confidence**\n\n" + "\n".join(f"- {x}" for x in w.get("raise_confidence", [])))
col_b.markdown("**Would lower confidence**\n\n" + "\n".join(f"- {x}" for x in w.get("lower_confidence", [])))

st.info(f"**This week:** {r.get('next_step','')}")

with st.expander("Full analysis by dimension"):
    st.caption("🟢 Evidence  🟡 Inference  🔴 Assumption")
    tabs = st.tabs(list(DIM_LABELS.values()))
    for tab, key in zip(tabs, DIM_LABELS):
        with tab:
            for c in r.get("dimensions", {}).get(key, []):
                st.markdown(f"{basis_tag(c.get('basis'))} — {c.get('claim')}")

if prev:
    with st.expander("Compare with previous version"):
        a, b = st.columns(2)
        a.markdown("**Before**\n\n" + "\n".join(f"- {x.get('assumption')}" for x in prev.get("riskiest_assumptions", [])))
        b.markdown("**Now**\n\n" + "\n".join(f"- {x.get('assumption')}" for x in r.get("riskiest_assumptions", [])))

st.divider()
st.download_button("Download as Markdown", to_markdown(r, last["idea"], last["mode"]),
                   file_name="assumption_review.md")

fb = st.radio("Will you run one of these tests in the next two weeks?",
              ["Yes", "Maybe", "No"], index=None, horizontal=True)
if fb:
    log_feedback(fb, r.get("one_line_summary", ""))
    st.caption("Thanks. Recorded.")

if st.button("Start a new idea"):
    st.session_state.history = []
    st.rerun()