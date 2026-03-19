import streamlit as st
from openai import OpenAI
import os
import json
import PyPDF2
import docx

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def analyze_product(product_input):
    system_prompt = """You are a senior product manager and market analyst with 15+ years of experience evaluating early-stage products.
You are critical and rigorous. Most products have significant weaknesses — reflect this honestly in your scores.
A score above 80/100 should be rare and only for truly exceptional concepts.

Return ONLY valid JSON using this exact structure:

{
  "market": {
    "tam": "e.g. $15B",
    "sam": "e.g. $3B",
    "som": "e.g. $300M",
    "trends": ["...", "...", "..."]
  },
  "competitors": [
    {"name": "...", "strengths": "...", "weaknesses": "..."}
  ],
  "user_pain_points": {
    "jtbd": ["...", "...", "..."],
    "frustrations": ["...", "...", "..."]
  },
  "business_model": {
    "key_partners": "...",
    "key_activities": "...",
    "key_resources": "...",
    "value_propositions": "...",
    "customer_relationships": "...",
    "channels": "...",
    "customer_segments": "...",
    "cost_structure": "...",
    "revenue_streams": "..."
  },
  "scoring": {
    "product_market_fit": {
      "score": 0,
      "max": 30,
      "rationale": "...",
      "risks": ["...", "..."]
    },
    "market_size_growth": {
      "score": 0,
      "max": 25,
      "rationale": "...",
      "risks": ["...", "..."]
    },
    "competitive_differentiation": {
      "score": 0,
      "max": 20,
      "rationale": "...",
      "risks": ["...", "..."]
    },
    "business_model_viability": {
      "score": 0,
      "max": 15,
      "rationale": "...",
      "risks": ["...", "..."]
    },
    "go_to_market_feasibility": {
      "score": 0,
      "max": 10,
      "rationale": "...",
      "risks": ["...", "..."]
    }
  },
  "go_no_go": {
    "total_score": 0,
    "verdict": "GO / CONDITIONAL GO / NO-GO",
    "summary": "...",
    "top_risks": ["...", "...", "..."],
    "recommendations": ["...", "...", "..."]
  }
}

Scoring guide:
- product_market_fit: max 30. Below 18 = weak PMF.
- market_size_growth: max 25. Below 15 = too small or shrinking.
- competitive_differentiation: max 20. Below 12 = not differentiated enough.
- business_model_viability: max 15. Below 8 = unclear path to revenue.
- go_to_market_feasibility: max 10. Below 5 = hard to reach users.
- Total verdict: GO = 75+, CONDITIONAL GO = 50-74, NO-GO = below 50."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=3000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Analyze this product concept and return JSON only:\n\n" + product_input}
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

st.set_page_config(page_title="PM Product Analyzer", page_icon="📊", layout="wide")
st.title("📊 PM Product Analyzer")
st.caption("Enter a product concept to generate market research, competitor analysis, and a Go/No-go recommendation.")

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
col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area(
        "Text Description",
        placeholder="Product name: ...\nTarget users: ...\nCore problem: ...\nSolution: ...\nMarket: ...\nMonetization: ...",
        height=220
    )

with col2:
    uploaded_file = st.file_uploader("Or upload a PDF / DOCX", type=["pdf", "docx"])
    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        st.success(f"File read successfully — {len(file_text)} characters extracted.")

final_input = text_input or ""
if uploaded_file and "file_text" in dir():
    final_input += "\n\n" + file_text

if st.button("Analyze Product", type="primary", disabled=not final_input.strip()):
    with st.spinner("Analyzing your product concept... this may take 20-40 seconds."):
        result = analyze_product(final_input)

    if "error" in result:
        st.error("Parsing failed. Raw output:")
        st.code(result["error"])
    else:
        st.success("Analysis complete!")
        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Size", "🏆 Competitors", "👤 User Pain Points", "💰 Business Model", "🚦 Go / No-go"
        ])

        with tab1:
            m = result.get("market", {})
            tam = m.get("tam", "N/A")
            sam = m.get("sam", "N/A")
            som = m.get("som", "N/A")
            st.markdown("### Market Size Overview")
            col_tam, col_sam, col_som = st.columns(3)
            with col_tam:
                st.markdown(f"""<div style="background:#1a6eb5;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
<div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">TAM</div>
<div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Total Addressable Market</div>
<div style="font-size:28px;font-weight:700;">{tam}</div></div>""", unsafe_allow_html=True)
            with col_sam:
                st.markdown(f"""<div style="background:#1d9e75;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
<div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">SAM</div>
<div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Serviceable Addressable Market</div>
<div style="font-size:28px;font-weight:700;">{sam}</div></div>""", unsafe_allow_html=True)
            with col_som:
                st.markdown(f"""<div style="background:#3b8bd4;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
<div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">SOM</div>
<div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Serviceable Obtainable Market</div>
<div style="font-size:28px;font-weight:700;">{som}</div></div>""", unsafe_allow_html=True)
            st.markdown("### Key Market Trends")
            for t in m.get("trends", []):
                st.write(f"- {t}")

        with tab2:
            competitors = result.get("competitors", [])
            if competitors:
                for c in competitors:
                    with st.expander(c.get("name", "Competitor")):
                        st.write(f"**Strengths:** {c.get('strengths', '')}")
                        st.write(f"**Weaknesses:** {c.get('weaknesses', '')}")
            else:
                st.info("No competitor data available.")

        with tab3:
            p = result.get("user_pain_points", {})
            st.write("**Jobs-to-be-done:**")
            for j in p.get("jtbd", []):
                st.write(f"- {j}")
            st.write("**Frustrations with existing solutions:**")
            for f in p.get("frustrations", []):
                st.write(f"- {f}")

        with tab4:
            b = result.get("business_model", {})
            st.markdown("### Business Model Canvas")

            cell_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#fff;height:100%;box-sizing:border-box;"
            label_style = "font-size:11px;font-weight:600;color:#666;margin-bottom:6px;"
            val_style = "font-size:12px;color:#222;line-height:1.5;overflow-wrap:break-word;word-break:break-word;"
            bot_style = "border:0.5px solid #ddd;border-radius:8px;padding:12px;background:#f8f8f8;box-sizing:border-box;"

            def c(icon, label, key, style=cell_style):
                val = b.get(key, "—")
                return f'<div style="{style}"><div style="{label_style}">{icon} {label}</div><div style="{val_style}">{val}</div></div>'

            st.markdown(f"""
<table style="width:100%;border-collapse:separate;border-spacing:6px;table-layout:fixed;">
<tr>
  <td rowspan="2" style="width:20%;vertical-align:top;">{c("🔗","Key Partners","key_partners")}</td>
  <td style="width:20%;vertical-align:top;">{c("✅","Key Activities","key_activities")}</td>
  <td rowspan="2" style="width:20%;vertical-align:top;">{c("🎁","Value Propositions","value_propositions")}</td>
  <td style="width:20%;vertical-align:top;">{c("❤️","Customer Relationships","customer_relationships")}</td>
  <td rowspan="2" style="width:20%;vertical-align:top;">{c("👥","Customer Segments","customer_segments")}</td>
</tr>
<tr>
  <td style="vertical-align:top;">{c("🏗️","Key Resources","key_resources")}</td>
  <td style="vertical-align:top;">{c("🚚","Channels","channels")}</td>
</tr>
<tr>
  <td colspan="2" style="vertical-align:top;">{c("🏷️","Cost Structure","cost_structure", bot_style)}</td>
  <td></td>
  <td colspan="2" style="vertical-align:top;">{c("💵","Revenue Streams","revenue_streams", bot_style)}</td>
</tr>
</table>
""", unsafe_allow_html=True)

        with tab5:
            g = result.get("go_no_go", {})
            scoring = result.get("scoring", {})
            total = g.get("total_score", 0)
            try:
                total = int(total)
            except Exception:
                total = 0
            verdict = g.get("verdict", "")

            if "NO-GO" in verdict:
                st.error(f"## {verdict}　　Total Score: {total} / 100")
            elif "CONDITIONAL" in verdict:
                st.warning(f"## {verdict}　　Total Score: {total} / 100")
            else:
                st.success(f"## {verdict}　　Total Score: {total} / 100")

            st.progress(total / 100)
            st.markdown("---")

            st.markdown("### Scoring Breakdown")

            dimensions = [
                ("Product-market fit", "product_market_fit", 30),
                ("Market size & growth", "market_size_growth", 25),
                ("Competitive differentiation", "competitive_differentiation", 20),
                ("Business model viability", "business_model_viability", 15),
                ("Go-to-market feasibility", "go_to_market_feasibility", 10),
            ]

            for label, key, max_score in dimensions:
                dim = scoring.get(key, {})
                score = dim.get("score", 0)
                try:
                    score = int(score)
                except Exception:
                    score = 0
                rationale = dim.get("rationale", "")
                risks = dim.get("risks", [])
                pct = score / max_score

                if pct >= 0.75:
                    color = "#1d9e75"
                elif pct >= 0.5:
                    color = "#f0a500"
                else:
                    color = "#e03c3c"

                with st.expander(f"{label}　　{score} / {max_score}"):
                    st.markdown(f"""<div style="background:#f5f5f5;border-radius:8px;padding:2px 8px;margin-bottom:10px;">
<div style="background:{color};height:8px;border-radius:6px;width:{int(pct*100)}%;"></div></div>""", unsafe_allow_html=True)
                    st.write(f"**Rationale:** {rationale}")
                    if risks:
                        st.write("**Risks:**")
                        for r in risks:
                            st.write(f"- {r}")

            st.markdown("---")
            st.markdown("### Overall Assessment")
            st.write(g.get("summary", ""))

            st.markdown("**Top Risks:**")
            for r in g.get("top_risks", []):
