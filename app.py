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
    system_prompt = """You are a senior product manager and market analyst.
When given a product concept, analyze it across 5 dimensions.
Always return ONLY valid JSON, no explanation outside the JSON.
Use this exact structure:
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
  "go_no_go": {
    "score": 7,
    "verdict": "GO / CONDITIONAL GO / NO-GO",
    "reasons": ["...", "...", "..."]
  }
}
Important: score must be an integer between 1 and 10."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=2000,
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

st.subheader("Enter Product Information")
col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area(
        "Text Description",
        placeholder="Example: An AI tool that helps PMs quickly conduct competitor analysis. Target users are PMs at tech companies who spend too much time gathering market data.",
        height=200
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
                st.markdown(
                    f"""
                    <div style="background:#1a6eb5;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
                        <div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">TAM</div>
                        <div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Total Addressable Market</div>
                        <div style="font-size:28px;font-weight:700;">{tam}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
            with col_sam:
                st.markdown(
                    f"""
                    <div style="background:#1d9e75;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
                        <div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">SAM</div>
                        <div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Serviceable Addressable Market</div>
                        <div style="font-size:28px;font-weight:700;">{sam}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
            with col_som:
                st.markdown(
                    f"""
                    <div style="background:#3b8bd4;border-radius:16px;padding:32px 16px;text-align:center;color:white;">
                        <div style="font-size:13px;font-weight:600;letter-spacing:1px;opacity:0.85;">SOM</div>
                        <div style="font-size:11px;opacity:0.7;margin-bottom:8px;">Serviceable Obtainable Market</div>
                        <div style="font-size:28px;font-weight:700;">{som}</div>
                    </div>
                    """, unsafe_allow_html=True
                )

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

            def cell(label, key, bg="var(--color-background-primary)"):
                val = b.get(key, "—")
                return f"""<div style="background:{bg};border:0.5px solid var(--color-border-secondary);border-radius:8px;padding:10px;height:100%;">
<div style="font-size:11px;font-weight:500;color:var(--color-text-secondary);margin-bottom:5px;">{label}</div>
<div style="font-size:12px;color:var(--color-text-primary);line-height:1.5;overflow-wrap:break-word;">{val}</div>
</div>"""

            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;margin-bottom:6px;">
  <div style="grid-row:1/3;">{cell("🔗 Key Partners","key_partners")}</div>
  <div>{cell("✅ Key Activities","key_activities")}</div>
  <div style="grid-row:1/3;">{cell("🎁 Value Propositions","value_propositions")}</div>
  <div>{cell("❤️ Customer Relationships","customer_relationships")}</div>
  <div style="grid-row:1/3;">{cell("👥 Customer Segments","customer_segments")}</div>
  <div>{cell("🏗️ Key Resources","key_resources")}</div>
  <div>{cell("🚚 Channels","channels")}</div>
</div>
<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;">
  {cell("🏷️ Cost Structure","cost_structure","var(--color-background-secondary)")}
  {cell("💵 Revenue Streams","revenue_streams","var(--color-background-secondary)")}
</div>
""", unsafe_allow_html=True)

        with tab5:
            g = result.get("go_no_go", {})
            score = g.get("score", 0)
            try:
                score = int(score)
            except Exception:
                score = 0
            verdict = g.get("verdict", "")

            if "NO-GO" in verdict:
                st.error(f"## {verdict}    Score: {score} / 10")
            elif "CONDITIONAL" in verdict:
                st.warning(f"## {verdict}    Score: {score} / 10")
            else:
                st.success(f"## {verdict}    Score: {score} / 10")

            st.progress(score / 10)

            st.write("**Key Reasons:**")
            for r in g.get("reasons", []):
                st.write(f"- {r}")

        st.divider()
        st.download_button(
            label="Download Full Report (JSON)",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="pm_analysis_report.json",
            mime="application/json"
        )
