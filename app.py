import streamlit as st
import anthropic
import os
import json
import PyPDF2
import docx

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

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
    "tam": "...",
    "trends": ["...", "..."]
  },
  "competitors": [
    {"name": "...", "strengths": "...", "weaknesses": "..."}
  ],
  "user_pain_points": {
    "jtbd": ["...", "..."],
    "frustrations": ["...", "..."]
  },
  "business_model": {
    "revenue_model": "...",
    "cost_risks": ["...", "..."]
  },
  "go_no_go": {
    "score": 0,
    "verdict": "GO / CONDITIONAL GO / NO-GO",
    "reasons": ["...", "...", "..."]
  }
}"""

    user_prompt = "Analyze this product concept and return JSON only:\n\n" + product_input

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    result_text = response.content[0].text

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
            st.metric("Estimated TAM", m.get("tam", "N/A"))
            st.write("**Key Market Trends:**")
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
            st.info(f"**Recommended Revenue Model:** {b.get('revenue_model', '')}")
            st.write("**Key Cost Drivers & Risks:**")
            for r in b.get("cost_risks", []):
                st.write(f"- {r}")

        with tab5:
            g = result.get("go_no_go", {})
            score = g.get("score", 0)
            verdict = g.get("verdict", "")

            if "NO-GO" in verdict:
                st.error(f"## {verdict}    Score: {score} / 10")
            elif "CONDITIONAL" in verdict:
                st.warning(f"## {verdict}    Score: {score} / 10")
            else:
                st.success(f"## {verdict}    Score: {score} / 10")

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
