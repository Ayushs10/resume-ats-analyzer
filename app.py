from dotenv import load_dotenv
load_dotenv()

import base64
import io
import os
import re
from collections import Counter

import google.generativeai as genai
import pandas as pd
import pdf2image
import plotly.express as px
from PIL import Image  # noqa: F401
from PyPDF2 import PdfReader
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# ---- Streamlit page config (must be the first Streamlit call) ----
st.set_page_config(page_title="ATS Resume Expert", layout="wide", page_icon="📄")


st.markdown("""
<style>
/* ====== App background ====== */
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp, .main {
  background: #f4f6f9 !important;
}

/* ====== Main content card ====== */
.block-container {
  max-width: 1300px !important;
  margin: 2rem auto;
  padding: 2rem 2.5rem 2.5rem;
  background: #ffffff !important;
  color: #1f2937 !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(2, 6, 23, 0.06);
  box-sizing: border-box;
}
.main .block-container { max-width: inherit !important; }

/* ====== Typography ====== */
h1, h2, h3, h4, h5, h6 {
  color: #111827 !important;
  font-weight: 700 !important;
  letter-spacing: .3px;
}
p, label, .stMarkdown, .stText {
  color: #374151 !important;
  font-size: 0.96rem !important;
}

/* ====== Inputs ====== */
textarea, input, select {
  background-color: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #d1d5db !important;
  border-radius: 10px !important;
  padding: .6rem .8rem !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04) inset !important;
  font-size: 0.95rem !important;
}
textarea:focus, input:focus, select:focus {
  outline: none !important;
  border-color: #3b82f6 !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}

/* ====== File uploader ====== */
[data-testid="stFileUploader"] {
  background: #f9fafb !important;
  border: 1px dashed #cbd5e1 !important;
  border-radius: 12px !important;
  padding: .9rem !important;
}

/* ====== Primary buttons ====== */
.stButton > button {
  background: linear-gradient(135deg, #ACDDDE, #74C3C5) !important;
  color: #003344 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: .65rem 1.25rem !important;
  font-weight: 600 !important;
  letter-spacing: .25px;
  font-size: 1rem !important;
  box-shadow: 0 4px 10px rgba(116, 195, 197, 0.25) !important;
  transition: transform .15s ease, box-shadow .2s ease, background .2s ease !important;
  cursor: pointer;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #74C3C5, #4BA7A9) !important;
  color: #002b36 !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 14px rgba(116, 195, 197, 0.28) !important;
}
.stButton > button:active {
  transform: translateY(0);
  box-shadow: 0 4px 10px rgba(116, 195, 197, 0.22) !important;
}

/* ====== Plotly figures ====== */
.stPlotlyChart {
  background: #ffffff !important;
  border-radius: 12px !important;
  padding: 10px !important;
  box-shadow: 0 4px 14px rgba(0,0,0,.05) !important;
  border: 1px solid rgba(2,6,23,.06) !important;
}

/* ====== Metrics ====== */
[data-testid="stMetricValue"] { color: #111827 !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; }

/* ====== Links ====== */
a, a:visited { color: #2563eb !important; text-decoration: none !important; font-weight: 500 !important; }
a:hover { text-decoration: underline !important; }
</style>
""", unsafe_allow_html=True)

# ===================== CONFIG =====================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
PASTEL = ["#93c5fd", "#a5f3fc", "#fde68a", "#c4b5fd", "#f9a8d4"]

# ===================== GEMINI HELPERS =====================
def get_gemini_response(instruction, pdf_content, jd_text):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    resp = model.generate_content([instruction, pdf_content[0], jd_text])
    return resp.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    images = pdf2image.convert_from_bytes(uploaded_file.read())
    first_page = images[0]
    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return [{
        "mime_type": "image/jpeg",
        "data": base64.b64encode(img_byte_arr).decode()
    }]

# ===================== ATS HELPERS =====================
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""

def normalize_words(text: str):
    words = re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}", text.lower())
    stop = set("""a an the and or of to in on for with by from as at is are be been was were it this that those these your you we us they them our their his her its""".split())
    return [w for w in words if w not in stop]

def build_keyword_set(job_desc: str, top_k: int = 40):
    toks = normalize_words(job_desc)
    common = [w for w, _ in Counter(toks).most_common(200)]
    keep = [w for w in common if len(w) >= 3]
    return list(dict.fromkeys(keep))[:top_k]

def coverage_score(resume_text: str, job_desc: str, strong_thr=85, weak_thr=60):
    kws = build_keyword_set(job_desc)
    present, weak, missing = [], [], []
    for kw in kws:
        match = process.extractOne(kw, [resume_text], scorer=fuzz.token_set_ratio)
        score = match[1] if match else 0
        if score >= strong_thr: present.append((kw, score))
        elif score >= weak_thr: weak.append((kw, score))
        else: missing.append(kw)
    cov = (len(present) + 0.5 * len(weak)) / max(len(kws), 1)
    return cov, present, weak, missing, kws

SECTION_PATTERNS = {
    "Contact": r"(email|phone|linkedin|github)",
    "Summary": r"(summary|objective|profile)",
    "Skills": r"(skills|technologies|tooling|tech stack)",
    "Experience": r"(experience|employment|work history|projects)",
    "Education": r"(education|b\.?e\.?|btech|m\.?s\.?|b\.?sc\.?|degree|university|college)"
}

def section_completeness(resume_text: str):
    t = resume_text.lower()
    hits = {name: (re.search(pat, t) is not None) for name, pat in SECTION_PATTERNS.items()}
    score = sum(hits.values()) / len(SECTION_PATTERNS)
    return score, hits

_SBERT = None
def semantic_similarity(resume_text: str, job_desc: str):
    global _SBERT
    if _SBERT is None:
        _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    e1, e2 = _SBERT.encode([resume_text, job_desc], convert_to_tensor=True)
    sim = float(util.cos_sim(e1, e2).item())
    sim = max(0.0, min(1.0, (sim + 1) / 2))
    return sim

def readability_length(resume_text: str):
    wc = len(normalize_words(resume_text))
    if wc <= 150: base = 0.2
    elif wc <= 350: base = 0.6
    elif wc <= 1100: base = 1.0
    elif wc <= 1600: base = 0.7
    else: base = 0.5
    return base

def formatting_health(text_ok: bool, pdf_bytes_ok: bool=True, tables_detected: bool=False):
    score = 1.0
    if not text_ok: score -= 0.7
    if tables_detected: score -= 0.2
    if not pdf_bytes_ok: score -= 0.1
    return max(0.0, min(1.0, score))

WEIGHTS = {
    "keyword": 0.40,
    "sections": 0.25,
    "semantic": 0.20,
    "readability": 0.10,
    "format": 0.05
}

def compute_ats(resume_text: str, job_desc: str):
    cov, present, weak, missing, kws = coverage_score(resume_text, job_desc)
    sec_score, sec_hits = section_completeness(resume_text)
    sim = semantic_similarity(resume_text, job_desc)
    read = readability_length(resume_text)
    fmt = formatting_health(text_ok=bool(resume_text.strip()))
    subscores = {
        "Keyword coverage": round(cov * 100, 1),
        "Section completeness": round(sec_score * 100, 1),
        "Semantic similarity": round(sim * 100, 1),
        "Readability/Length": round(read * 100, 1),
        "Formatting health": round(fmt * 100, 1),
    }
    overall = round(
        WEIGHTS["keyword"] * subscores["Keyword coverage"] +
        WEIGHTS["sections"] * subscores["Section completeness"] +
        WEIGHTS["semantic"] * subscores["Semantic similarity"] +
        WEIGHTS["readability"] * subscores["Readability/Length"] +
        WEIGHTS["format"] * subscores["Formatting health"], 1
    )
    debug = {"present": present, "weak": weak, "missing": missing, "all_kws": kws, "sections": sec_hits}
    return overall, subscores, debug

def donut_and_bars(overall: float, subscores: dict):
    labels = list(subscores.keys())
    values = list(subscores.values())

    donut = px.pie(
        names=labels,
        values=values,
        hole=0.55,
        template="plotly_white",
        width=900, height=420
    )
    donut.update_traces(
        textinfo="percent+label",
        textposition="inside",
        insidetextorientation="radial",
        marker=dict(colors=PASTEL, line=dict(color="white", width=2))
    )
    donut.update_layout(
        title=f"ATS Score: {overall} / 100",
        legend=dict(orientation="v", y=0.5, x=1.03),
        margin=dict(l=20, r=120, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    df = pd.DataFrame({"Metric": labels, "Score": values})
    bars = px.bar(
        df, x="Metric", y="Score",
        range_y=[0, 100],
        template="plotly_white",
        width=900, height=350
    )
    bars.update_traces(
        marker_color=PASTEL,
        marker_line_color="white",
        marker_line_width=1.6
    )
    bars.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="rgba(15,23,42,.06)")
    )
    return donut, bars

# ===================== DASHBOARD RENDERER =====================
def render_ats_dashboard(overall: float, subscores: dict, debug: dict):
    st.markdown("## ATS Score Overview")
    c1, c2 = st.columns([1.4, 1])

    donut_fig, bars_fig = donut_and_bars(overall, subscores)

    with c1:
        st.plotly_chart(donut_fig, use_container_width=True)

    with c2:
        safe_overall = max(0, min(int(round(overall)), 100))
        st.metric("Overall ATS", f"{overall}/100")
        st.progress(safe_overall)
        sections_found = [k for k, v in (debug.get("sections") or {}).items() if v]
        st.markdown("**Sections found:** " + (", ".join(sections_found) or "_none_"))
        missing = debug.get("missing", [])[:12]
        st.markdown("**Top missing keywords:** " + (", ".join(missing) or "_none_"))

    st.plotly_chart(bars_fig, use_container_width=True)

# ===================== BULLET REWRITE HELPERS =====================
def _normalize_words(text: str):
    return re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}", text.lower())

def _build_keyword_set(job_desc: str, top_k: int = 40):
    toks = _normalize_words(job_desc)
    common = [w for w, _ in Counter(toks).most_common(200)]
    keep = [w for w in common if len(w) >= 3]
    seen, out = set(), []
    for w in keep:
        if w not in seen:
            out.append(w); seen.add(w)
    return out[:top_k]

def quick_missing_keywords(resume_text: str, jd_text: str, strong_thr=85, weak_thr=60):
    jd_kws = _build_keyword_set(jd_text)
    present, weak, missing = [], [], []
    for kw in jd_kws:
        match = process.extractOne(kw, [resume_text], scorer=fuzz.token_set_ratio)
        score = match[1] if match else 0
        if score >= strong_thr: present.append((kw, score))
        elif score >= weak_thr: weak.append((kw, score))
        else: missing.append(kw)
    return present, weak, missing, jd_kws

def rewrite_bullets_gemini(resume_text: str, jd_text: str, target_keywords, role_title: str = "Data Role"):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    prompt = f"""
You are a senior resume editor. Improve bullets to align with the job description while staying 100% truthful.

Job Title: {role_title}
Job Description (condensed):
{jd_text[:1500]}

Target keywords to incorporate ONLY if actually supported by the resume content:
{", ".join(target_keywords[:18])}

Resume content (raw text sample for context; do not copy phrasing verbatim):
{resume_text[:3500]}

Rewrite up to 6 bullets that:
- start with a strong verb,
- include concrete impact and realistic numbers (use ranges if the exact value isn't known),
- weave in relevant JD keywords naturally,
- keep past tense, concise (one line each).

If a keyword is truly missing (not supported), do NOT invent a bullet. Instead, list a short "Action to add" item telling the user what to do.

Output strictly in this Markdown structure:

### Improved bullets
- ...

### Actions to add (if any)
- ...
"""
    resp = model.generate_content(prompt)
    return resp.text

# ===================== APP =====================
st.header("ATS Tracking System")

# Job Description with clear instructions
input_text = st.text_area(
    "Job Description:",
    key="input",
    placeholder=(
        "Paste the full job description here (roles, responsibilities, qualifications).\n"
        "Exclude 'About Us/Company', salary/benefits, and diversity sections.\n"
    ),
    height=200
)
st.caption(
    "Tip: Include **Responsibilities**, **Requirements/Qualifications**, and **Preferred Skills**. "
    "Skip company boilerplate so the scan focuses on hard skills."
)

# Resume uploader (with a small help hint)
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)...",
    type=["pdf"],
    help="Text-based PDFs work best. If it's a scan/image, extraction quality may drop."
)

if uploaded_file is not None:
    st.success("✅ PDF Uploaded Successfully")

# 3-button action row (horizontal)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    submit_summary = st.button("Tell Me About the Resume", use_container_width=True)
with c2:
    submit_match = st.button("Percentage match", use_container_width=True)
with c3:
    submit_rewrite = st.button("Improve bullets (AI)", use_container_width=True)

prompt_summary = """
You are an experienced Technical Human Resource Manager. 
Carefully review the candidate’s resume in relation to the given job description. 
Provide a clear, structured evaluation covering:
1. Overall alignment with the role (high-level fit).
2. Key strengths that match the requirements.
3. Specific weaknesses or gaps that could hinder selection.
4. Recommendation: Does this resume merit moving forward to the interview stage? Why or why not?
Keep the tone professional, concise, and tailored for a recruiter’s perspective.
"""
prompt_match = """
You are an advanced ATS (Applicant Tracking System) scanner.
Compare the resume with the job description and return results in the following structured format:
1. Percentage Match (0–100): A realistic estimate of how well the resume aligns.
2. Missing Keywords: List only the most important missing skills, tools, or requirements.
3. Final Thoughts: Brief actionable advice for the candidate to improve ATS compatibility.
Keep your response structured and easy to read, like an ATS report.
"""


# ---- Summary flow
if submit_summary:
    if uploaded_file is None:
        st.write("Please upload the resume")
    else:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(prompt_summary, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)

# ---- Percentage match + dashboard
elif submit_match:
    if uploaded_file is None:
        st.write("Please upload the resume")
    else:
        # Gemini's narrative
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(prompt_match, pdf_content, input_text)

        # Your computed ATS score
        resume_text = extract_text_from_pdf(uploaded_file)
        overall, subscores, debug = compute_ats(resume_text, input_text)

        # Replace any 'Percentage Match:' in Gemini text with your computed score
        response = re.sub(
            r'(?i)percentage\s*match\s*:\s*\d+(\.\d+)?%?',
            f'Percentage Match (ATS): {overall}%',
            response
        )

        st.subheader("Resume Evaluation")
        st.write(response)

        # Dashboard with the same overall
        render_ats_dashboard(overall, subscores, debug)

# ---- Bullet rewrite
if submit_rewrite:
    if uploaded_file is None:
        st.write("Please upload the resume")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)
        _, _, missing, _ = quick_missing_keywords(resume_text, input_text)

        suggestions_md = rewrite_bullets_gemini(
            resume_text=resume_text,
            jd_text=input_text,
            target_keywords=missing[:15],
            role_title="Data Analyst / Engineer / BA"
        )

        st.subheader("AI-Tailored Bullet Suggestions")
        st.caption("Aligned to the JD and your real content—no invented claims.")
        st.markdown(suggestions_md)

        st.markdown("**Top missing keywords:** " + (", ".join(missing[:12]) or "_none_"))
        st.download_button(
            "Download suggestions (.md)",
            suggestions_md.encode("utf-8"),
            file_name="bullet_suggestions.md"
        )
