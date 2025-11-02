"""
Streamlit UI for AI-Based Resume Screening System
"""
import streamlit as st
import hashlib
import json
import os
import tempfile
from typing import List, Dict, Any
import pandas as pd
from io import BytesIO

from resume_screener import ResumeScreener
from parser import extract_text_from_pdf, extract_text_from_docx

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "screener" not in st.session_state:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            st.session_state.screener = ResumeScreener(api_key)
        except Exception as e:
            st.session_state.screener = None
            st.error(f"❌ Failed to initialize Gemini model: {e}")
    else:
        st.session_state.screener = None
        st.error("❌ GEMINI_API_KEY not found in environment.")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .instruction-box {
        background-color: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .keyword-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def get_ai_keywords(screener, jd_text):
    """Helper to call Gemini for keyword extraction."""

    prompt = f"""You are an expert technical recruiter and talent analyst with deep industry knowledge.

            Your task is to meticulously analyze the job description below and extract a prioritized list of the most critical keywords that define the core requirements of the role.

            Follow these rules:
            1.  **Prioritize and Extract:** Focus on keywords that are absolutely essential for a candidate to succeed. Extract these as atomic, standalone terms (specific skills, technologies, required years of experience, tools, etc.).
            2.  **Add Limited, High-Value Synonyms:** For a few of the MOST CRITICAL keywords, you may add 1-2 essential synonyms or direct technological equivalents if they are industry-standard. This is vital for not missing a top-tier candidate who uses a slightly different term.
                - **Crucial Rule:** This must be done sparingly. Over 90% of the keywords must come directly from the text. Only add synonyms for concepts central to the job's core function.
                - **Example:** If a job requires "AWS," it is appropriate to add "Azure" or "GCP." If it requires "PyTorch," adding "TensorFlow" is a valuable addition. Do not add generic synonyms for less critical skills.

            **Format as JSON:** Return your response as a single, valid JSON object in the format {{"keywords": ["keyword1", "keyword2", ...]}} and nothing else.
            
            **Job Description:**
            ---
            {jd_text}
            ---
            """

    generation_config = {"temperature": 0.1, "response_mime_type": "application/json"}
    response = screener.model.generate_content(prompt, generation_config=generation_config)
    result_json = json.loads(response.text)
    return result_json.get("keywords", [])

def generate_jd_hash(job_description: str) -> str:
    """Generate a unique hash for job description to cache keywords"""
    return hashlib.md5(job_description.encode()).hexdigest()

def display_keywords(keywords: List[str], title: str):
    """Display keywords as styled tags"""
    if keywords:
        st.markdown(f"**{title}** ({len(keywords)} keywords)")
        keywords_html = "".join([f'<span class="keyword-tag">{keyword}</span>' for keyword in keywords])
        st.markdown(keywords_html, unsafe_allow_html=True)
    else:
        st.markdown(f"**{title}** No keywords available")

def display_result_card(result: Dict[str, Any], resume_name: str):
    """Display screening result in a professional card format"""
    
    # Determine colors based on decision
    if result["decision"] == "Shortlisted":
        decision_color = "#28a745"
        decision_emoji = "✅"
    else:
        decision_color = "#dc3545"
        decision_emoji = "❌"
    
    # Initial decision color
    initial_decision = result.get("initial_decision", "UNKNOWN")
    if initial_decision == "STRONG":
        initial_color = "#28a745"
        initial_emoji = "🟢"
    elif initial_decision == "WEAK":
        initial_color = "#ffc107"
        initial_emoji = "🟡"
    else:
        initial_color = "#dc3545"
        initial_emoji = "🔴"
    
    # Create main result container
    with st.container():
        st.markdown(f"### 📄 {resume_name}")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: {decision_color}15; border-radius: 0.5rem; border: 2px solid {decision_color};">
                <h2 style="color: {decision_color}; margin: 0;">{decision_emoji}</h2>
                <h3 style="color: {decision_color}; margin: 0;">{result["decision"]}</h3>
                <p style="margin: 0; color: #666;">Final Decision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            score = result.get("overall_score", 0) or 0
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; border: 2px solid #dee2e6;">
                <h2 style="color: #495057; margin: 0;">📊</h2>
                <h3 style="color: #495057; margin: 0;">{score}/100</h3>
                <p style="margin: 0; color: #666;">Overall Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            matched_count = len(result.get("matching_keywords", []))
            total_count = result.get("total_keywords", 0)
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: {initial_color}15; border-radius: 0.5rem; border: 2px solid {initial_color};">
                <h2 style="color: {initial_color}; margin: 0;">{initial_emoji}</h2>
                <h3 style="color: {initial_color}; margin: 0;">{matched_count}/{total_count}</h3>
                <p style="margin: 0; color: #666;">Keywords Matched ({initial_decision})</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display matched keywords
        if result.get("matching_keywords"):
            st.markdown("**🎯 Matched Keywords:**")
            keywords_html = "".join([f'<span class="keyword-tag" style="background-color: #d4edda; color: #155724;">{keyword}</span>' 
                                   for keyword in result["matching_keywords"]])
            st.markdown(keywords_html, unsafe_allow_html=True)
        
        # Display evaluation summary
        st.markdown("**📝 Evaluation Summary:**")
        st.markdown(f'<div class="instruction-box">{result.get("evaluation_summary", "No summary available")}</div>', 
                   unsafe_allow_html=True)
        
        # Requirements breakdown in expandable sections
        col_met, col_missing = st.columns(2)
        
        with col_met:
            with st.expander("✅ Requirements Met", expanded=False):
                requirements_met = result.get("requirements_met", [])
                if requirements_met:
                    for req in requirements_met:
                        st.markdown(f"• {req}")
                else:
                    st.markdown("No specific requirements listed as met")
        
        with col_missing:
            with st.expander("❌ Requirements Missing", expanded=False):
                requirements_missing = result.get("requirements_missing", [])
                if requirements_missing:
                    for req in requirements_missing:
                        st.markdown(f"• {req}")
                else:
                    st.markdown("No specific requirements listed as missing")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AI Resume Screening System</h1>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="instruction-box">
        <h3>📋 Instructions</h3>
        <ol>
            <li><strong>Job Description:</strong> Paste the complete job description in the text area below</li>
            <li><strong>Keywords:</strong> Add custom keywords (optional) and generate AI keywords for comprehensive screening</li>
            <li><strong>Resume Upload:</strong> Upload one or multiple resumes (PDF/DOCX format)</li>
            <li><strong>Screen:</strong> Click the screening button to analyze candidates</li>
        </ol>
        <p><strong>Note:</strong> AI keywords are cached per job description to improve efficiency. Generate once and reuse for multiple candidates.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "ai_keywords_cache" not in st.session_state:
        st.session_state.ai_keywords_cache = {}
    if "current_ai_keywords" not in st.session_state:
        st.session_state.current_ai_keywords = []
    if "screener" not in st.session_state:
        st.session_state.screener = None
    # Initialize last_screening_results to ensure it always exists
    if "last_screening_results" not in st.session_state:
        st.session_state.last_screening_results = []
    
    # Sidebar for API configuration
    with st.sidebar:
        st.markdown("### 📊 Screening Statistics")

        if "screening_stats" not in st.session_state:
            st.session_state.screening_stats = {"total_screened": 0, "shortlisted": 0}
        
        st.metric("Total Screened", st.session_state.screening_stats["total_screened"])
        st.metric("Shortlisted", st.session_state.screening_stats["shortlisted"])
        
        if st.session_state.screening_stats["total_screened"] > 0:
            success_rate = (st.session_state.screening_stats["shortlisted"] / 
                            st.session_state.screening_stats["total_screened"]) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Job Description Input
        st.markdown("### 📋 Job Description")
        job_description = st.text_area(
            "Paste the complete job description here:",
            height=200,
            placeholder="Enter the full job description including required skills, experience, and qualifications..."
        )
        
        # User Keywords Input
        st.markdown("### 🏷️ Custom Keywords")
        user_keywords_input = st.text_input(
            "Additional Keywords (comma-separated):",
            placeholder="Python, Machine Learning, AWS, 5+ years experience...",
            help="Add any specific keywords that are crucial for this role"
        )
        
        # Parse user keywords
        user_keywords = []
        if user_keywords_input.strip():
            user_keywords = [kw.strip() for kw in user_keywords_input.split(",") if kw.strip()]
    
    with col_right:
        # Keyword Management Section
        st.markdown("### 🤖 AI Keyword Generation")
        
        if job_description.strip():
            jd_hash = generate_jd_hash(job_description)
            
            # Check if we have cached keywords for this JD
            if jd_hash in st.session_state.ai_keywords_cache:
                st.success("✅ AI keywords already generated for this job description")
                st.session_state.current_ai_keywords = st.session_state.ai_keywords_cache[jd_hash]
            else:
                st.info("💡 Click to generate AI keywords for this job description")
            
            # Generate AI Keywords Button
            if st.button("🚀 Generate AI Keywords", disabled=(st.session_state.screener is None)):
                if st.session_state.screener:
                    with st.spinner("Generating AI keywords..."):
                        try:
                            ai_keywords = get_ai_keywords(st.session_state.screener, job_description)
                            st.session_state.ai_keywords_cache[jd_hash] = ai_keywords
                            st.session_state.current_ai_keywords = ai_keywords
                            st.success(f"✅ Generated {len(ai_keywords)} AI keywords")
                        except Exception as e:
                            st.error(f"❌ Error generating keywords: {str(e)}")
                else:
                    st.error("❌ Please configure your API key first")
        else:
            st.warning("⚠️ Please enter a job description first")
        
        # Display current keywords
        if st.session_state.current_ai_keywords or user_keywords:
            st.markdown("### 📝 Current Keywords")
            
            if user_keywords:
                display_keywords(user_keywords, "🏷️ Custom Keywords")
                st.markdown("<br>", unsafe_allow_html=True)
            
            if st.session_state.current_ai_keywords:
                display_keywords(st.session_state.current_ai_keywords, "🤖 AI Keywords")
            
            # Combined keywords count
            total_keywords = len(user_keywords) + len(st.session_state.current_ai_keywords)
            st.info(f"📊 Total Keywords: {total_keywords}")
    
    # Resume Upload Section
    st.markdown("### 📄 Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload resume files:",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Select one or more resume files in PDF or DOCX format"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully")
        for file in uploaded_files:
            st.markdown(f"• {file.name} ({file.size} bytes)")
    
    # Screen Resumes Button
    st.markdown("---")
    
    can_screen = (
        job_description.strip() and 
        uploaded_files and 
        st.session_state.screener is not None and
        (st.session_state.current_ai_keywords or user_keywords)
    )
    
    if st.button("🔍 Screen Resume(s)", disabled=not can_screen, type="primary"):
        # This block now ONLY handles the screening logic, not the display
        with st.spinner(f"Screening {len(uploaded_files)} resume(s)..."):
            all_keywords = list(set(user_keywords + st.session_state.current_ai_keywords))
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Screening {uploaded_file.name}...")
                
                try:
                    file_extension = uploaded_file.name.lower().split('.')[-1]
                    if file_extension == 'pdf':
                        resume_text = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
                    elif file_extension == 'docx':
                        resume_text = extract_text_from_docx(BytesIO(uploaded_file.getvalue()))
                    else:
                        raise ValueError(f"Unsupported file format: {file_extension}")
                        
                    result = st.session_state.screener.screen(
                        all_keywords=all_keywords,
                        resume_text=resume_text,
                        job_description_text=job_description,
                        resume_filename=uploaded_file.name
                    )
                    results.append(result)
                        
                    st.session_state.screening_stats["total_screened"] += 1
                    if result.get("decision") == "Shortlisted":
                        st.session_state.screening_stats["shortlisted"] += 1
                                        
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    results.append({
                        "decision": "Error", "matching_keywords": [],
                        "evaluation_summary": f"Error processing file: {str(e)}",
                        "requirements_met": [], "requirements_missing": [], "overall_score": None,
                        "initial_decision": "ERROR", "total_keywords": len(all_keywords),
                        "resume_filename": uploaded_file.name
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))

            # Persist the results to session_state
            st.session_state.last_screening_results = results
            
            progress_bar.empty()
            status_text.empty()
            st.success("Screening complete! See results below.")

    if st.session_state.last_screening_results:
        st.markdown("## 📊 Screening Results")
        
        results = st.session_state.last_screening_results
        
        # Summary statistics
        shortlisted_count = sum(1 for r in results if r.get("decision") == "Shortlisted")
        avg_score = sum(r.get("overall_score", 0) or 0 for r in results) / len(results) if results else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            st.metric("Shortlisted", shortlisted_count)
        with col3:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        
        st.markdown("---")

        # EXPORT LOGIC
        # Prepare data for download here, right before rendering the button.
        csv_data = []
        for result in results:
            csv_data.append({
                "Candidate": result.get("resume_filename", "Unknown"),
                "Decision": result.get("decision", "Unknown"),
                "Overall Score": result.get("overall_score", 0) or 0,
                "Keywords Matched": len(result.get("matching_keywords", [])),
                "Total Keywords": result.get("total_keywords", 0),
                "Initial Decision": result.get("initial_decision", "Unknown"),
                "Evaluation Summary": result.get("evaluation_summary", "")
            })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="resume_screening_results.csv",
                mime="text/csv",
                key='download-csv' # Adding a key is good practice
            )
            st.markdown("---")

        # Sort results by score (shortlisted first, then by score)
        results.sort(key=lambda x: (
            0 if x.get("decision") == "Shortlisted" else 1,
            -(x.get("overall_score", 0) or 0)
        ))
        
        # Display individual results
        for result in results:
            display_result_card(result, result.get("resume_filename", "Unknown"))
            st.markdown("---")
if __name__ == "__main__":
    main()