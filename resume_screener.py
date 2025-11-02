import json
from typing import Dict, List, Any, TypedDict
import logging
import dotenv
dotenv.load_dotenv()
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
import google.generativeai as genai

from keyword_matcher import match_keywords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScreeningState(TypedDict):
    """State schema for the resume screening workflow"""
    all_keywords: List[str]
    resume_text: str
    job_description_text: str
    matching_keywords: List[str]
    initial_decision: str  
    decision: str
    evaluation_summary: str
    requirements_met: List[str]
    requirements_missing: List[str]
    overall_score: int | None
    error: str

class ResumeScreener:
    """AI-Based Resume Screening Agent"""
    
    def __init__(self, api_key: str):
        """
        Initialize the resume screener with Google AI API key
        
        Args:
            api_key: Google AI API key for Gemini access
        """
        self.api_key = api_key
        self._setup_genai()
        self.graph = self._build_graph()
    
    def _setup_genai(self):
        """Configure Google Generative AI"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Google Generative AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Google AI: {e}")
            raise
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow"""
        workflow = StateGraph(ResumeScreeningState)
        
        workflow.add_node("match_keywords", self.match_keywords_node)
        workflow.add_node("evaluate_candidate", self.evaluate_candidate_node)
        
        # The graph now starts directly at keyword matching
        workflow.add_edge(START, "match_keywords")
        workflow.add_edge("match_keywords", "evaluate_candidate")
        workflow.add_edge("evaluate_candidate", END)
        
        return workflow.compile()
    
    def match_keywords_node(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Node 3: Match extracted keywords against resume text
        
        Args:
            state: Current state with resume text and extracted keywords
            
        Returns:
            Updated state with matching keywords
        """
        try:
            logger.info("Matching keywords against resume...")
            
            if "error" in state and state["error"]:
                return {}
            
            matching_keywords, initial_decision = match_keywords(
                state["resume_text"], 
                state["all_keywords"]
            )
            
            logger.info(f"Found {len(matching_keywords)} matching keywords")
            
            return {
                "matching_keywords": matching_keywords,
                "initial_decision": initial_decision
            }
            
        except Exception as e:
            logger.error(f"Error matching keywords: {e}")
            return {"error": str(e)}
    
    def evaluate_candidate_node(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Node 4: Final evaluation using Gemini to make hiring decision
        
        Args:
            state: Complete state with all information
            
        Returns:
            Final decision and evaluation summary
        """
        try:
            logger.info("Evaluating candidate...")
            
            if "error" in state and state["error"]:
                return {}
            
            print(state['initial_decision'])

            evaluation_prompt = f"""You are an expert-level Senior Technical Recruiter and Hiring Manager. Your task is to perform a definitive, final-stage evaluation of a job candidate.

            You have been provided with a comprehensive dossier containing:
            1.  **The Candidate's Full Resume:** The complete text of their resume.
            2.  **The Full Job Description:** The requirements for the role.
            3.  **An Automated Screening Result:** An initial analysis was performed to see how many of the total required keywords were present in the resume.
                -   Total Keywords Required: {len(state['all_keywords'])}
                -   Keywords Found in Resume: {len(state['matching_keywords'])} ({state['matching_keywords']})
                -   Initial Automated Decision: **{state['initial_decision']}**

            **Your Core Task:**
            Your role is to synthesize this information into a final, expert hiring decision. You must go beyond the automated score. The keyword match is a critical starting point, but your final judgment must be based on a holistic review of the candidate's actual experience and qualifications as detailed in their resume, weighed against the job description.

            **Follow this structured evaluation process:**
            1.  **Acknowledge the Automated Result:** Start by considering the `{state['initial_decision']}` score. A "STRONG" match is a positive signal, while a "WEAK" or "FAIL" match requires you to find very compelling, context-rich evidence in the resume to override it.
            2.  **Conduct a Deep-Dive Analysis:** Compare the candidate's resume directly against the job description. Look for:
                -   **Experience Level Match (Crucial Filter):** First, critically assess if the candidate's stated years of experience and career level (e.g., Senior Developer, Junior, Intern) align with the seniority and requirements of the role (e.g., "Internship," "2-4 years experience," "Senior Lead"). 
                -   **Direct Experience:** Does their work history align with the core responsibilities of the role?
                -   **Technical Proficiency:** Do their listed skills and technologies (e.g., programming languages, frameworks, tools) meet the job's essential requirements?
                -   **Contextual Fit:** Even if a specific keyword is missing, does their project history or role description imply proficiency in that area? (e.g., "led a cloud migration project" implies experience with AWS, GCP, or Azure).
            3.  **Formulate a Final Decision and Rationale:** Based on your complete analysis, make a final call.

            **Final Output Format:**
            Return a single, valid JSON object with exactly four keys:
            {{
            "decision": "Shortlisted" or "Not Shortlisted",
            "evaluation_summary": "Your concise, professional rationale for the decision.This must clearly state why the candidate is or isn't a fit based on resume vs. JD."
            "criteria_breakdown": {{
                "requirements_met": "List of string, containing keywords/requirements from the job description that were clearly satisfied or demonstrated by the candidate.",
                "requirements_missing": "List of string, containing core keywords/requirements from the job description that were not adequately met or evidenced in the resume."
            }}
            "overall_score": "A integer numeric score (0 to 100) representing how well the candidate meets the job requirements. Base this on resume-to-JD alignment, matched/unmatched requirements, and contextual fit (not just keywords)."
            }}

            ---
            **Candidate's Resume:**
            {state['resume_text']}

            ---
            **Job Description:**
            {state['job_description_text']}
            ---
            """

            generation_config = {
            "temperature": 0.4,
            "response_mime_type": "application/json",
            }
            response = self.model.generate_content(evaluation_prompt, generation_config=generation_config)
            
            try:
                evaluation_result = json.loads(response.text)
                decision = evaluation_result.get('decision', 'Error: No Decision Provided')
                evaluation_summary = evaluation_result.get('evaluation_summary', 'Error: No summary provided')

                criteria_breakdown = evaluation_result.get('criteria_breakdown', {})
                requirements_met = criteria_breakdown.get('requirements_met', [])
                requirements_missing = criteria_breakdown.get('requirements_missing', [])

                overall_score = evaluation_result.get('overall_score', None)
                try:
                    overall_score = int(overall_score)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid or missing overall_score: {overall_score}")
                    overall_score = None

            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON response in evaluation, using fallback. Response: {response.text}")
                decision = "Not Shortlisted"
                evaluation_summary = "Failed to get a structured evaluation from the AI model."
                requirements_met = []
                requirements_missing = []
                overall_score = None
            
            return {
                "decision": decision,
                "evaluation_summary": evaluation_summary,
                "requirements_met": requirements_met,
                "requirements_missing": requirements_missing,
                "overall_score": overall_score
            }
            
        except Exception as e:
            logger.error(f"Error evaluating candidate: {e}")
            return {"error": str(e)}

    def screen(self, all_keywords: list[str], resume_text: str, job_description_text: str, resume_filename: str) -> Dict[str, Any]:
        """
        Screens a resume using text content directly without file I/O.
        """
        try:
            initial_state = ResumeScreeningState(
                all_keywords=all_keywords,
                resume_text=resume_text,
                job_description_text=job_description_text,
                matching_keywords=[],
                initial_decision="",
                decision="",
                evaluation_summary="",
                requirements_met=[],
                requirements_missing=[],
                overall_score=None,
                error=""
            )
            
            logger.info(f"Starting screening workflow for {resume_filename}...")
            final_state = self.graph.invoke(initial_state)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            result = {
                "decision": final_state["decision"],
                "matching_keywords": final_state["matching_keywords"],
                "evaluation_summary": final_state["evaluation_summary"],
                "requirements_met": final_state["requirements_met"],
                "requirements_missing": final_state["requirements_missing"],
                "overall_score": final_state["overall_score"],
                "initial_decision": final_state["initial_decision"],
                "total_keywords": len(final_state["all_keywords"]),
                "resume_filename": resume_filename
            }
            return result
            
        except Exception as e:
            logger.error(f"Error in screening workflow for {resume_filename}: {e}")
            return {
                "decision": "Error", "evaluation_summary": str(e), "overall_score": None,
                "matching_keywords": [], "requirements_met": [], "requirements_missing": [],
                "initial_decision": "ERROR", "total_keywords": len(all_keywords), "resume_filename": resume_filename
            }
