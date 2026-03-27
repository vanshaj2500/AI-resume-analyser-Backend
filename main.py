import os
import io
import json
import PyPDF2
from groq import Groq
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Resume Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://ai-resume-analyser-frontend-c2bz.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert resume analyst and career coach with 15+ years of experience in HR, recruiting, and talent acquisition across multiple industries.

Your task is to provide a comprehensive, actionable resume analysis. Analyze the resume and return a structured JSON response with the following exact format:

{
  "overall_score": <integer 0-100>,
  "summary": "<2-3 sentence executive summary of the candidate>",
  "candidate_level": "<Junior | Mid-level | Senior | Executive>",
  "target_roles": ["<role1>", "<role2>", "<role3>"],
  "sections": {
    "contact_info": {
      "score": <integer 0-100>,
      "status": "<Complete | Incomplete | Missing>",
      "feedback": "<specific feedback>"
    },
    "summary_section": {
      "score": <integer 0-100>,
      "status": "<Strong | Adequate | Weak | Missing>",
      "feedback": "<specific feedback>"
    },
    "experience": {
      "score": <integer 0-100>,
      "years_of_experience": <integer or null>,
      "feedback": "<specific feedback>",
      "highlights": ["<highlight1>", "<highlight2>"]
    },
    "education": {
      "score": <integer 0-100>,
      "highest_degree": "<degree or Unknown>",
      "feedback": "<specific feedback>"
    },
    "skills": {
      "score": <integer 0-100>,
      "technical_skills": ["<skill1>", "<skill2>"],
      "soft_skills": ["<skill1>", "<skill2>"],
      "missing_key_skills": ["<skill1>", "<skill2>"],
      "feedback": "<specific feedback>"
    }
  },
  "strengths": [
    {"title": "<strength title>", "description": "<detailed description>"},
    {"title": "<strength title>", "description": "<detailed description>"},
    {"title": "<strength title>", "description": "<detailed description>"}
  ],
  "weaknesses": [
    {"title": "<weakness title>", "description": "<detailed description>"},
    {"title": "<weakness title>", "description": "<detailed description>"}
  ],
  "improvements": [
    {"priority": "High", "area": "<area>", "suggestion": "<specific actionable suggestion>"},
    {"priority": "High", "area": "<area>", "suggestion": "<specific actionable suggestion>"},
    {"priority": "Medium", "area": "<area>", "suggestion": "<specific actionable suggestion>"},
    {"priority": "Low", "area": "<area>", "suggestion": "<specific actionable suggestion>"}
  ],
  "ats_analysis": {
    "score": <integer 0-100>,
    "issues": ["<issue1>", "<issue2>"],
    "recommendations": ["<rec1>", "<rec2>"]
  },
  "keywords": {
    "present": ["<keyword1>", "<keyword2>"],
    "recommended": ["<keyword1>", "<keyword2>"]
  },
  "skill_match": null
}

IMPORTANT: If a job description is provided, replace the "skill_match": null with a fully populated skill_match object using this schema:

  "skill_match": {
    "match_score": <integer 0-100, how well the candidate's skills match the JD>,
    "job_title": "<inferred job title from the JD>",
    "summary": "<2-sentence summary of how well the candidate matches this role>",
    "required_skills": [
      {
        "skill": "<exact skill name from JD>",
        "category": "<Technical | Soft | Domain | Tool | Certification>",
        "importance": "<Must-have | Nice-to-have>",
        "status": "<Matched | Partial | Missing>",
        "evidence": "<quote or paraphrase from resume proving this skill, or empty string if missing>",
        "gap_advice": "<specific advice to close the gap, or empty string if matched>"
      }
    ],
    "matched_count": <integer — count of skills with status Matched>,
    "partial_count": <integer — count of skills with status Partial>,
    "missing_count": <integer — count of skills with status Missing>,
    "top_gaps": ["<most critical missing skill 1>", "<most critical missing skill 2>", "<most critical missing skill 3>"],
    "strengths_for_role": ["<strength directly relevant to this JD>", "<strength>"],
    "hiring_likelihood": "<Low | Medium | High | Very High>",
    "hiring_rationale": "<1-2 sentence honest assessment of likelihood to get an interview>"
  }

Extract EVERY distinct skill, tool, technology, qualification, and competency mentioned in the job description — typically 10-25 items. Be thorough. Be honest about gaps; do not mark skills as Matched unless there is clear evidence in the resume.

Be specific, actionable, and honest. Do not be overly generous with scores. A score of 70+ means the resume is genuinely competitive. Return ONLY valid JSON, no markdown, no extra text."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AI Resume Analyzer"}


@app.post("/analyze")
async def analyze_resume(
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
):
    if not file and not resume_text:
        raise HTTPException(status_code=400, detail="Provide either a file or resume text.")

    text = ""

    if file:
        content = await file.read()
        if file.filename.lower().endswith(".pdf"):
            try:
                text = extract_text_from_pdf(content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file.")
    else:
        text = resume_text

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume content too short.")

    user_message = f"Please analyze the following resume:\n\n{text}"

    if job_description and job_description.strip():
        user_message += f"\n\n--- JOB DESCRIPTION ---\n{job_description.strip()}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])

        parsed = json.loads(result_text)

        return {"status": "complete", "data": parsed}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
