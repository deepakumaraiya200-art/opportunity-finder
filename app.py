"""
InternFinder — AI-Powered Internship Matching
Flask backend with Gemini AI + Real Job API integration
"""

import os
import json
import re
import secrets
import time
import uuid
import urllib.parse

from flask import Flask, render_template, request, redirect, url_for, session, flash
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from groq import Groq
from job_fetcher import fetch_all_jobs
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# ---- Flask App Setup ----
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configure AI Providers ----
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle resume upload, extract text, fetch real jobs, score with AI."""

    # --- Validate file ---
    if 'resume' not in request.files:
        flash('No file uploaded. Please select your resume.')
        return redirect(url_for('index'))

    file = request.files['resume']

    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.pdf'):
        flash('Only PDF files are accepted.')
        return redirect(url_for('index'))

    # --- Extract text from PDF ---
    try:
        reader = PdfReader(file)
        resume_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + '\n'
    except Exception as e:
        flash(f'Error reading PDF: {str(e)}')
        return redirect(url_for('index'))

    if not resume_text.strip():
        flash('Could not extract text from the PDF. Please try a different file.')
        return redirect(url_for('index'))

    # --- Get user's desired job count ---
    job_count = int(request.form.get('job_count', 20))
    job_count = max(10, min(50, job_count))

    # --- Call AI Pipeline ---
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        flash('No AI API key configured.')
        return redirect(url_for('index'))

    try:
        result = analyze_resume_hybrid(resume_text, job_count)
    except ValueError as e:
        if str(e) == "NO_SKILLS_FOUND":
            flash("We couldn't detect any professional skills in this document. Please ensure you are uploading a valid resume.")
            return redirect(url_for('index'))
        print(f"[Upload] analyze_resume_hybrid Value Error: {e}")
        raise e
    except Exception as e:
        print(f"[Upload] analyze_resume_hybrid failed: {e}")
        # Ultimate fallback: still fetch real jobs from APIs
        fallback_skills = ["Python", "JavaScript", "SQL", "React", "Data Analysis"]
        try:
            fallback_jobs_raw = fetch_all_jobs(fallback_skills, max_per_source=15)
        except Exception:
            fallback_jobs_raw = []
        result = {
            "skills": fallback_skills,
            "action_plan": "<strong>Tip:</strong> The AI scoring system is currently at capacity. Here are live internships we found for you — try again in a few minutes for personalized scoring!",
            "jobs": []
        }
        for job in fallback_jobs_raw[:20]:
            result["jobs"].append({
                "company": job.get("company", "Unknown"),
                "role": job.get("role", "Internship"),
                "tier": "A",
                "match_percentage": 75,
                "matched_skills": job.get("tags", [])[:2],
                "missing_skills": [],
                "location": job.get("location", "India"),
                "duration": "Varies",
                "stipend": job.get("salary", "Competitive"),
                "work_mode": "Remote" if job.get("remote") else "On-site",
                "deadline_text": "\ud83d\udfe2 Open Now",
                "deadline_status": "live",
                "source_text": job.get("source_text", "\u2705 Verified Live Listing"),
                "category": "startup",
                "apply_url": job.get("apply_url", "#"),
                "verified": True
            })

    # Sort: FAANG (tier S) first, then by match percentage, trim to count
    if 'jobs' in result and result['jobs']:
        def sort_key(j):
            tier_order = {'S': 0, 'A': 1, 'B': 2}
            return (tier_order.get(j.get('tier', 'B'), 2), -j.get('match_percentage', 0))
        result['jobs'] = sorted(result['jobs'], key=sort_key)
        result['jobs'] = result['jobs'][:job_count]

    # --- Store results in a server-side file (avoids cookie size limit) ---
    result_id = str(uuid.uuid4())
    result_path = os.path.join(RESULTS_DIR, f'{result_id}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f)
    session['result_id'] = result_id

    return redirect(url_for('dashboard', id=result_id))


@app.route('/dashboard')
def dashboard():
    """Render the results dashboard."""
    result_id = request.args.get('id') or session.get('result_id')
    if not result_id:
        flash('No results found. Please upload your resume first.')
        return redirect(url_for('index'))

    result_path = os.path.join(RESULTS_DIR, f'{result_id}.json')
    if not os.path.exists(result_path):
        flash('Results expired. Please upload your resume again.')
        return redirect(url_for('index'))

    with open(result_path, 'r') as f:
        result = json.load(f)

    jobs = result.get('jobs', [])
    skills = result.get('skills', [])
    action_plan = result.get('action_plan', '')

    return render_template('dashboard.html', jobs=jobs, skills=skills, action_plan=action_plan)


# ============================================
# HYBRID AI + REAL JOBS PIPELINE
# ============================================

def analyze_resume_hybrid(resume_text: str, job_count: int = 20) -> dict:
    """
    Full accuracy pipeline (sequential):
    1. AI extracts skills from resume
    2. Fetch real jobs using those EXACT skills (more sources for higher count)
    3. AI scores and enriches the jobs to match requested count
    """
    import time as _time
    t0 = _time.time()

    # --- STEP 1: Extract skills with AI ---
    skills = extract_skills_with_gemini(resume_text)
    print(f"[AI] Extracted skills: {skills} ({_time.time()-t0:.1f}s)")
    
    if not skills:
        raise ValueError("NO_SKILLS_FOUND")

    # --- STEP 2: Fetch real jobs — fetch more if user wants more ---
    # Fetch extra to account for deduplication + verification filtering
    fetch_per_source = max(25, job_count * 2)
    real_jobs_raw = fetch_all_jobs(skills, max_per_source=fetch_per_source)
    print(f"[API] Fetched {len(real_jobs_raw)} real jobs ({_time.time()-t0:.1f}s)")

    # --- STEP 3: Score real jobs + generate AI suggestions ---
    result = score_and_suggest_with_gemini(resume_text, skills, real_jobs_raw, job_count)
    print(f"[Total] Pipeline complete in {_time.time()-t0:.1f}s")

    return result


def extract_skills_with_gemini(resume_text: str) -> list:
    """Extract skills from resume — tries Gemini first, falls back to keyword matching."""

    prompt = f"""Extract the core professional skills and keywords from this resume.
The resume could be from ANY field (Tech, Marketing, Finance, Design, HR, Sales, etc.).
Return ONLY a JSON array of strings, nothing else. Example: ["SEO", "Content Strategy", "Google Analytics"] or ["Financial Modeling", "Excel"] or ["Python", "React"].
Keep it to 10-15 of the most important skills or tools.

RESUME:
{resume_text}"""

    try:
        response = _call_ai(prompt, max_retries=3)
        raw = response.text.strip()
    except Exception as e:
        print(f"[Gemini] Skill extraction failed, using local extractor: {e}")
        return _extract_skills_local(resume_text)

    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        skills = json.loads(raw)
        if isinstance(skills, list) and len(skills) > 0:
            return skills[:15]
    except:
        pass

    # Fallback: extract from brackets
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if result:
                return result[:15]
        except:
            pass

    # If Gemini returned garbage, use local extractor
    return _extract_skills_local(resume_text)


# Known skills for local extraction (Domain Agnostic)
_KNOWN_SKILLS = [
    # Tech
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C", "Go", "Rust", "Swift", "Kotlin",
    "React", "Angular", "Vue", "Next.js", "Node.js", "Express", "Django", "Flask", "Spring",
    "HTML", "CSS", "Tailwind", "SQL", "MySQL", "PostgreSQL", "MongoDB", "AWS", "Azure", "GCP", 
    "Docker", "Kubernetes", "Git", "Linux", "Machine Learning", "Data Science", "Cyber Security",
    
    # Marketing / Sales
    "SEO", "SEM", "Content Marketing", "Digital Marketing", "Social Media", "Google Analytics",
    "Copywriting", "Email Marketing", "CRM", "Salesforce", "B2B Sales", "Lead Generation",
    "Market Research", "Public Relations", "Brand Management", "HubSpot", "Advertising",

    # Finance / Accounting
    "Financial Analysis", "Financial Modeling", "Accounting", "Excel", "VBA", "Auditing",
    "Taxation", "Investment Banking", "Corporate Finance", "Bookkeeping", "QuickBooks",
    "Risk Management", "Valuation", "Private Equity", "Bloomberg",

    # Design / Creative
    "Figma", "UI/UX", "Adobe Photoshop", "Adobe Illustrator", "Graphic Design", "Video Editing",
    "Adobe Premiere", "After Effects", "User Research", "Wireframing", "Prototyping",
    "Typography", "Branding", "Animation", "3D Modeling", "Blender",

    # Business / HR / Operations
    "Project Management", "Agile", "Scrum", "JIRA", "Product Management", "Operations",
    "Human Resources", "Talent Acquisition", "Recruiting", "Employee Relations", 
    "Supply Chain", "Logistics", "Business Analytics", "Strategy", "Consulting",
    "Communication", "Leadership", "Team Management", "Problem Solving"
]


def _extract_skills_local(resume_text: str) -> list:
    """Extract skills from resume using keyword matching — no AI needed."""
    text_lower = resume_text.lower()
    found = []
    for skill in _KNOWN_SKILLS:
        if skill.lower() in text_lower:
            found.append(skill)
    return found[:15]


def score_and_suggest_with_gemini(resume_text: str, skills: list, real_jobs: list, job_count: int = 20) -> dict:
    """
    Send real jobs + resume to AI for scoring.
    Also ask for additional AI-suggested opportunities.
    Generates exactly job_count results with FAANG priority.
    """

    # Format real jobs for the prompt — include more for higher counts
    max_real_in_prompt = min(len(real_jobs), max(25, job_count))
    real_jobs_text = ""
    for i, job in enumerate(real_jobs[:max_real_in_prompt], 1):
        real_jobs_text += f"""
JOB #{i}:
- Company: {job['company']}
- Role: {job['role']}
- Location: {job['location']}
- Source: {job['source']}
- URL: {job['apply_url']}
- Tags: {', '.join(job.get('tags', []))}
- Remote: {job.get('remote', False)}
"""

    prompt = f"""You are an expert personalized recruiter AI. A student from ANY field (Tech, Business, Arts, Marketing, Finance, etc.) has uploaded their resume.

STUDENT'S SKILLS: {json.dumps(skills)}

RESUME TEXT:
---
{resume_text}
---

REAL JOB LISTINGS (from live job boards — these are verified, actual openings):
---
{real_jobs_text if real_jobs_text.strip() else "No real jobs were found from APIs. Generate all suggestions from your knowledge."}
---

YOUR TASKS:
1. Score each REAL job listing above against the student's resume (match_percentage).
2. ALSO suggest 8-10 ADDITIONAL opportunities from your knowledge (mark these as AI-suggested).
   IMPORTANT: AI-suggested jobs MUST be based in India (Bangalore, Mumbai, Delhi, Pune, Hyderabad, etc.) or Remote. Do NOT suggest US/European on-site roles.
3. Provide a personalized action plan.

MANDATORY TIER-S SUGGESTIONS:
- You MUST suggest at least 3-5 top-tier, highly prestigious internships (tier "S") that are perfectly aligned with the student's specific field.
- For Tech: FAANG/Big Tech (Google, Microsoft, Amazon, etc).
- For Finance: Top banks (Goldman Sachs, JPMorgan, Morgan Stanley).
- For Marketing/Business/Design: Top agencies, Fortune 500 FMCGs, or industry leaders.
- These should be real internship/new-grad programs that these top companies typically offer in India.

IMPORTANT: Return ONLY valid JSON, no markdown, no code fences, no explanation.

{{
  "skills": {json.dumps(skills)},
  "action_plan": "HTML string with <strong>, <br>, <a> tags. 3-4 lines of personalized, actionable advice.",
  "jobs": [
    {{
      "company": "Company Name",
      "role": "Role Title · Track",
      "tier": "S or A or B",
      "match_percentage": 85,
      "matched_skills": ["skill1", "skill2"],
      "missing_skills": ["skill3"],
      "location": "City, Country",
      "duration": "3 months",
      "stipend": "₹40K/mo or Competitive",
      "work_mode": "Remote / On-site / Hybrid",
      "deadline_text": "🟢 Open Now",
      "deadline_status": "live",
      "source_text": "✅ RemoteOK (Live Listing) or 🤖 AI Suggested",
      "category": "faang top-tier startup research platform urgent",
      "apply_url": "https://actual-url-from-above.com",
      "verified": true
    }}
  ]
}}

CRITICAL RULES:
- For REAL jobs from the listings above: set verified=true, keep their EXACT apply_url unchanged, and use their source (e.g., "✅ RemoteOK (Live Listing)").
- For AI-SUGGESTED jobs: set verified=false, source_text="🤖 AI Suggested", and set apply_url to EXACTLY "SEARCH" (we will generate the correct URL server-side). Do NOT invent or guess any URL.
- tier: "S" = FAANG/Fortune 500/Top Prestige in their field, "A" = funded startups/unicorns/mid-size, "B" = research/academic/other.
- category: space-separated from "faang", "top-tier", "startup", "research", "platform", "urgent".
- match_percentage: 40-98, be realistic based on skill overlap.
- deadline_status: "live", "soon", or "urgent".
- MUST include at least {max(3, job_count // 4)} tier-S (Top Prestige / FAANG / Fortune 500) jobs in the suggestions.
- You MUST return EXACTLY {job_count} jobs total (real + suggested combined). Not less, not more.
- Put tier-S jobs FIRST, then other jobs sorted by match percentage.
- STRICT INTERNSHIP POLICY: ALL jobs (real and suggested) MUST be Internships, Co-ops, Trainee, or Fresher roles. You MUST REJECT AND IGNORE any full-time or senior/managerial roles.
"""

    try:
        response = _call_ai(prompt, max_retries=2)
        raw_text = response.text.strip()

        # Remove markdown code fences
        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
        raw_text = re.sub(r'\s*```$', '', raw_text)

        # Fix common JSON issues from LLMs (trailing commas, etc.)
        raw_text = re.sub(r',\s*}', '}', raw_text)  # trailing comma before }
        raw_text = re.sub(r',\s*]', ']', raw_text)  # trailing comma before ]

        data = None
        # Try 1: direct parse
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Try 2: extract JSON object with regex
        if not data:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                cleaned = json_match.group()
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        if not data or 'jobs' not in data:
            raise ValueError('AI response could not be parsed.')

    except Exception as e:
        print(f"[AI] Scoring failed, applying smart fallback: {e}")

        # --- SMART FALLBACK: compute real match scores and assign tiers ---
        _FAANG_KEYWORDS = ['google', 'microsoft', 'amazon', 'meta', 'apple', 'netflix',
                           'facebook', 'adobe', 'uber', 'stripe', 'salesforce', 'oracle',
                           'goldman', 'jpmorgan', 'morgan stanley', 'ibm', 'intel', 'nvidia',
                           'samsung', 'flipkart', 'paytm', 'razorpay', 'phonepe', 'cred',
                           'zomato', 'swiggy', 'byju', 'ola', 'meesho', 'groww']

        skills_lower = set(s.lower() for s in skills)

        data = {
            "skills": skills,
            "action_plan": "<strong>Tip:</strong> Your skills match well with current openings! The AI scoring is at capacity right now, but we've computed skill-based match scores for you. Try again in a few minutes for full AI-powered analysis.",
            "jobs": []
        }

        for i, job in enumerate(real_jobs[:job_count]):
            # Compute real match percentage based on skill overlap
            job_text = f"{job.get('role', '')} {' '.join(job.get('tags', []))} {job.get('company', '')}".lower()
            matched = [s for s in skills if s.lower() in job_text]
            match_pct = min(95, max(40, int(len(matched) / max(len(skills), 1) * 100) + 35))

            # Smart tier assignment
            company_lower = job.get('company', '').lower()
            if any(k in company_lower for k in _FAANG_KEYWORDS):
                tier = 'S'
                category = 'faang'
            elif i < 7:
                tier = 'A'
                category = 'startup'
            else:
                tier = 'B'
                category = 'research'

            data["jobs"].append({
                "company": job.get("company", "Unknown"),
                "role": job.get("role", "Internship"),
                "tier": tier,
                "match_percentage": match_pct,
                "matched_skills": matched[:3],
                "missing_skills": [],
                "location": job.get("location", "India"),
                "duration": "Varies",
                "stipend": job.get("salary", "Competitive"),
                "work_mode": "Remote" if job.get("remote") else "On-site",
                "deadline_text": "🟢 Open Now",
                "deadline_status": "live",
                "source_text": job.get("source_text", "✅ Verified Live Listing"),
                "category": category,
                "apply_url": job.get("apply_url", "#"),
                "verified": True
            })

        # Add FAANG suggestions proportionally (at least 3, up to count//4)
        faang_pool = [
            {"company": "Google", "role": "SWE Intern · STEP Program", "location": "Bangalore, India"},
            {"company": "Microsoft", "role": "Software Engineering Intern", "location": "Hyderabad, India"},
            {"company": "Amazon", "role": "SDE Intern", "location": "Bangalore, India"},
            {"company": "Goldman Sachs", "role": "Engineering Intern", "location": "Bangalore, India"},
            {"company": "Adobe", "role": "Software Development Intern", "location": "Noida, India"},
            {"company": "Uber", "role": "Software Engineer Intern", "location": "Bangalore, India"},
            {"company": "Meta", "role": "Software Engineer Intern", "location": "Remote"},
            {"company": "Salesforce", "role": "Software Engineering Intern", "location": "Hyderabad, India"},
            {"company": "Oracle", "role": "Software Developer Intern", "location": "Bangalore, India"},
            {"company": "Atlassian", "role": "SDE Intern", "location": "Remote"},
            {"company": "Netflix", "role": "Software Engineer Intern", "location": "Remote"},
            {"company": "JPMorgan Chase", "role": "Software Engineer Program", "location": "Mumbai, India"}
        ]
        
        num_faang_to_add = max(3, job_count // 4)
        for fs in faang_pool[:num_faang_to_add]:
            search_q = f"{fs['company']} {fs['role']} apply internship 2026"
            data["jobs"].append({
                "company": fs["company"],
                "role": fs["role"],
                "tier": "S",
                "match_percentage": min(95, max(50, int(len(skills) / max(1, 15) * 80) + 10)),
                "matched_skills": skills[:3],
                "missing_skills": [],
                "location": fs["location"],
                "duration": "2-6 months",
                "stipend": "Competitive",
                "work_mode": "Hybrid" if "Remote" not in fs["location"] else "Remote",
                "deadline_text": "🟡 Rolling Applications",
                "deadline_status": "soon",
                "source_text": "🤖 AI Suggested",
                "category": "faang",
                "apply_url": "https://www.google.com/search?q=" + urllib.parse.quote_plus(search_q),
                "verified": False
            })

    # Validate and set defaults for all jobs
    for job in data['jobs']:
        job.setdefault('company', 'Unknown')
        job.setdefault('role', 'Internship')
        job.setdefault('tier', 'B')
        job.setdefault('match_percentage', 50)
        job.setdefault('matched_skills', [])
        job.setdefault('missing_skills', [])
        job.setdefault('location', 'Remote')
        job.setdefault('duration', 'Varies')
        job.setdefault('stipend', 'Competitive')
        job.setdefault('work_mode', 'Remote')
        job.setdefault('deadline_text', '🟢 Open')
        job.setdefault('deadline_status', 'live')
        job.setdefault('verified', False)
        job.setdefault('apply_url', '#')

        # For AI-suggested jobs, generate a Google search link instead of hallucinated URLs
        if not job.get('verified'):
            search_query = f"{job['company']} {job['role']} apply internship 2026"
            job['apply_url'] = 'https://www.google.com/search?q=' + urllib.parse.quote_plus(search_query)

        # Set source text based on verification
        if job.get('verified'):
            job.setdefault('source_text', '✅ Verified Live Listing')
        else:
            job.setdefault('source_text', '🤖 AI Suggested')

        job.setdefault('category', '')

        # Clamp match percentage
        job['match_percentage'] = max(0, min(100, int(job['match_percentage'])))

        # Validate tier
        if job['tier'] not in ('S', 'A', 'B'):
            job['tier'] = 'B'

        # Validate deadline_status
        if job['deadline_status'] not in ('live', 'soon', 'urgent'):
            job['deadline_status'] = 'live'

        # Ensure category has tier mapping
        tier_cat_map = {'S': 'faang', 'A': 'startup', 'B': 'research'}
        base_cat = tier_cat_map.get(job['tier'], 'research')
        if base_cat not in job['category']:
            job['category'] = base_cat + ' ' + job.get('category', '')

    data['skills'] = skills
    return data


def _call_ai(prompt: str, max_retries: int = 3):
    """
    Dual AI engine: tries Gemini first, falls back to Groq instantly.
    Returns an object with a .text attribute containing the response.
    """

    # --- Try Gemini first ---
    if GEMINI_API_KEY:
        gemini_models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite']
        for model_name in gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                print(f"[AI] ✅ Gemini ({model_name}) responded")
                return response
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'exhausted' in err_str or 'quota' in err_str:
                    print(f"[AI] Gemini {model_name} rate limited, trying next...")
                    continue
                elif '404' in err_str:
                    continue
                else:
                    print(f"[AI] Gemini error: {e}")
                    break  # Don't retry on non-rate-limit errors

    # --- Fallback to Groq ---
    if groq_client:
        groq_models = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768']
        for model_name in groq_models:
            try:
                chat = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=4096,
                )
                # Wrap in a simple object with .text to match Gemini's interface
                class GroqResponse:
                    def __init__(self, text):
                        self.text = text
                result = GroqResponse(chat.choices[0].message.content)
                print(f"[AI] ✅ Groq ({model_name}) responded")
                return result
            except Exception as e:
                print(f"[AI] Groq {model_name} error: {e}")
                continue

    # --- If both fail, retry Gemini with patience ---
    if GEMINI_API_KEY:
        for attempt in range(max_retries):
            for model_name in ['gemini-2.0-flash', 'gemini-2.0-flash-lite']:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    print(f"[AI] ✅ Gemini retry succeeded ({model_name})")
                    return response
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'exhausted' in err_str or 'quota' in err_str:
                        wait_time = 15
                        print(f"[AI] Gemini retry {attempt+1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    elif '404' in err_str:
                        continue

    raise Exception("All AI providers failed. Please try again later.")


# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
