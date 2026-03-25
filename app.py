"""
OpportunityFinder — AI-Powered Internship & Job Matching
Flask backend with Gemini AI + Real Job API integration
Supports search_type='intern' (internships) and search_type='job' (full-time jobs)
"""

import os
import json
import requests as http_requests
import re
import secrets
import time
import uuid
import urllib.parse
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for, session, flash
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from groq import Groq
from job_fetcher import fetch_all_jobs, _is_elite_company

from embedding_matcher import rank_jobs_by_similarity
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

CEREBRAS_API_KEY = os.environ.get('CEREBRAS_API_KEY')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')


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

    # --- Get user's desired job count, search type, and target role ---
    job_count = int(request.form.get('job_count', 20))
    job_count = max(10, min(50, job_count))
    search_type = request.form.get('search_type', 'intern')  # 'intern' or 'job'
    if search_type not in ('intern', 'job'):
        search_type = 'intern'
    target_role = request.form.get('target_role', '').strip()
    if target_role:
        print(f"[Upload] Target role specified: {target_role}")

    # --- Call AI Pipeline ---
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        flash('No AI API key configured.')
        return redirect(url_for('index'))

    try:
        result: dict = analyze_resume_hybrid(resume_text, job_count, search_type, target_role)
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
            fallback_jobs_raw = fetch_all_jobs(fallback_skills, max_per_source=15, search_type=search_type)
        except Exception:
            fallback_jobs_raw = []
        mode_label = 'internships' if search_type == 'intern' else 'jobs'
        
        fallback_jobs_list = []
        for job in fallback_jobs_raw[:20]:
            fallback_jobs_list.append({
                "company": job.get("company", "Unknown"),
                "role": job.get("role", "Internship"),
                "tier": "A",
                "match_percentage": 75,
                "matched_skills": job.get("tags", [])[:2],
                "missing_skills": [],
                "location": job.get("location", "India"),
                "duration": "Varies",
                "stipend": job.get("salary") or "Stipend not disclosed",
                "work_mode": "Remote" if job.get("remote") else "On-site",
                "deadline_text": "\ud83d\udfe2 Open Now",
                "deadline_status": "live",
                "source_text": job.get("source_text", "\u2705 Verified Live Listing"),
                "category": "startup",
                "apply_url": job.get("apply_url", "#"),
                "verified": True
            })

        result: dict = {
            "skills": fallback_skills,
            "action_plan": f"<strong>Tip:</strong> The AI scoring system is currently at capacity. Here are live {mode_label} we found for you — try again in a few minutes for personalized scoring!",
            "jobs": fallback_jobs_list,
            "search_type": search_type
        }

    # Sort: FAANG (tier S) first, then by match percentage, trim to count
    if 'jobs' in result and result['jobs']:
        def sort_key(j):
            tier_order = {'S': 0, 'A': 1, 'B': 2}
            return (tier_order.get(j.get('tier', 'B'), 2), -j.get('match_percentage', 0))
        result['jobs'] = sorted(result['jobs'], key=sort_key)
        result['jobs'] = result['jobs'][:job_count]

    # Ensure search_type is stored in result
    result['search_type'] = search_type

    # --- Store results with freshness metadata ---
    verified_jobs = [j for j in result.get('jobs', []) if j.get('verified')]
    sources_used = list(set(j.get('source', 'Unknown') for j in result.get('jobs', [])))
    result['generated_at'] = datetime.now(timezone.utc).isoformat()
    result['verified_count'] = len(verified_jobs)
    result['total_count'] = len(result.get('jobs', []))
    result['sources_used'] = sources_used

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

    # --- Check for stale results (older than 30 minutes) ---
    generated_at_str = result.get('generated_at', '')
    is_stale = False
    generated_at_display = ''
    if generated_at_str:
        try:
            gen_time = datetime.fromisoformat(generated_at_str)
            age_minutes = (datetime.now(timezone.utc) - gen_time).total_seconds() / 60
            is_stale = age_minutes > 30
            # Format for display (e.g., "2 min ago", "15 min ago")
            if age_minutes < 1:
                generated_at_display = 'Just now'
            elif age_minutes < 60:
                generated_at_display = f'{int(age_minutes)} min ago'
            else:
                hours = int(age_minutes // 60)
                generated_at_display = f'{hours}h {int(age_minutes % 60)}m ago'
        except Exception:
            generated_at_display = 'Unknown'
    else:
        generated_at_display = 'Unknown'

    jobs = result.get('jobs', [])
    skills = result.get('skills', [])
    action_plan = result.get('action_plan', '')
    search_type = result.get('search_type', 'intern')
    verified_count = result.get('verified_count', 0)
    total_count = result.get('total_count', len(jobs))
    sources_used = result.get('sources_used', [])

    return render_template(
        'dashboard.html',
        jobs=jobs,
        skills=skills,
        action_plan=action_plan,
        search_type=search_type,
        generated_at=generated_at_display,
        is_stale=is_stale,
        verified_count=verified_count,
        total_count=total_count,
        sources_used=sources_used,
    )


# ============================================
# HYBRID AI + REAL JOBS PIPELINE
# ============================================

def analyze_resume_hybrid(resume_text: str, job_count: int = 20, search_type: str = 'intern', target_role: str = '') -> dict:
    """
    Full accuracy pipeline (sequential):
    1. AI extracts structured profile from resume (skills with proficiency)
    2. Fetch real jobs/internships using extracted skills + target role
    2.5. Rank all jobs via embedding similarity → keep top N
    3. AI scores pre-ranked jobs using the full profile context
    """
    import time as _time
    t0 = _time.time()

    # --- STEP 1: Extract structured profile with AI ---
    profile = extract_profile_with_gemini(resume_text)
    skills = profile.get('skill_names', [])
    print(f"[AI] Extracted profile: {profile.get('domain', '?')} | {len(skills)} skills | {profile.get('experience_level', '?')} ({_time.time()-t0:.1f}s)")
    
    if not skills:
        raise ValueError("NO_SKILLS_FOUND")

    # If user specified a target role, inject it as a skill keyword for better fetching
    if target_role:
        profile['target_role'] = target_role
        # Add role keywords to skills for search (don't duplicate)
        role_keywords = [w.strip() for w in target_role.replace('/', ' ').replace('(', ' ').replace(')', ' ').split() if len(w.strip()) > 2]
        for kw in role_keywords:
            if kw not in skills:
                skills.append(kw)

    # --- STEP 2: Fetch real jobs — fetch more if user wants more ---
    fetch_per_source = max(25, job_count * 2)
    real_jobs_raw = fetch_all_jobs(skills, max_per_source=fetch_per_source, search_type=search_type)
    mode_label = 'internships' if search_type == 'intern' else 'jobs'
    print(f"[API] Fetched {len(real_jobs_raw)} real {mode_label} ({_time.time()-t0:.1f}s)")

    # --- STEP 2.5: Rank via embedding similarity (pre-filter) ---
    # Build a richer resume summary for embedding using the profile
    profile_summary = _build_profile_summary(profile, resume_text)
    top_n_for_llm = min(len(real_jobs_raw), max(25, job_count))
    try:
        ranked_jobs = rank_jobs_by_similarity(profile_summary, real_jobs_raw, top_n=top_n_for_llm)
        print(f"[Embedding] Pre-ranked to {len(ranked_jobs)} best matches ({_time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[Embedding] Failed, falling back to unranked: {e}")
        ranked_jobs = real_jobs_raw[:top_n_for_llm]

    # --- STEP 3: Score pre-ranked jobs + generate AI suggestions ---
    result = score_and_suggest_with_gemini(resume_text, profile, ranked_jobs, job_count, search_type, target_role)
    print(f"[Total] Pipeline complete in {_time.time()-t0:.1f}s")

    return result


def _build_profile_summary(profile: dict, resume_text: str) -> str:
    """Build a rich text summary from the structured profile for embedding."""
    parts = []
    if profile.get('domain'):
        parts.append(f"Domain: {profile['domain']}")
    if profile.get('experience_level'):
        parts.append(f"Experience Level: {profile['experience_level']}")
    if profile.get('experience_years') is not None:
        parts.append(f"Years of Experience: {profile['experience_years']}")
    if profile.get('education'):
        parts.append(f"Education: {profile['education']}")
    
    # Include skills with proficiency
    skill_parts = []
    for s in profile.get('skills', []):
        if isinstance(s, dict):
            skill_parts.append(f"{s.get('name', '')} ({s.get('proficiency', 'unknown')})")
        else:
            skill_parts.append(str(s))
    if skill_parts:
        parts.append(f"Skills: {', '.join(skill_parts)}")
    
    if profile.get('certifications'):
        parts.append(f"Certifications: {', '.join(profile['certifications'])}")
    
    # Append truncated resume for additional context
    parts.append(resume_text[:1500])
    
    return '\n'.join(parts)


def extract_profile_with_gemini(resume_text: str) -> dict:
    """
    Extract a structured candidate profile from resume.
    Returns a dict with skills (including proficiency), experience, domain, etc.
    Falls back to flat skill extraction if AI fails.
    """

    prompt = f"""Analyze this resume and extract a structured candidate profile.
The resume could be from ANY field (Tech, Marketing, Finance, Design, HR, Sales, etc.).

Return ONLY valid JSON, no markdown, no code fences, no explanation.

{{
  "domain": "Primary professional domain (e.g., Full-Stack Web Development, Data Science, Digital Marketing, Finance)",
  "experience_level": "beginner / intermediate / advanced / expert",
  "experience_years": 2,
  "education": "Degree and field (e.g., B.Tech Computer Science, MBA Finance)",
  "skills": [
    {{
      "name": "Python",
      "proficiency": "advanced",
      "evidence": "3 projects including ML pipeline and Django backend"
    }},
    {{
      "name": "React",
      "proficiency": "intermediate",
      "evidence": "1 project with basic components"
    }},
    {{
      "name": "SQL",
      "proficiency": "beginner",
      "evidence": "mentioned in coursework only"
    }}
  ],
  "project_complexity": "low / medium / high",
  "certifications": ["AWS Cloud Practitioner"],
  "key_achievements": ["Built a production app serving 1000+ users"]
}}

PROFICIENCY RULES:
- "advanced": Built multiple real projects, used in work/production, deep understanding
- "intermediate": 1-2 projects, comfortable but not expert, used in academic or personal projects
- "beginner": Mentioned in coursework, listed but no project evidence, minimal usage
- "expert": Professional production experience, certifications, teaches/leads in this area

Keep to 10-15 of the most important skills. Be honest and realistic about proficiency — do NOT inflate.

RESUME:
{resume_text}"""

    try:
        response = _call_ai(prompt, max_retries=3)
        raw = response.text.strip()
    except Exception as e:
        print(f"[AI] Profile extraction failed, using local fallback: {e}")
        return _build_fallback_profile(resume_text)

    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)

    profile = None
    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                profile = json.loads(re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', json_match.group())))
            except json.JSONDecodeError:
                pass

    if not profile or 'skills' not in profile:
        print("[AI] Profile parse failed, using local fallback")
        return _build_fallback_profile(resume_text)

    # Ensure skill_names list exists (flat list for job fetching API)
    profile['skill_names'] = [s['name'] if isinstance(s, dict) else str(s) for s in profile.get('skills', [])]
    profile.setdefault('domain', 'General')
    profile.setdefault('experience_level', 'intermediate')
    profile.setdefault('experience_years', 0)
    profile.setdefault('education', 'Not specified')
    profile.setdefault('project_complexity', 'medium')
    profile.setdefault('certifications', [])
    profile.setdefault('key_achievements', [])

    return profile


def _build_fallback_profile(resume_text: str) -> dict:
    """Build a basic profile using local keyword extraction when AI fails."""
    skill_names = _extract_skills_local(resume_text)
    return {
        'domain': 'General',
        'experience_level': 'intermediate',
        'experience_years': 0,
        'education': 'Not specified',
        'skills': [{'name': s, 'proficiency': 'intermediate', 'evidence': 'Found in resume'} for s in skill_names],
        'skill_names': skill_names,
        'project_complexity': 'medium',
        'certifications': [],
        'key_achievements': [],
    }


def _format_profile_for_prompt(profile: dict) -> str:
    """Format the structured profile as readable text for the LLM prompt."""
    lines = []
    lines.append(f"Domain: {profile.get('domain', 'General')}")
    lines.append(f"Experience Level: {profile.get('experience_level', 'intermediate')}")
    lines.append(f"Experience: ~{profile.get('experience_years', 0)} years")
    lines.append(f"Education: {profile.get('education', 'Not specified')}")
    lines.append(f"Project Complexity: {profile.get('project_complexity', 'medium')}")
    
    if profile.get('certifications'):
        lines.append(f"Certifications: {', '.join(profile['certifications'])}")
    if profile.get('key_achievements'):
        lines.append(f"Key Achievements: {', '.join(profile['key_achievements'][:3])}")
    
    lines.append("")
    lines.append("SKILLS WITH PROFICIENCY:")
    for s in profile.get('skills', []):
        if isinstance(s, dict):
            name = s.get('name', '?')
            prof = s.get('proficiency', 'unknown').upper()
            evidence = s.get('evidence', '')
            lines.append(f"  - {name} [{prof}]: {evidence}")
        else:
            lines.append(f"  - {s} [INTERMEDIATE]")
    
    return '\n'.join(lines)


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


def score_and_suggest_with_gemini(resume_text: str, profile: dict, real_jobs: list, job_count: int = 20, search_type: str = 'intern', target_role: str = '') -> dict:
    """
    Send real jobs + structured profile to AI for scoring.
    Uses proficiency-aware matching for accurate percentages.
    Generates exactly job_count results with FAANG priority.
    """

    # Extract flat skills list from profile
    skills = profile.get('skill_names', [])

    # Build proficiency context for the prompt
    profile_text = _format_profile_for_prompt(profile)

    # Format real jobs for the prompt
    max_real_in_prompt = min(len(real_jobs), max(25, job_count))
    real_jobs_text = ""
    for i, job in enumerate(real_jobs[:max_real_in_prompt], 1):
        sim_line = ""
        if 'similarity_score' in job:
            sim_line = f"\n- Relevance Score: {job['similarity_score']:.4f}  (mathematical similarity to resume)"
        real_jobs_text += f"""
JOB #{i}:
- Company: {job['company']}
- Role: {job['role']}
- Location: {job['location']}
- Source: {job['source']}
- URL: {job['apply_url']}
- Tags: {', '.join(job.get('tags', []))}
- Remote: {job.get('remote', False)}{sim_line}
"""

    # Target role context for the prompt
    role_context = ''
    if target_role:
        role_context = f"\n\nIMPORTANT: The user has specifically requested jobs for the role: \"{target_role}\". PRIORITIZE and SCORE HIGHER any jobs that match this target role. Jobs that don't match this role should receive lower scores."

    # Mode-dependent prompt sections
    if search_type == 'intern':
        mode_intro = f"A student from ANY field (Tech, Business, Arts, Marketing, Finance, etc.) has uploaded their resume and is looking for INTERNSHIPS.{role_context}"
        mode_suggest = """2. ALSO suggest 8-10 ADDITIONAL internship opportunities from your knowledge (mark these as AI-suggested).
   IMPORTANT: AI-suggested internships MUST be based in India (Bangalore, Mumbai, Delhi, Pune, Hyderabad, etc.) or Remote. Do NOT suggest US/European on-site roles."""
        tier_s_text = f"""MANDATORY TIER-S SUGGESTIONS:
- You MUST suggest at least 3-5 top-tier, highly prestigious internships (tier "S") that are perfectly aligned with the student's specific field.
- For Tech: FAANG/Big Tech (Google, Microsoft, Amazon, etc).
- For Finance: Top banks (Goldman Sachs, JPMorgan, Morgan Stanley).
- For Marketing/Business/Design: Top agencies, Fortune 500 FMCGs, or industry leaders.
- These should be real internship/new-grad programs that these top companies typically offer in India."""
        role_policy = "- STRICT INTERNSHIP POLICY: ALL jobs (real and suggested) MUST be Internships, Co-ops, Trainee, or Fresher roles. You MUST REJECT AND IGNORE any full-time or senior/managerial roles."
        duration_example = '"3 months"'
        pay_example = '"\u20b940K/mo or Stipend not disclosed"'
    else:
        mode_intro = f"A professional from ANY field (Tech, Business, Arts, Marketing, Finance, etc.) has uploaded their resume and is looking for FULL-TIME JOBS.{role_context}"
        mode_suggest = """2. ALSO suggest 8-10 ADDITIONAL full-time job opportunities from your knowledge (mark these as AI-suggested).
   IMPORTANT: AI-suggested jobs MUST be based in India (Bangalore, Mumbai, Delhi, Pune, Hyderabad, etc.) or Remote. Do NOT suggest US/European on-site roles."""
        tier_s_text = f"""MANDATORY TIER-S SUGGESTIONS:
- You MUST suggest at least 3-5 top-tier, highly prestigious full-time positions (tier "S") that are perfectly aligned with the candidate's specific field.
- For Tech: FAANG/Big Tech (Google, Microsoft, Amazon, etc).
- For Finance: Top banks (Goldman Sachs, JPMorgan, Morgan Stanley).
- For Marketing/Business/Design: Top agencies, Fortune 500 FMCGs, or industry leaders.
- These should be real job openings or hiring programs that these top companies typically offer in India."""
        role_policy = "- STRICT JOB POLICY: ALL results MUST be entry-level, mid-level, or junior full-time positions. You MUST REJECT AND IGNORE any internship, co-op, or senior/director/VP roles."
        duration_example = '"Full-time" or "Contract 6 months"'
        pay_example = '"\u20b98-15 LPA or Not disclosed"'

    prompt = f"""You are an expert personalized recruiter AI. {mode_intro}

CANDIDATE PROFILE (AI-analyzed from their resume):
---
{profile_text}
---

RESUME TEXT:
---
{resume_text}
---

REAL JOB LISTINGS (pre-ranked by embedding similarity, most relevant first):
---
{real_jobs_text if real_jobs_text.strip() else "No real jobs were found from APIs. Generate all suggestions from your knowledge."}
---

SCORING INSTRUCTIONS:
- Use the CANDIDATE PROFILE above (especially proficiency levels and evidence) to determine match_percentage.
- A candidate with "advanced" Python (3 production projects) should score HIGHER on a Python-heavy role than one with "beginner" Python (coursework only).
- Consider experience_level, project_complexity, and domain alignment — not just skill name overlap.
- The Relevance Score on each job is the mathematical embedding similarity. Use it as a starting baseline but adjust based on proficiency depth.

YOUR TASKS:
1. Score each REAL job listing above against the candidate's profile (match_percentage).
{mode_suggest}
3. Provide a personalized action plan that references specific skill gaps.

{tier_s_text}

IMPORTANT: Return ONLY valid JSON, no markdown, no code fences, no explanation.

{{
  "skills": {json.dumps(skills)},
  "action_plan": "HTML string with <strong>, <br>, <a> tags. 3-4 lines of personalized, actionable advice referencing specific skill proficiencies.",
  "jobs": [
    {{
      "company": "Company Name",
      "role": "Role Title · Track",
      "tier": "S or A or B",
      "match_percentage": 85,
      "matched_skills": ["skill1 (advanced)", "skill2 (intermediate)"],
      "missing_skills": ["skill3"],
      "location": "City, Country",
      "duration": {duration_example},
      "stipend": {pay_example},
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
- match_percentage: 40-98, be realistic based on PROFICIENCY DEPTH not just skill name matching.
- deadline_status: "live", "soon", or "urgent".
- MUST include at least {max(3, job_count // 4)} tier-S (Top Prestige / FAANG / Fortune 500) jobs in the suggestions.
- You MUST return EXACTLY {job_count} jobs total (real + suggested combined). Not less, not more.
- Put tier-S jobs FIRST, then other jobs sorted by match percentage.
{role_policy}
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
        skills_lower = set(s.lower() for s in skills)

        data: dict = {
            "skills": skills,
            "action_plan": f"<strong>Tip:</strong> Your skills match well with current openings! The AI scoring is at capacity right now, but we've computed skill-based match scores for you. Try again shortly for full proficiency-aware analysis.",
            "jobs": []
        }

        for i, job in enumerate(real_jobs[:job_count]):
            # Compute real match percentage based on skill overlap
            job_text = f"{job.get('role', '')} {' '.join(job.get('tags', []))} {job.get('company', '')} {job.get('description', '')}".lower()
            matched = [s for s in skills if s.lower() in job_text]
            not_matched = [s for s in skills if s.lower() not in job_text]
            match_pct = min(95, max(40, int(len(matched) / max(len(skills), 1) * 100) + 35))

            # Boost match if target_role matches job title
            if target_role:
                role_lower = target_role.lower()
                title_lower = job.get('role', '').lower()
                if role_lower in title_lower or any(w in title_lower for w in role_lower.split() if len(w) > 2):
                    match_pct = min(95, match_pct + 10)

            # Elite-only FAANG rule: elite companies always tier-S/faang, non-elite never faang
            if _is_elite_company(job.get('company', '')):
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
                "matched_skills": matched[:5],
                "missing_skills": not_matched[:3],
                "location": job.get("location", "India"),
                "duration": "Varies",
                "stipend": job.get("salary") or "Stipend not disclosed",
                "work_mode": "Remote" if job.get("remote") else "On-site",
                "deadline_text": "🟢 Open Now",
                "deadline_status": "live",
                "source_text": job.get("source_text", "✅ Verified Live Listing"),
                "category": category,
                "apply_url": job.get("apply_url", "#"),
                "verified": True
            })

    # --- ALWAYS ADD: AI-Suggested Elite Company Jobs ---
    if search_type == 'intern':
        faang_pool = [
            {"company": "Google", "role": "SWE Intern · STEP Program", "location": "Bangalore, India"},
            {"company": "Microsoft", "role": "Software Engineering Intern", "location": "Hyderabad, India"},
            {"company": "Amazon", "role": "SDE Intern", "location": "Bangalore, India"},
            {"company": "Goldman Sachs", "role": "Engineering Intern", "location": "Bangalore, India"},
            {"company": "Adobe", "role": "Software Development Intern", "location": "Noida, India"},
            {"company": "Uber", "role": "Software Engineer Intern", "location": "Bangalore, India"},
            {"company": "Meta", "role": "Software Engineer Intern", "location": "Remote"},
            {"company": "Salesforce", "role": "Software Engineering Intern", "location": "Hyderabad, India"},
            {"company": "Flipkart", "role": "SDE Intern", "location": "Bangalore, India"},
            {"company": "Razorpay", "role": "Software Engineering Intern", "location": "Bangalore, India"},
            {"company": "NVIDIA", "role": "Deep Learning Intern", "location": "Pune, India"},
            {"company": "JPMorgan Chase", "role": "Software Engineer Program", "location": "Mumbai, India"}
        ]
    else:
        faang_pool = [
            {"company": "Google", "role": "Software Engineer · L3", "location": "Bangalore, India"},
            {"company": "Microsoft", "role": "Software Development Engineer", "location": "Hyderabad, India"},
            {"company": "Amazon", "role": "SDE I", "location": "Bangalore, India"},
            {"company": "Goldman Sachs", "role": "Analyst · Engineering", "location": "Bangalore, India"},
            {"company": "Adobe", "role": "Software Developer", "location": "Noida, India"},
            {"company": "Uber", "role": "Software Engineer I", "location": "Bangalore, India"},
            {"company": "Meta", "role": "Software Engineer", "location": "Remote"},
            {"company": "Salesforce", "role": "Associate Software Engineer", "location": "Hyderabad, India"},
            {"company": "Flipkart", "role": "SDE-1", "location": "Bangalore, India"},
            {"company": "Razorpay", "role": "Software Engineer", "location": "Bangalore, India"},
            {"company": "NVIDIA", "role": "Deep Learning Engineer", "location": "Pune, India"},
            {"company": "JPMorgan Chase", "role": "Associate · Software Engineer", "location": "Mumbai, India"}
        ]

    # Only add suggestions that aren't already in results
    existing_companies = {j.get('company', '').lower() for j in data.get('jobs', [])}
    search_suffix = 'internship' if search_type == 'intern' else 'careers apply'
    num_faang_to_add = max(3, job_count // 4)
    added = 0
    for fs in faang_pool:
        if added >= num_faang_to_add:
            break
        if fs['company'].lower() in existing_companies:
            continue
        search_q = f"{fs['company']} {fs['role']} apply {search_suffix} 2026"
        data["jobs"].append({
            "company": fs["company"],
            "role": fs["role"],
            "tier": "S",
            "match_percentage": min(95, max(50, int(len(skills) / max(1, 15) * 80) + 10)),
            "matched_skills": skills[:4],
            "missing_skills": [],
            "location": fs["location"],
            "duration": "2-6 months" if search_type == 'intern' else "Full-time",
            "stipend": "Stipend not disclosed",
            "work_mode": "Hybrid" if "Remote" not in fs["location"] else "Remote",
            "deadline_text": "🟡 Rolling Applications",
            "deadline_status": "soon",
            "source_text": "🤖 AI Suggested",
            "category": "faang",
            "apply_url": "https://www.google.com/search?q=" + urllib.parse.quote_plus(search_q),
            "verified": False
        })
        added += 1
    print(f"[AI Suggest] Added {added} AI-suggested elite company jobs")

    # ================================================================
    # DETERMINISTIC SCORING FORMULA
    # ================================================================
    # Skill alias map: maps common abbreviations and alternates
    # Skill alias map (keys are capitalized for display)
    SKILL_ALIASES = {
        'JavaScript': ['js', 'javascript', 'ecmascript'],
        'TypeScript': ['ts', 'typescript'],
        'Python': ['python', 'py', 'django', 'flask', 'fastapi'],
        'React': ['react', 'reactjs', 'react.js'],
        'Angular': ['angular', 'angularjs', 'angular.js'],
        'Vue': ['vue', 'vuejs', 'vue.js'],
        'Node.js': ['node', 'nodejs', 'node.js', 'express'],
        'Java': ['java', 'spring', 'springboot', 'spring boot'],
        'C++': ['c++', 'cpp', 'c plus plus'],
        'C#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
        'C': ['c language', 'c programming'],
        'Go': ['golang', 'go lang', 'go'],
        'Rust': ['rust', 'rustlang'],
        'Ruby': ['ruby', 'rails', 'ruby on rails'],
        'PHP': ['php', 'laravel', 'symfony'],
        'Swift': ['swift', 'swiftui'],
        'Kotlin': ['kotlin', 'android'],
        'SQL': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'database', 'rdbms'],
        'NoSQL': ['nosql', 'mongodb', 'mongo', 'cassandra', 'dynamodb', 'redis'],
        'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
        'Azure': ['azure', 'microsoft azure'],
        'GCP': ['gcp', 'google cloud', 'google cloud platform'],
        'Docker': ['docker', 'containerization', 'containers'],
        'Kubernetes': ['kubernetes', 'k8s'],
        'Git': ['git', 'github', 'gitlab', 'version control'],
        'Linux': ['linux', 'unix', 'bash', 'shell'],
        'Machine Learning': ['machine learning', 'ml', 'deep learning', 'dl', 'neural network'],
        'Data Science': ['data science', 'data analysis', 'analytics', 'pandas', 'numpy'],
        'DevOps': ['devops', 'ci/cd', 'cicd', 'jenkins', 'terraform'],
        'HTML': ['html', 'html5', 'markup'],
        'CSS': ['css', 'css3', 'sass', 'scss', 'less', 'tailwind', 'bootstrap'],
        'TensorFlow': ['tensorflow', 'tf', 'keras'],
        'PyTorch': ['pytorch', 'torch'],
        'Excel': ['excel', 'spreadsheet', 'ms excel'],
        'PowerBI': ['power bi', 'powerbi'],
        'Tableau': ['tableau', 'data visualization'],
        'R': ['r programming', 'r language', 'rstudio'],
        'Figma': ['figma', 'ui design', 'ux design'],
        'MATLAB': ['matlab', 'simulink'],
    }

    import re

    def _has_term(term: str, doc: str) -> bool:
        """Safely check if a term exists as a whole word (using lookaround to support symbols like C++)"""
        escaped = re.escape(term.lower())
        pattern = r'(?<!\w)' + escaped + r'(?!\w)'
        return bool(re.search(pattern, doc))

    def compute_match_score(job: dict, user_skills: list, target_role: str, search_type: str) -> dict:
        """
        Multi-factor scoring formula:
        Total = Skill Match (50%) + Role Match (30%) + Experience Match (20%) + Bonus
        Returns dict with: match_percentage, matched_skills, missing_skills
        """
        # --- 1. SKILL MATCH (50 points max) ---
        # Build comprehensive search text from all available job fields
        job_parts = [
            job.get('role', ''),
            job.get('company', ''),
            job.get('description', ''),
            ' '.join(job.get('tags', [])),
            ' '.join(job.get('matched_skills', [])),  # Include pre-existing AI skill matches
            job.get('category', ''),
        ]
        job_text = ' '.join(str(p) for p in job_parts).lower()

        # Step A: Identify the JOB's Required Skills
        job_req_skills_set = set()
        for display_name, aliases in SKILL_ALIASES.items():
            if _has_term(display_name, job_text) or any(_has_term(a, job_text) for a in aliases):
                job_req_skills_set.add(display_name)
                
        # Include skills the AI detected as required for this job (to catch contextual requirements)
        ai_skills = job.get('matched_skills', []) + job.get('missing_skills', [])
        for s in ai_skills:
            raw_skill = s.split(' (')[0].strip()
            found = False
            for display_name, aliases in SKILL_ALIASES.items():
                if raw_skill.lower() == display_name.lower() or raw_skill.lower() in aliases:
                    job_req_skills_set.add(display_name)
                    found = True
                    break
            if not found and raw_skill:
                job_req_skills_set.add(raw_skill.title())
                
        job_req_skills = list(job_req_skills_set)
        
        # Step B: Normalize User's Skills
        user_canonical = set()
        for us in user_skills:
            found = False
            for display_name, aliases in SKILL_ALIASES.items():
                if us.lower().strip() == display_name.lower() or us.lower().strip() in aliases:
                    user_canonical.add(display_name)
                    found = True
                    break
            if not found and us.strip():
                user_canonical.add(us.title().strip())

        # Step C: Compare Job's Required Skills against User's Skills
        matched = [s for s in job_req_skills if s in user_canonical]
        missing = [s for s in job_req_skills if s not in user_canonical]

        # Scoring Logic
        if not job_req_skills:
            # Fallback if job text is super sparse: reward user for matching any subset of their own skills
            matched = [s for s in user_canonical if _has_term(s, job_text)]
            missing = []
            skill_ratio = len(matched) / max(len(user_canonical), 1)
        else:
            skill_ratio = len(matched) / max(len(job_req_skills), 1)

        skill_score = min(50, int(skill_ratio * 50))  # 0-50 points, capped

        # --- 2. ROLE TITLE MATCH (30 points max) ---
        role_score = 0
        if target_role:
            title_lower = job.get('role', '').lower()
            # Support multiple comma-separated roles — take the BEST match
            roles = [r.strip() for r in target_role.split(',') if r.strip()]
            best_role_score = 0
            for single_role in roles:
                role_lower = single_role.lower()
                role_words = [w for w in role_lower.replace('/', ' ').replace('(', ' ').replace(')', ' ').split() if len(w) > 2]
                
                # Exact role match (full target role appears in title)
                if role_lower in title_lower:
                    best_role_score = 30
                    break  # Can't do better than exact
                # Strong partial match (most role words appear)
                elif role_words:
                    word_hits = sum(1 for w in role_words if w in title_lower)
                    word_ratio = word_hits / len(role_words)
                    if word_ratio >= 0.6:
                        best_role_score = max(best_role_score, 25)
                    elif word_ratio >= 0.3:
                        best_role_score = max(best_role_score, 15)
            role_score = best_role_score
        else:
            # No target role specified — give a neutral 15/30 to all jobs
            role_score = 15

        # --- 3. EXPERIENCE ALIGNMENT (20 points max) ---
        exp_score = 10  # Default: neutral
        title_lower = job.get('role', '').lower()
        intern_keywords = ['intern', 'internship', 'co-op', 'trainee', 'fresher', 'apprentice', 'student', 'graduate', 'entry']
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'director', 'vp', 'head', 'manager']
        
        if search_type == 'intern':
            if any(kw in title_lower for kw in intern_keywords):
                exp_score = 20  # Perfect: student looking for internship
            elif any(kw in title_lower for kw in senior_keywords):
                exp_score = 0   # Hard mismatch: student vs senior role
            else:
                exp_score = 10  # Neutral: generic title
        else:
            if any(kw in title_lower for kw in senior_keywords):
                exp_score = 5   # Slight mismatch: entry-level candidate vs senior
            elif any(kw in title_lower for kw in intern_keywords):
                exp_score = 5   # Slight mismatch: professional vs internship
            else:
                exp_score = 20  # Good: professional looking for full-time

        # --- 4. PRESTIGE BONUS (up to +15 points) ---
        bonus = 0
        if _is_elite_company(job.get('company', '')):
            bonus += 10
        if job.get('remote') or 'remote' in job.get('location', '').lower():
            bonus += 5

        # --- FINAL SCORE ---
        raw_score = skill_score + role_score + exp_score + bonus
        final_score = max(0, min(100, int(raw_score)))

        return {
            'match_percentage': final_score,
            'matched_skills': matched[:6],
            'missing_skills': missing[:4],
        }

    # Validate and set defaults for all jobs
    if not isinstance(data, dict):
        return data # fallback
    
    jobs_list = data.get('jobs', [])
    for job_item in jobs_list:
        if not isinstance(job_item, dict):
            continue
        job: dict = job_item
        job.setdefault('company', 'Unknown')
        job.setdefault('role', 'Internship')
        job.setdefault('tier', 'B')
        job.setdefault('matched_skills', [])
        job.setdefault('missing_skills', [])
        job.setdefault('location', 'Remote')
        job.setdefault('duration', 'Varies')

        # --- FIX STIPEND: Map 'salary' from fetchers → 'stipend' for dashboard ---
        if 'stipend' not in job or not job['stipend']:
            # Pull from the raw 'salary' field that fetchers provide
            raw_salary = job.get('salary', '')
            if raw_salary and str(raw_salary).strip().lower() not in ('', 'none', 'null', 'nan', '0'):
                job['stipend'] = str(raw_salary).strip()
            else:
                job['stipend'] = 'Stipend not disclosed'
        
        # Normalize stipend: clean up bad values
        stipend_val = str(job['stipend']).strip().lower()
        if not stipend_val or stipend_val in ('none', 'null', 'nan', 'competitive', 'not disclosed', 'unpaid', '0', 'stipend not disclosed'):
            job['stipend'] = 'Stipend not disclosed'
        job.setdefault('work_mode', 'Remote')
        job.setdefault('deadline_text', '🟢 Open')
        job.setdefault('deadline_status', 'live')
        job.setdefault('verified', False)
        job.setdefault('apply_url', '#')

        # For AI-suggested jobs, generate a Google search link instead of hallucinated URLs
        if not job.get('verified'):
            search_suffix = 'internship' if search_type == 'intern' else 'careers apply'
            search_query = f"{job['company']} {job['role']} apply {search_suffix} 2026"
            job['apply_url'] = 'https://www.google.com/search?q=' + urllib.parse.quote_plus(search_query)

        # Set source text based on verification
        if job.get('verified'):
            job.setdefault('source_text', '✅ Verified Live Listing')
        else:
            job.setdefault('source_text', '🤖 AI Suggested')

        job.setdefault('category', '')

        # --- APPLY DETERMINISTIC SCORING FORMULA ---
        # Compute our text-based skill match
        score_result = compute_match_score(job, skills, target_role, search_type)
        
        # Use our deterministic match_percentage (consistent, reproducible)
        job['match_percentage'] = score_result['match_percentage']
        
        # For matched/missing skills: use BEST available source —
        # If AI provided richer matched_skills (more than our text search found), keep AI's.
        # Otherwise use our computed ones (which have alias support).
        ai_matched = job.get('matched_skills', [])
        computed_matched = score_result['matched_skills']
        computed_missing = score_result['missing_skills']
        
        if len(computed_matched) >= len(ai_matched):
            # Our text search found more/equal matches — use ours
            job['matched_skills'] = computed_matched
            job['missing_skills'] = computed_missing
        else:
            # AI found more matches (it has context about job requirements)
            # Keep AI's matched_skills. Use AI's missing skills if provided, otherwise use ours.
            if not job.get('missing_skills'):
                job['missing_skills'] = computed_missing

        # Validate tier
        if job['tier'] not in ('S', 'A', 'B'):
            job['tier'] = 'B'

        # --- ELITE-ONLY FAANG RULE ---
        is_elite = _is_elite_company(job.get('company', ''))
        if is_elite:
            # Elite companies ALWAYS go to FAANG section as tier-S
            job['tier'] = 'S'
            job['category'] = 'faang'
        else:
            # Non-elite companies NEVER appear in FAANG section
            # If AI mistakenly gave 'faang' category, reassign
            if 'faang' in job.get('category', ''):
                job['category'] = job['category'].replace('faang', '').strip()
            # Assign based on tier
            if job['tier'] == 'S' and not is_elite:
                job['tier'] = 'A'  # Demote non-elite from S to A

        # Validate deadline_status
        if job['deadline_status'] not in ('live', 'soon', 'urgent'):
            job['deadline_status'] = 'live'

        # Ensure category has tier mapping (for non-elite only)
        if not is_elite:
            tier_cat_map = {'A': 'startup', 'B': 'research'}
            base_cat = tier_cat_map.get(job['tier'], 'research')
            if base_cat not in job['category']:
                job['category'] = base_cat + ' ' + job.get('category', '')

    data['skills'] = skills

    # --- FILTER: Remove jobs below 60% match (except elite + AI-suggested) ---
    MIN_MATCH_THRESHOLD = 60
    pre_filter_count = len(data.get('jobs', []))
    filtered_jobs = []
    for j in data.get('jobs', []):
        if not isinstance(j, dict):
            continue
        match_pct = j.get('match_percentage', 0)
        is_elite_job = _is_elite_company(j.get('company', ''))
        is_ai_suggested = '🤖' in j.get('source_text', '')
        # Keep: elite companies, AI-suggestions, and anything >= 60%
        if match_pct >= MIN_MATCH_THRESHOLD or is_elite_job or is_ai_suggested:
            filtered_jobs.append(j)
    data['jobs'] = filtered_jobs
    if pre_filter_count != len(filtered_jobs):
        print(f"[Filter] Removed {pre_filter_count - len(filtered_jobs)} jobs below {MIN_MATCH_THRESHOLD}% threshold")

    # --- HARDCORE ENFORCE EXACT COUNT: trim or pad ---
    jobs_list = data.get('jobs', [])
    if len(jobs_list) > job_count:
        data['jobs'] = jobs_list[:job_count]
        print(f"[Count] Trimmed {len(jobs_list)} → {job_count} jobs (exact match)")
    elif len(jobs_list) < job_count:
        print(f"[Count] Warning: LLM returned {len(jobs_list)}/{job_count}. HARDCORE PADDING ACTIVE.")
        
        # Phase 1 Padding: Pull from real_jobs that weren't included
        seen_urls = {j.get('apply_url', '') for j in jobs_list if j.get('apply_url')}
        for job in real_jobs:
            if len(jobs_list) >= job_count:
                break
            url = job.get('apply_url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                company = job.get("company", "Unknown")
                tier = "S" if any(c in company.lower() for c in ['google','microsoft','amazon','meta','apple']) else "B"
                jobs_list.append({
                    "company": company,
                    "role": job.get("role", "Unknown Role"),
                    "tier": tier,
                    "match_percentage": min(85, max(45, int(job.get('similarity_score', 0.5) * 100))),
                    "matched_skills": skills[:2] if skills else [],
                    "missing_skills": [],
                    "location": job.get("location", "India"),
                    "duration": "Varies",
                    "stipend": job.get("salary") or "Stipend not disclosed",
                    "work_mode": "Remote" if job.get("remote") else "On-site",
                    "deadline_text": "🟢 Open Now",
                    "deadline_status": "live",
                    "source_text": job.get("source_text", "✅ Verified Live Listing"),
                    "category": job.get("category", "startup"),
                    "apply_url": url,
                    "verified": True
                })
                
        # Phase 2 Padding: If STILL short, inject fallback generic suggestions
        generic_companies = ["IBM", "Cisco", "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "Tech Mahindra", "HCL", "Deloitte", "EY", "PwC", "KPMG", "Optum", "Walmart"]
        padding_idx = 0
        while len(jobs_list) < job_count:
            search_suffix = 'internship' if search_type == 'intern' else 'careers apply'
            role_type = "Software Engineer Intern" if search_type == 'intern' else "Software Engineer"
            padding_company = generic_companies[padding_idx % len(generic_companies)]
            search_q = f"{padding_company} {role_type} apply {search_suffix} 2026"
            
            jobs_list.append({
                "company": padding_company,
                "role": role_type,
                "tier": "A",
                "match_percentage": 50,
                "matched_skills": skills[:2] if skills else [],
                "missing_skills": [],
                "location": "India",
                "duration": "2-6 months" if search_type == 'intern' else "Full-time",
                "stipend": "Stipend not disclosed",
                "work_mode": "Hybrid",
                "deadline_text": "🟡 Rolling Applications",
                "deadline_status": "soon",
                "source_text": "🤖 AI Suggested",
                "category": "startup",
                "apply_url": "https://www.google.com/search?q=" + urllib.parse.quote_plus(search_q),
                "verified": False
            })
            padding_idx += 1
            
        data['jobs'] = jobs_list
        print(f"[Count] Padded to exactly {len(jobs_list)} jobs.")

    return data


def _call_openai_compatible(api_key: str, base_url: str, model: str, prompt: str, provider: str, extra_headers: dict = None):
    """Generic caller for any OpenAI-compatible chat API (Cerebras, Together, OpenRouter, etc.)"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 4096,
    }
    resp = http_requests.post(f'{base_url}/chat/completions', headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data['choices'][0]['message']['content']

    class GenericResponse:
        def __init__(self, t):
            self.text = t
    print(f"[AI] ✅ {provider} ({model}) responded")
    return GenericResponse(text)


def _call_ai(prompt: str, max_retries: int = 2):
    """
    Multi-provider AI engine with 5-level fallback chain:
    Gemini → Groq → Cerebras → Together → OpenRouter → Gemini retry
    Returns an object with a .text attribute containing the response.
    """

    # --- 1. Try Gemini first ---
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
                    break

    # --- 2. Fallback to Groq ---
    if groq_client:
        groq_models = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'gemma2-9b-it']
        for model_name in groq_models:
            try:
                chat = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )
                class GroqResponse:
                    def __init__(self, text):
                        self.text = text
                result = GroqResponse(chat.choices[0].message.content)
                print(f"[AI] ✅ Groq ({model_name}) responded")
                return result
            except Exception as e:
                print(f"[AI] Groq {model_name} error: {e}")
                continue

    # --- 3. Fallback to Cerebras ---
    if CEREBRAS_API_KEY:
        cerebras_models = ['llama-3.3-70b', 'llama-3.1-8b']
        for model_name in cerebras_models:
            try:
                return _call_openai_compatible(
                    CEREBRAS_API_KEY, 'https://api.cerebras.ai/v1',
                    model_name, prompt, 'Cerebras'
                )
            except Exception as e:
                print(f"[AI] Cerebras {model_name} error: {e}")
                continue

    # --- 4. Fallback to Together AI ---
    if TOGETHER_API_KEY:
        together_models = ['meta-llama/Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Llama-3.1-8B-Instruct-Turbo']
        for model_name in together_models:
            try:
                return _call_openai_compatible(
                    TOGETHER_API_KEY, 'https://api.together.xyz/v1',
                    model_name, prompt, 'Together'
                )
            except Exception as e:
                print(f"[AI] Together {model_name} error: {e}")
                continue

    # --- 5. Fallback to OpenRouter ---
    if OPENROUTER_API_KEY:
        openrouter_models = ['meta-llama/llama-3.3-70b-instruct:free', 'google/gemini-2.0-flash-exp:free']
        for model_name in openrouter_models:
            try:
                return _call_openai_compatible(
                    OPENROUTER_API_KEY, 'https://openrouter.ai/api/v1',
                    model_name, prompt, 'OpenRouter',
                    extra_headers={'HTTP-Referer': 'http://localhost:5000', 'X-Title': 'InternFinder'}
                )
            except Exception as e:
                print(f"[AI] OpenRouter {model_name} error: {e}")
                continue

    # --- 6. Last resort: retry Gemini with patience ---
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
                        wait_time = 5
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
