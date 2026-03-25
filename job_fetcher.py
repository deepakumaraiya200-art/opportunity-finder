"""
job_fetcher.py — Fetches real job/internship listings from 15 free platforms
Platforms: RemoteOK, Arbeitnow, Unstop, Naukri, LinkedIn/Indeed/Glassdoor (via jobspy),
           Foundit, Remotive, Internshala, EliteCompanies, HackerNews/YC, The Muse,
           Greenhouse ATS, Lever ATS, and optionally JSearch (RapidAPI)
Supports search_type='intern' (internships only) and search_type='job' (full-time/entry-level jobs).
"""

import requests
import os
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, Future
from bs4 import BeautifulSoup

try:
    import cloudscraper
except ImportError:
    cloudscraper = None


# --- Elite / FAANG Companies to track ---
ELITE_COMPANIES = [
    # FAANG / Big Tech
    'Google', 'Microsoft', 'Amazon', 'Apple', 'Meta', 'Netflix',
    'LinkedIn', 'Atlassian', 'Adobe', 'Salesforce', 'SAP Labs',
    'NVIDIA', 'Intel', 'Qualcomm', 'AMD', 'Cisco', 'Oracle', 'IBM',
    'Texas Instruments', 'Samsung',
    # Top Startups & Unicorns
    'Stripe', 'Airbnb', 'Uber', 'Rubrik', 'Snowflake', 'Postman',
    'Intuit', 'Nutanix',
    # Finance / Quant
    'Goldman Sachs', 'JP Morgan', 'JPMorgan Chase', 'Tower Research Capital',
    'D.E. Shaw', 'AlphaGrep',
    # Indian Tech & Unicorns
    'Flipkart', 'Razorpay', 'Swiggy', 'Zerodha', 'Walmart Global Tech',
    # Indian Conglomerates
    'Adani', 'Reliance', 'Tata',
]

# Lowercase set for fast matching
_ELITE_LOWER = {c.lower() for c in ELITE_COMPANIES}


def fetch_all_jobs(skills: list, max_per_source: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch jobs from all free APIs CONCURRENTLY and return a normalized list.
    Each job has: company, role, url, location, tags, source, remote
    
    Naukri runs as a BACKGROUND thread — it never blocks the response.
    It serves stale cache instantly + refreshes in background for next user.
    """
    all_jobs: List[Dict] = []
    futures: Dict[Future, str] = {}

    # --- Naukri: Non-blocking background approach ---
    # 1. Serve any cached Naukri results immediately (even stale)
    # 2. Fire background thread to refresh cache for next user
    try:
        from naukri_scraper import fetch_naukri_on_demand, _read_category_cache, _skills_to_slugs, _is_cache_fresh, _init_scraper_providers, SCRAPER_PROVIDERS, _scrape_and_cache_category
        import threading as _threading
        
        # Serve cached results instantly (fresh or stale — anything is better than nothing)
        slugs = _skills_to_slugs(skills)
        naukri_cached: List[Dict] = []
        stale_slugs: List[str] = []
        for slug in slugs:
            cached = _read_category_cache(slug)
            if cached:
                naukri_cached.extend(cached)
                if not _is_cache_fresh(slug):
                    stale_slugs.append(slug)
            else:
                stale_slugs.append(slug)
        
        if naukri_cached:
            # Filter by skill match
            skills_lower = [s.lower() for s in skills]
            matched_naukri = []
            for job in naukri_cached:
                searchable = f"{job.get('role', '')} {job.get('company', '')} {' '.join(job.get('tags', []))} {job.get('description', '')}".lower()
                if any(any(w in searchable for w in sk.split()) for sk in skills_lower):
                    matched_naukri.append(job)
                    if len(matched_naukri) >= max_per_source:
                        break
            all_jobs.extend(matched_naukri)
            print(f"[Naukri] Served {len(matched_naukri)} jobs from cache instantly")
        
        # Fire background thread to refresh stale categories (non-blocking)
        if stale_slugs:
            def _bg_refresh(slugs_to_refresh, st):
                if not SCRAPER_PROVIDERS:
                    _init_scraper_providers()
                for slug in slugs_to_refresh:
                    try:
                        _scrape_and_cache_category(slug, st)
                        print(f"[Naukri BG] ✅ '{slug}' refreshed for next user")
                    except Exception as e:
                        print(f"[Naukri BG] ❌ '{slug}' failed: {e}")
            
            bg = _threading.Thread(target=_bg_refresh, args=(stale_slugs, search_type), daemon=True, name="naukri-bg")
            bg.start()
            print(f"[Naukri] Background refresh started for: {stale_slugs}")
    except ImportError:
        print("[Naukri] naukri_scraper module not found, skipping")
    except Exception as e:
        print(f"[Naukri] Error: {e}")

    # --- All other job sources: run in ThreadPoolExecutor with timeout ---
    with ThreadPoolExecutor(max_workers=12) as executor:
        # Submit all fetches in parallel (NO Naukri here — it's handled above)
        futures[executor.submit(fetch_remoteok, skills, max_per_source)] = 'RemoteOK'
        futures[executor.submit(fetch_arbeitnow, skills, max_per_source)] = 'Arbeitnow'
        futures[executor.submit(fetch_unstop, skills, max_per_source, search_type)] = 'Unstop'
        futures[executor.submit(fetch_jobspy, skills, max_per_source, search_type)] = 'JobSpy'
        futures[executor.submit(fetch_elite_companies, skills, max_per_source, search_type)] = 'EliteCompanies'
        futures[executor.submit(fetch_foundit, skills, max_per_source, search_type)] = 'Foundit'
        futures[executor.submit(fetch_remotive, skills, max_per_source)] = 'Remotive'
        # New platforms
        futures[executor.submit(fetch_hackernews_jobs, skills, max_per_source)] = 'HackerNews'
        futures[executor.submit(fetch_themuse, skills, max_per_source, search_type)] = 'TheMuse'
        futures[executor.submit(fetch_greenhouse_ats, skills, max_per_source, search_type)] = 'Greenhouse'
        futures[executor.submit(fetch_lever_ats, skills, max_per_source, search_type)] = 'Lever'
        # Internshala is internship-only; skip in job mode
        if search_type == 'intern':
            futures[executor.submit(fetch_internshala, skills, max_per_source)] = 'Internshala'

        rapidapi_key = os.environ.get('RAPIDAPI_KEY')
        if rapidapi_key:
            futures[executor.submit(fetch_jsearch, skills, rapidapi_key, max_per_source)] = 'JSearch'

        try:
            for future in as_completed(futures, timeout=15):
                source = futures[future]
                try:
                    jobs = future.result()
                    all_jobs.extend(jobs)
                    print(f"[{source}] Fetched {len(jobs)} jobs")
                except Exception as e:
                    print(f"[{source}] Error: {e}")
        except TimeoutError:
            print("[Fetcher] Global 15s timeout reached! Proceeding with fetched jobs.")

    # --- Role Filter (mode-dependent) ---
    filtered_jobs = []
    reject_keywords = ['senior', 'lead', 'manager', 'principal', 'staff', 'director', 'vp', 'head']

    if search_type == 'intern':
        # Strict Internship Filter: only keep explicit internship roles
        accept_keywords = ['intern', 'internship', 'co-op', 'trainee', 'fresher', 'apprentice', 'student', 'step']
        for job in all_jobs:
            title_lower = str(job.get('role', '')).lower()
            company_lower = str(job.get('company', '')).lower()
            if any(bad in title_lower for bad in reject_keywords):
                continue
            if any(good in title_lower for good in accept_keywords) or any(good in company_lower for good in accept_keywords):
                filtered_jobs.append(job)
                continue
    else:
        # Job mode: reject senior/leadership roles, accept everything else
        intern_keywords = ['intern', 'internship', 'co-op', 'trainee', 'apprentice']
        for job in all_jobs:
            title_lower = str(job.get('role', '')).lower()
            if any(bad in title_lower for bad in reject_keywords):
                continue
            # Skip pure internship listings in job mode
            if any(iw in title_lower for iw in intern_keywords):
                continue
            filtered_jobs.append(job)

    all_jobs = filtered_jobs
    mode_label = 'internship' if search_type == 'intern' else 'job'
    print(f"[Fetcher] After {mode_label} filtering: {len(all_jobs)} jobs remain")

    # --- Fair Distribution: round-robin across platforms ---
    all_jobs = _fair_distribute(all_jobs, max_per_source)

    # Verify job URLs are reachable
    all_jobs = verify_job_urls(all_jobs)

    return all_jobs


def _fair_distribute(jobs: List[Dict], total_limit: int) -> List[Dict]:
    """
    Distribute jobs fairly across platforms using round-robin.
    No single platform dominates the results.

    If user asks for 10 and 5 platforms have results:
      → 2 from each platform (round-robin).
    If some platforms have fewer, remaining slots go to platforms with extras.
    """
    from collections import defaultdict

    # Group by source
    by_source: dict = defaultdict(list)
    for job in jobs:
        source = job.get('source', 'Unknown')
        by_source[source].append(job)

    sources = list(by_source.keys())
    if not sources:
        return []

    print(f"[Fair Distribute] {len(sources)} platforms: {sources}")
    for s in sources:
        print(f"  {s}: {len(by_source[s])} jobs")

    # Round-robin pick with per-platform cap
    MAX_PER_PLATFORM = 5
    result: List[Dict] = []
    idx = {s: 0 for s in sources}  # pointer per source
    picked_per_source = {s: 0 for s in sources}
    active_sources = list(sources)

    while len(result) < total_limit and active_sources:
        made_progress = False
        for source in list(active_sources):
            if len(result) >= total_limit:
                break
            # Skip if this platform already hit its cap
            if picked_per_source[source] >= MAX_PER_PLATFORM:
                active_sources.remove(source)
                continue
            pool = by_source[source]
            pointer = idx[source]
            if pointer < len(pool):
                result.append(pool[pointer])
                idx[source] = pointer + 1
                picked_per_source[source] += 1
                made_progress = True
            else:
                active_sources.remove(source)
        if not made_progress:
            break

    # Log distribution
    dist: dict = defaultdict(int)
    for j in result:
        dist[j.get('source', '?')] += 1
    print(f"[Fair Distribute] Final: {len(result)} jobs → {dict(dist)}")

    return result


def verify_job_urls(jobs: List[Dict], timeout: float = 1.5) -> List[Dict]:
    """
    Verify job URLs are reachable via HEAD request.
    Removes jobs with dead links (404/500/timeout).
    """
    if not jobs:
        return jobs

    def check_url(job):
        url = job.get('apply_url', '')
        if not url or url == '#':
            return job, False
        try:
            resp = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={'User-Agent': 'OpportunityFinder/1.0'}
            )
            if resp.status_code < 400:
                job['link_verified'] = True
                return job, True
            else:
                print(f"[Verify] Dead link ({resp.status_code}): {url}")
                return job, False
        except Exception:
            # Timeout or connection error — still include but mark as unverified
            job['link_verified'] = False
            return job, True  # Keep it, just unverified

    verified_jobs = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_url, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                job, keep = future.result()
                if keep:
                    verified_jobs.append(job)
            except Exception:
                pass

    print(f"[Verify] {len(verified_jobs)}/{len(jobs)} jobs passed URL verification")
    return verified_jobs


def _matches_skills(text: str, skills: list) -> bool:
    """Check if any skill appears in the text."""
    text_lower = text.lower()
    return any(skill.lower() in text_lower for skill in skills)


_INDIA_KEYWORDS = [
    'india', 'remote', 'worldwide', 'global', 'anywhere',
    'asia', 'apac', 'oceania', 'emea',
    'bangalore', 'bengaluru', 'mumbai', 'delhi', 'hyderabad',
    'pune', 'chennai', 'kolkata', 'noida', 'gurugram', 'gurgaon',
    'ahmedabad', 'jaipur', 'chandigarh', 'lucknow', 'kochi',
    'indore', 'bhopal', 'coimbatore', 'nagpur', 'thiruvananthapuram',
]


def _is_india_accessible(location: str) -> bool:
    """Check if a job location is India-based or remote/worldwide."""
    loc = location.lower()
    if not loc or loc in ('', 'unknown', 'n/a'):
        return True  # Unknown location, keep it
    return any(kw in loc for kw in _INDIA_KEYWORDS)


def fetch_remoteok(skills: list, limit: int = 30) -> List[Dict]:
    """Fetch from RemoteOK API (free, no auth)."""
    resp = requests.get(
        'https://remoteok.com/api',
        headers={'User-Agent': 'OpportunityFinder/1.0'},
        timeout=8
    )
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for item in data[1:]:  # First item is metadata
        title = item.get('position', '')
        company = item.get('company', '')
        description = item.get('description', '')
        tags = item.get('tags', [])

        # Filter: must match at least one skill
        searchable = f"{title} {company} {description} {' '.join(tags)}"
        if not _matches_skills(searchable, skills):
            continue

        # Filter: only keep remote or India-accessible jobs
        loc = item.get('location', '').lower()
        if not _is_india_accessible(loc):
            continue

        jobs.append({
            'company': company,
            'role': title,
            'apply_url': item.get('url', ''),
            'location': item.get('location', 'Remote'),
            'tags': tags[:6],
            'source': 'RemoteOK',
            'source_text': '✅ RemoteOK (Live Listing)',
            'remote': True,
            'description': (description[:200] + '...') if len(description) > 200 else description,
            'salary': item.get('salary', ''),
            'date': item.get('date', ''),
        })

        if len(jobs) >= limit:
            break

    return jobs


def fetch_arbeitnow(skills: list, limit: int = 30) -> List[Dict]:
    """Fetch from Arbeitnow API (free, no auth)."""
    resp = requests.get(
        'https://www.arbeitnow.com/api/job-board-api',
        timeout=8
    )
    resp.raise_for_status()
    data = resp.json().get('data', [])

    jobs = []
    for item in data:
        title = item.get('title', '')
        company = item.get('company_name', '')
        description = item.get('description', '')
        tags = item.get('tags', [])

        # Filter: must match at least one skill
        searchable = f"{title} {company} {description} {' '.join(tags)}"
        if not _matches_skills(searchable, skills):
            continue

        # Filter: only keep remote or India-accessible jobs
        loc = item.get('location', '').lower()
        if not _is_india_accessible(loc):
            continue

        jobs.append({
            'company': company,
            'role': title,
            'apply_url': item.get('url', ''),
            'location': item.get('location', 'Unknown'),
            'tags': tags[:6],
            'source': 'Arbeitnow',
            'source_text': '✅ Arbeitnow (Live Listing)',
            'remote': item.get('remote', False),
            'description': '',
            'salary': '',
            'date': item.get('created_at', ''),
        })

        if len(jobs) >= limit:
            break

    return jobs


def fetch_jsearch(skills: list, api_key: str, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """Fetch from JSearch API on RapidAPI (free tier: 200 req/month)."""
    suffix = 'intern' if search_type == 'intern' else 'developer'
    query = ' OR '.join(skills[:5]) + f' {suffix}'

    resp = requests.get(
        'https://jsearch.p.rapidapi.com/search',
        params={
            'query': query,
            'page': '1',
            'num_pages': '1',
            'date_posted': 'month',
        },
        headers={
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
        },
        timeout=8
    )
    resp.raise_for_status()
    data = resp.json().get('data', [])

    jobs = []
    for item in data:
        jobs.append({
            'company': item.get('employer_name', ''),
            'role': item.get('job_title', ''),
            'apply_url': item.get('job_apply_link', ''),
            'location': f"{item.get('job_city', '')} {item.get('job_country', '')}".strip(),
            'tags': [],
            'source': 'JSearch',
            'source_text': '✅ JSearch (LinkedIn/Indeed/Glassdoor)',
            'remote': item.get('job_is_remote', False),
            'description': (item.get('job_description', '')[:200] + '...'),
            'salary': '',
            'date': item.get('job_posted_at_datetime_utc', ''),
        })

        if len(jobs) >= limit:
            break

    return jobs


def fetch_unstop(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch opportunities from Unstop (formerly Dare2Compete) public API.
    Returns LIVE internships or jobs with real apply URLs.
    """
    jobs = []
    search_query = ' '.join(skills[:3])  # Use top 3 skills as search
    opp_type = 'internships' if search_type == 'intern' else 'jobs'

    try:
        resp = requests.get(
            'https://unstop.com/api/public/opportunity/search-result',
            params={
                'opportunity': opp_type,
                'per_page': min(limit, 15),
                'oppstatus': 'open',
                'search': search_query,
            },
            headers={'User-Agent': 'OpportunityFinder/1.0'},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle paginated response — data is at top level or nested in 'data' dictionary
        listings = []
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                listings = data['data']
            elif 'data' in data and isinstance(data['data'], dict) and 'data' in data['data']:
                listings = data['data']['data']
        elif isinstance(data, list):
            listings = data

        for item in listings:
            title = item.get('title', '')
            org = item.get('organisation', {})
            company = org.get('name', 'Unknown')
            seo_url = item.get('seo_url', '')
            apply_url = seo_url if seo_url else f"https://unstop.com/{item.get('public_url', '')}"

            # Extract skills from the listing
            req_skills = item.get('required_skills', [])
            skill_names = [s.get('skill_name', s.get('skill', '')) for s in req_skills]

            # Extract location
            location_data = item.get('address_with_country_logo', {})
            location = location_data.get('city', '') or location_data.get('state', '') or ''
            region = item.get('region', '')
            if region == 'online':
                location = 'Remote'
            elif not location:
                location = 'India'

            # Extract salary
            job_detail = item.get('jobDetail', {})
            min_salary = job_detail.get('min_salary')
            max_salary = job_detail.get('max_salary')
            salary = ''
            if min_salary and max_salary:
                salary = f"₹{min_salary}-{max_salary}/mo"
            elif max_salary:
                salary = f"Up to ₹{max_salary}/mo"

            # Extract deadline
            regn = item.get('regnRequirements', {})
            remain_days = regn.get('remain_days', '')

            # Check skill match
            listing_text = f"{title} {' '.join(skill_names)} {company}".lower()
            if not _matches_skills(listing_text, skills):
                continue

            tags = skill_names[:5] if skill_names else [s for s in skills[:3]]

            jobs.append({
                'company': company,
                'role': title,
                'apply_url': apply_url,
                'location': location,
                'tags': tags,
                'source': 'Unstop',
                'source_text': '✅ Unstop (Live Listing)',
                'remote': region == 'online',
                'description': '',
                'salary': salary,
                'date': remain_days,
            })

            if len(jobs) >= limit:
                break

    except Exception as e:
        print(f"[Unstop] Error fetching: {e}")

    return jobs


def fetch_jobspy(skills: list, limit: int = 20, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch jobs from LinkedIn, Indeed, and Glassdoor using python-jobspy.
    Naukri is handled separately via fetch_naukri() to avoid captcha issues.
    Returns real job listings with direct apply URLs from multiple platforms.
    """
    jobs = []
    try:
        from jobspy import scrape_jobs

        suffix = 'intern' if search_type == 'intern' else 'developer engineer'
        search_query = f"{' '.join(skills[:3])} {suffix}"

        # --- Batch 1: LinkedIn + Indeed + Glassdoor ---
        try:
            results = scrape_jobs(
                site_name=["linkedin", "indeed", "glassdoor"],
                search_term=search_query,
                location="India",
                results_wanted=min(limit, 20),
                hours_old=168,  # Last 7 days
                country_indeed='India',
            )
            if results is not None and not results.empty:
                for _, row in results.iterrows():
                    job = _parse_jobspy_row(row, skills)
                    if job:
                        jobs.append(job)
                    if len(jobs) >= limit:
                        break
        except Exception as e:
            print(f"[JobSpy] LinkedIn/Indeed/Glassdoor batch error: {e}")

    except ImportError:
        print("[JobSpy] python-jobspy not installed, skipping")
    except Exception as e:
        print(f"[JobSpy] Error: {e}")

    return jobs


def _parse_jobspy_row(row, skills: list) -> dict | None:
    """Parse a single row from python-jobspy DataFrame into a normalized job dict."""
    title = str(row.get('title', ''))
    company = str(row.get('company_name', row.get('company', 'Unknown')))
    job_url = str(row.get('job_url', row.get('link', '#')))
    location = str(row.get('location', ''))
    site = str(row.get('site', 'LinkedIn'))
    is_remote = bool(row.get('is_remote', False))
    description = str(row.get('description', ''))[:200]

    if job_url == 'nan' or not job_url or job_url == '#':
        return None

    # Check skill match
    listing_text = f"{title} {company} {description}".lower()
    if not _matches_skills(listing_text, skills):
        return None

    source_name = site.capitalize() if site != 'nan' else 'LinkedIn'
    return {
        'company': company if company != 'nan' else 'Unknown',
        'role': title if title != 'nan' else 'Internship',
        'apply_url': job_url,
        'location': location if location != 'nan' else '',
        'tags': skills[:3],
        'source': source_name,
        'source_text': f'✅ {source_name} (Live Listing)',
        'remote': is_remote,
        'description': description,
        'salary': '',
        'date': '',
    }


def fetch_elite_companies(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Targeted fetch: search LinkedIn/Indeed specifically for elite/FAANG companies.
    Searches batches of companies combined with the user's skills.
    Results are auto-tagged as 'elite' for tier-S classification.
    """
    jobs = []
    try:
        from jobspy import scrape_jobs

        suffix = 'intern internship' if search_type == 'intern' else 'developer engineer'
        skill_str = ' '.join(skills[:3])

        # Batch companies into groups of 5 for broader search coverage
        company_batches = [ELITE_COMPANIES[i:i+5] for i in range(0, len(ELITE_COMPANIES), 5)]

        for batch in company_batches[:3]:  # Limit to 3 batches (15 companies) to keep it fast
            company_query = ' OR '.join(batch)
            search_query = f"({company_query}) {skill_str} {suffix}"

            try:
                results = scrape_jobs(
                    site_name=["linkedin", "indeed"],
                    search_term=search_query,
                    location="India",
                    results_wanted=min(limit // 3, 10),
                    hours_old=336,  # Last 14 days
                    country_indeed='India',
                )
                if results is not None and not results.empty:
                    for _, row in results.iterrows():
                        company = str(row.get('company_name', row.get('company', '')))
                        # Only keep if company is actually in our elite list
                        if not _is_elite_company(company):
                            continue
                        job = _parse_jobspy_row(row, skills)
                        if job:
                            job['elite'] = True
                            job['source'] = 'EliteCompany'
                            job['source_text'] = f'🏆 {job["company"]} (Career Page)'
                            jobs.append(job)
                        if len(jobs) >= limit:
                            break
            except Exception as e:
                print(f"[EliteCompanies] Batch error for {batch[:2]}: {e}")
                continue

            if len(jobs) >= limit:
                break

    except ImportError:
        print("[EliteCompanies] python-jobspy not installed, skipping")
    except Exception as e:
        print(f"[EliteCompanies] Error: {e}")

    return jobs


def _is_elite_company(company_name: str) -> bool:
    """Check if a company name matches any elite company (fuzzy)."""
    name_lower = company_name.lower().strip()
    # Direct match
    if name_lower in _ELITE_LOWER:
        return True
    # Partial match (e.g., "Google LLC" matches "Google")
    for elite in _ELITE_LOWER:
        if elite in name_lower or name_lower in elite:
            return True
    return False

# ── Internshala skill-to-category mapping ──
_INTERNSHALA_CATEGORIES = {
    'python': 'python-django-development',
    'java': 'java-development',
    'javascript': 'javascript-development',
    'react': 'reactjs-development',
    'node': 'nodejs-development',
    'angular': 'angularjs-development',
    'web development': 'web-development',
    'full stack': 'full-stack-development',
    'frontend': 'front-end-development',
    'backend': 'back-end-development',
    'android': 'android-app-development',
    'ios': 'ios-app-development',
    'flutter': 'flutter-development',
    'react native': 'mobile-app-development',
    'machine learning': 'machine-learning',
    'deep learning': 'artificial-intelligence-ai',
    'ai': 'artificial-intelligence-ai',
    'data science': 'data-science',
    'data analytics': 'data-analytics',
    'cloud': 'cloud-computing',
    'aws': 'cloud-computing',
    'devops': 'devops',
    'cyber security': 'cyber-security',
    'ui/ux': 'ui-ux-design',
    'graphic design': 'graphic-design',
    'digital marketing': 'digital-marketing',
    'content writing': 'content-writing',
    'sql': 'database-management',
    'database': 'database-management',
    'c++': 'programming',
    'c': 'programming',
    'embedded': 'embedded-systems',
    'iot': 'internet-of-things-iot',
    'blockchain': 'blockchain-development',
    'testing': 'software-testing',
    'automation': 'automation-testing',
    'computer science': 'computer-science',
}


def fetch_internshala(skills: list, limit: int = 30) -> List[Dict]:
    """
    Scrape Internshala listing pages for internships matching user skills.
    Uses BeautifulSoup to parse structured listing pages.
    """
    jobs = []

    # Map skills to Internshala category slugs
    categories = set()
    for skill in skills:
        sk = skill.lower().strip()
        if sk in _INTERNSHALA_CATEGORIES:
            categories.add(_INTERNSHALA_CATEGORIES[sk])
        # Also try partial matching
        for key, cat in _INTERNSHALA_CATEGORIES.items():
            if key in sk or sk in key:
                categories.add(cat)

    if not categories:
        categories = {'computer-science'}  # Default fallback

    def _fetch_category(category: str) -> List[Dict]:
        cat_jobs = []
        try:
            url = f'https://internshala.com/internships/{category}-internship'
            resp = requests.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                },
                timeout=8
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find internship cards
            cards = soup.select('.internship_meta, .individual_internship, [class*="internship"]')
            if not cards:
                # Fallback: find links matching internship detail pattern
                links = soup.find_all('a', href=re.compile(r'/internship/detail/'))
                for link in links:
                    href = link.get('href', '')
                    if not href:
                        continue
                    full_url = f"https://internshala.com{href}" if href.startswith('/') else href
                    text = link.get_text(strip=True)
                    if not text or len(text) < 3:
                        continue

                    # Parse role from link text, company from URL
                    role = text
                    company_match = re.search(r'-at-(.+?)(\d{5,})', href)
                    company = company_match.group(1).replace('-', ' ').title() if company_match else 'Unknown'

                    loc_match = re.search(r'-in-(.+?)-at-', href)
                    location = loc_match.group(1).replace('-', ' ').title() if loc_match else ''
                    if 'work-from-home' in href:
                        location = 'Remote'

                    cat_jobs.append({
                        'company': company,
                        'role': role,
                        'apply_url': full_url,
                        'location': location or 'India',
                        'tags': skills[:3],
                        'source': 'Internshala',
                        'source_text': '✅ Internshala (Live Listing)',
                        'remote': 'work-from-home' in href or 'remote' in href.lower(),
                        'description': '',
                        'salary': '',
                        'date': '',
                    })

                    if len(cat_jobs) >= limit:
                        break
                return cat_jobs

            for card in cards:
                title_el = card.select_one('.profile, h3, [class*="profile"]')
                company_el = card.select_one('.company_name, [class*="company"], h4')
                location_el = card.select_one('.location_link, [class*="location"]')
                stipend_el = card.select_one('.stipend, [class*="stipend"]')
                link_el = card.select_one('a[href*="/internship/detail/"]')

                if not link_el:
                    continue

                href = link_el.get('href', '')
                full_url = f"https://internshala.com{href}" if href.startswith('/') else href
                role = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
                company = company_el.get_text(strip=True) if company_el else 'Unknown'
                location = location_el.get_text(strip=True) if location_el else 'India'
                stipend = stipend_el.get_text(strip=True) if stipend_el else ''

                if not role or len(role) < 3:
                    continue

                cat_jobs.append({
                    'company': company,
                    'role': role,
                    'apply_url': full_url,
                    'location': location,
                    'tags': skills[:3],
                    'source': 'Internshala',
                    'source_text': '✅ Internshala (Live Listing)',
                    'remote': 'work from home' in location.lower() or 'remote' in location.lower(),
                    'description': '',
                    'salary': stipend,
                    'date': '',
                })

                if len(cat_jobs) >= limit:
                    break

        except Exception as e:
            print(f"[Internshala] Error fetching {category}: {e}")

        return cat_jobs

    import itertools
    # Fetch up to 3 categories in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_fetch_category, cat) for cat in itertools.islice(categories, 3)]
        for future in as_completed(futures):
            jobs.extend(future.result())
            if len(jobs) >= limit * 2:
                break

    # Deduplicate by URL
    seen = set()
    unique_jobs = []
    for job in jobs:
        if job['apply_url'] not in seen:
            seen.add(job['apply_url'])
            unique_jobs.append(job)

    return unique_jobs


def fetch_foundit(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Scrape job listings from Foundit.in (formerly Monster India).
    Uses BeautifulSoup to parse search result pages.
    """
    jobs = []
    try:
        suffix = 'intern' if search_type == 'intern' else ''
        search_query = ' '.join(skills[:3]) + (f' {suffix}' if suffix else '')
        encoded_query = requests.utils.quote(search_query)

        url = f'https://www.foundit.in/srp/results?query={encoded_query}&locations=India&sort=1'

        resp = requests.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            },
            timeout=10
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Try multiple selectors for job cards on Foundit
        cards = soup.select('.card-apply-content, .jobTuple, [class*="cardWithSalary"], [class*="job-card"]')
        if not cards:
            # Fallback: find all links that look like job detail pages
            links = soup.find_all('a', href=re.compile(r'/job/'))
            seen_urls = set()
            for link in links:
                href = link.get('href', '')
                if not href or href in seen_urls:
                    continue
                seen_urls.add(href)
                full_url = f"https://www.foundit.in{href}" if href.startswith('/') else href
                text = link.get_text(strip=True)
                if not text or len(text) < 3:
                    continue

                listing_text = text.lower()
                if not _matches_skills(listing_text, skills):
                    continue

                jobs.append({
                    'company': 'Unknown',
                    'role': text[:100],
                    'apply_url': full_url,
                    'location': 'India',
                    'tags': skills[:3],
                    'source': 'Foundit',
                    'source_text': '✅ Foundit (Live Listing)',
                    'remote': False,
                    'description': '',
                    'salary': '',
                    'date': '',
                })

                if len(jobs) >= limit:
                    break
            return jobs

        for card in cards:
            # Extract job details from card
            title_el = card.select_one('.title, [class*="title"], h3, h2')
            company_el = card.select_one('.company-name, [class*="company"], .subTitle')
            location_el = card.select_one('.location-text, [class*="location"], .loc')
            salary_el = card.select_one('.salary, [class*="salary"]')
            link_el = card.select_one('a[href*="/job/"]') or card.select_one('a[href]')

            if not link_el:
                continue

            href = link_el.get('href', '')
            full_url = f"https://www.foundit.in{href}" if href.startswith('/') else href
            role = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
            company = company_el.get_text(strip=True) if company_el else 'Unknown'
            location = location_el.get_text(strip=True) if location_el else 'India'
            salary = salary_el.get_text(strip=True) if salary_el else ''

            if not role or len(role) < 3:
                continue

            # Check skill match
            listing_text = f"{role} {company}".lower()
            if not _matches_skills(listing_text, skills):
                continue

            jobs.append({
                'company': company,
                'role': role,
                'apply_url': full_url,
                'location': location,
                'tags': skills[:3],
                'source': 'Foundit',
                'source_text': '✅ Foundit (Live Listing)',
                'remote': 'remote' in location.lower() or 'work from home' in location.lower(),
                'description': '',
                'salary': salary,
                'date': '',
            })

            if len(jobs) >= limit:
                break

    except Exception as e:
        print(f"[Foundit] Error fetching: {e}")

    return jobs


def fetch_remotive(skills: list, limit: int = 30) -> List[Dict]:
    """
    Fetch remote jobs from Remotive.io free API (no auth required).
    Returns real remote job listings with direct apply URLs.
    """
    jobs = []
    try:
        search_query = ' '.join(skills[:3])
        resp = requests.get(
            'https://remotive.com/api/remote-jobs',
            params={
                'search': search_query,
                'limit': min(limit * 2, 100),  # Fetch extra for filtering
            },
            headers={'User-Agent': 'OpportunityFinder/1.0'},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json().get('jobs', [])

        for item in data:
            title = item.get('title', '')
            company = item.get('company_name', '')
            description = item.get('description', '')
            tags = item.get('tags', [])
            candidate_required_location = item.get('candidate_required_location', '')
            job_type = item.get('job_type', '')

            # Filter: must match at least one skill
            searchable = f"{title} {company} {description} {' '.join(tags)}"
            if not _matches_skills(searchable, skills):
                continue

            # Filter: India-accessible locations only
            if candidate_required_location and not _is_india_accessible(candidate_required_location):
                continue

            # Build clean location string
            location = candidate_required_location if candidate_required_location else 'Remote'

            jobs.append({
                'company': company,
                'role': title,
                'apply_url': item.get('url', ''),
                'location': location,
                'tags': tags[:6] if tags else skills[:3],
                'source': 'Remotive',
                'source_text': '✅ Remotive (Live Listing)',
                'remote': True,
                'description': (description[:200] + '...') if len(description) > 200 else description,
                'salary': item.get('salary', ''),
                'date': item.get('publication_date', ''),
            })

            if len(jobs) >= limit:
                break

    except Exception as e:
        print(f"[Remotive] Error fetching: {e}")

    return jobs


# =====================================================================
# NEW PLATFORMS
# =====================================================================

def fetch_hackernews_jobs(skills: list, limit: int = 30) -> List[Dict]:
    """
    Fetch from HackerNews (Y Combinator) Jobs — free Firebase API, no auth.
    These are jobs posted by YC-funded startups directly on HN.
    """
    jobs: List[Dict] = []
    try:
        # Get IDs of job stories
        resp = requests.get(
            'https://hacker-news.firebaseio.com/v0/jobstories.json',
            timeout=8
        )
        resp.raise_for_status()
        story_ids = resp.json()

        # Fetch details for the first N stories (limit API calls)
        max_to_check = min(len(story_ids), 60)

        def _fetch_story(sid: int) -> dict:
            r = requests.get(
                f'https://hacker-news.firebaseio.com/v0/item/{sid}.json',
                timeout=5
            )
            r.raise_for_status()
            return r.json()

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_fetch_story, sid): sid for sid in story_ids[:max_to_check]}
            for future in as_completed(futures, timeout=12):
                try:
                    item = future.result()
                    if not item or item.get('type') != 'job':
                        continue

                    title = item.get('title', '')
                    text = item.get('text', '')
                    hn_url = f"https://news.ycombinator.com/item?id={item.get('id', '')}"

                    # Extract company name from title (usually "Company is hiring" or "Company (YC Sxx)")
                    company = 'YC Startup'
                    if ' is hiring' in title.lower():
                        company = title.split(' is hiring')[0].split(' Is Hiring')[0].strip()
                    elif '(YC' in title:
                        company = title.split('(YC')[0].strip()
                    elif ' – ' in title:
                        company = title.split(' – ')[0].strip()
                    elif ' - ' in title:
                        company = title.split(' - ')[0].strip()

                    # Filter: must match at least one skill
                    searchable = f"{title} {text}"
                    if not _matches_skills(searchable, skills):
                        continue

                    # Try to extract location from text
                    location = 'Remote'
                    text_lower = text.lower() if text else ''
                    if 'remote' in text_lower:
                        location = 'Remote'
                    elif 'india' in text_lower or 'bangalore' in text_lower or 'mumbai' in text_lower:
                        location = 'India'
                    elif not _is_india_accessible(text_lower[:300]):
                        continue  # Skip non-India non-remote jobs

                    jobs.append({
                        'company': company,
                        'role': title[:100],
                        'apply_url': hn_url,
                        'location': location,
                        'tags': skills[:4],
                        'source': 'HackerNews',
                        'source_text': '✅ HackerNews/YC (Live Listing)',
                        'remote': 'remote' in text_lower,
                        'description': (text[:200] + '...') if text and len(text) > 200 else (text or ''),
                        'salary': '',
                        'date': '',
                    })

                    if len(jobs) >= limit:
                        break
                except Exception:
                    continue

    except Exception as e:
        print(f"[HackerNews] Error fetching: {e}")

    return jobs


def fetch_themuse(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch from The Muse API — free public API, no auth needed.
    https://www.themuse.com/developers/api/v2
    """
    jobs: List[Dict] = []
    try:
        # Build category filter based on search type
        level = 'Internship' if search_type == 'intern' else 'Entry Level'
        top_skills = skills[:3] if skills else ['software']

        for skill in top_skills:
            if len(jobs) >= limit:
                break

            params = {
                'page': 1,
                'descending': 'true',
                'level': level,
                'category': 'Software Engineering' if any(
                    s.lower() in ['python', 'java', 'javascript', 'react', 'node', 'sql', 'c++', 'go', 'rust']
                    for s in skills
                ) else 'Data Science',
            }

            resp = requests.get(
                'https://www.themuse.com/api/public/jobs',
                params=params,
                headers={'User-Agent': 'OpportunityFinder/1.0'},
                timeout=8
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            results = data.get('results', [])

            for item in results:
                if len(jobs) >= limit:
                    break

                title = item.get('name', '')
                company_obj = item.get('company', {})
                company = company_obj.get('name', 'Unknown') if isinstance(company_obj, dict) else str(company_obj)
                
                # Get location
                locations = item.get('locations', [])
                loc_str = ', '.join(
                    loc.get('name', '') for loc in locations if isinstance(loc, dict)
                ) if locations else 'Remote'

                # Filter: India-accessible locations only
                if not _is_india_accessible(loc_str):
                    continue

                # Build description from contents
                contents = item.get('contents', '')
                # Strip HTML tags
                if contents:
                    contents = re.sub(r'<[^>]+>', ' ', contents)
                    contents = re.sub(r'\s+', ' ', contents).strip()

                # Filter: must match at least one skill
                searchable = f"{title} {company} {contents[:500]}"
                if not _matches_skills(searchable, skills):
                    continue

                apply_url = item.get('refs', {}).get('landing_page', '') if isinstance(item.get('refs'), dict) else ''

                jobs.append({
                    'company': company,
                    'role': title,
                    'apply_url': apply_url or f"https://www.themuse.com/jobs/{item.get('id', '')}",
                    'location': loc_str if loc_str else 'Remote',
                    'tags': [cat.get('name', '') for cat in item.get('categories', []) if isinstance(cat, dict)][:4] or skills[:3],
                    'source': 'TheMuse',
                    'source_text': '✅ The Muse (Live Listing)',
                    'remote': 'remote' in loc_str.lower() or 'flexible' in loc_str.lower(),
                    'description': (contents[:200] + '...') if contents and len(contents) > 200 else (contents or ''),
                    'salary': '',
                    'date': item.get('publication_date', ''),
                })

    except Exception as e:
        print(f"[TheMuse] Error fetching: {e}")

    return jobs


# --- ATS (Applicant Tracking System) Scrapers ---
# Many elite companies host their career pages on Greenhouse or Lever.
# We scrape these JSON APIs directly for the freshest listings.

# Mapping of elite companies to their Greenhouse board tokens
_GREENHOUSE_BOARDS = {
    'Airbnb': 'airbnb',
    'Stripe': 'stripe',
    'Cloudflare': 'cloudflare',
    'Coinbase': 'coinbase',
    'DoorDash': 'doordash',
    'Figma': 'figma',
    'Notion': 'notion',
    'Discord': 'discord',
    'Roblox': 'roblox',
    'Databricks': 'databricks',
    'Palantir': 'palantir',
    'Snyk': 'snyk',
    'HashiCorp': 'hashicorp',
    'Rubrik': 'rubrikofficial',
    'Postman': 'postman',
}

# Mapping of elite companies to their Lever board tokens
_LEVER_BOARDS = {
    'Netflix': 'netflix',
    'Snowflake': 'snowflakecomputing',
    'Nutanix': 'nutanix',
    'CRED': 'cred',
    'Razorpay': 'razorpay',
    'Meesho': 'meesho',
    'Ola': 'olacabs',
}


def fetch_greenhouse_ats(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch jobs directly from Greenhouse ATS boards for elite companies.
    Uses the free public JSON API: https://boards-api.greenhouse.io/v1/boards/{token}/jobs
    """
    jobs: List[Dict] = []

    def _fetch_board(company: str, token: str) -> List[Dict]:
        board_jobs = []
        try:
            resp = requests.get(
                f'https://boards-api.greenhouse.io/v1/boards/{token}/jobs',
                params={'content': 'true'},
                headers={'User-Agent': 'OpportunityFinder/1.0'},
                timeout=8
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            for item in data.get('jobs', []):
                title = item.get('title', '')
                title_lower = title.lower()

                # Filter by search type
                if search_type == 'intern':
                    if not any(kw in title_lower for kw in ['intern', 'co-op', 'trainee', 'fresher', 'new grad', 'graduate']):
                        continue
                else:
                    if any(kw in title_lower for kw in ['senior', 'staff', 'principal', 'director', 'vp', 'head', 'lead', 'manager']):
                        continue

                # Get location
                location_obj = item.get('location', {})
                location = location_obj.get('name', 'Remote') if isinstance(location_obj, dict) else 'Remote'

                if not _is_india_accessible(location):
                    continue

                # Get description text
                content = item.get('content', '')
                if content:
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = re.sub(r'\s+', ' ', content).strip()

                # Check skill match
                searchable = f"{title} {content[:500]}"
                if not _matches_skills(searchable, skills):
                    continue

                apply_url = item.get('absolute_url', '')

                board_jobs.append({
                    'company': company,
                    'role': title,
                    'apply_url': apply_url,
                    'location': location,
                    'tags': skills[:4],
                    'source': 'Greenhouse',
                    'source_text': f'✅ {company} Careers (Greenhouse)',
                    'remote': 'remote' in location.lower(),
                    'description': (content[:200] + '...') if content and len(content) > 200 else (content or ''),
                    'salary': '',
                    'date': item.get('updated_at', ''),
                })

            return board_jobs

        except Exception as e:
            print(f"[Greenhouse/{company}] Error: {e}")
            return []

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_fetch_board, company, token): company
                for company, token in _GREENHOUSE_BOARDS.items()
            }
            for future in as_completed(futures, timeout=12):
                try:
                    board_jobs = future.result()
                    jobs.extend(board_jobs)
                except Exception:
                    continue
    except TimeoutError:
        print("[Greenhouse] Timeout reached, proceeding with fetched results")

    return jobs[:limit]


def fetch_lever_ats(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch jobs directly from Lever ATS boards for elite companies.
    Uses the free public JSON API: https://api.lever.co/v0/postings/{token}
    """
    jobs: List[Dict] = []

    def _fetch_board(company: str, token: str) -> List[Dict]:
        board_jobs = []
        try:
            resp = requests.get(
                f'https://api.lever.co/v0/postings/{token}',
                params={'mode': 'json'},
                headers={'User-Agent': 'OpportunityFinder/1.0'},
                timeout=8
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            if not isinstance(data, list):
                return []

            for item in data:
                title = item.get('text', '')
                title_lower = title.lower()

                # Filter by search type
                if search_type == 'intern':
                    if not any(kw in title_lower for kw in ['intern', 'co-op', 'trainee', 'fresher', 'new grad', 'graduate']):
                        continue
                else:
                    if any(kw in title_lower for kw in ['senior', 'staff', 'principal', 'director', 'vp', 'head', 'lead', 'manager']):
                        continue

                # Get location from categories
                categories = item.get('categories', {})
                location = categories.get('location', 'Remote') if isinstance(categories, dict) else 'Remote'
                team = categories.get('team', '') if isinstance(categories, dict) else ''

                if not _is_india_accessible(location):
                    continue

                # Get description
                desc_plain = item.get('descriptionPlain', '')
                additional_plain = item.get('additionalPlain', '')
                full_text = f"{desc_plain} {additional_plain}"

                # Check skill match
                searchable = f"{title} {team} {full_text[:500]}"
                if not _matches_skills(searchable, skills):
                    continue

                apply_url = item.get('hostedUrl', '') or item.get('applyUrl', '')

                board_jobs.append({
                    'company': company,
                    'role': title,
                    'apply_url': apply_url,
                    'location': location,
                    'tags': [team] + skills[:3] if team else skills[:4],
                    'source': 'Lever',
                    'source_text': f'✅ {company} Careers (Lever)',
                    'remote': 'remote' in location.lower(),
                    'description': (full_text[:200] + '...') if len(full_text) > 200 else full_text,
                    'salary': '',
                    'date': '',
                })

            return board_jobs

        except Exception as e:
            print(f"[Lever/{company}] Error: {e}")
            return []

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(_fetch_board, company, token): company
                for company, token in _LEVER_BOARDS.items()
            }
            for future in as_completed(futures, timeout=12):
                try:
                    board_jobs = future.result()
                    jobs.extend(board_jobs)
                except Exception:
                    continue
    except TimeoutError:
        print("[Lever] Timeout reached, proceeding with fetched results")

    return jobs[:limit]


def fetch_naukri(skills: list, limit: int = 30, search_type: str = 'intern') -> List[Dict]:
    """
    Fetch Naukri.com jobs ON-DEMAND.
    Only scrapes categories relevant to the user's skills.
    Uses per-category 24hr cache to avoid redundant API calls.
    Runs inside the ThreadPoolExecutor — parallel with other job sources.
    """
    try:
        from naukri_scraper import fetch_naukri_on_demand
        return fetch_naukri_on_demand(skills, limit, search_type)
    except ImportError:
        print("[Naukri] naukri_scraper module not found")
        return []
    except Exception as e:
        print(f"[Naukri] On-demand fetch error: {e}")
        return []


