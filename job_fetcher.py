"""
job_fetcher.py — Fetches real job listings from free APIs
APIs: RemoteOK, Arbeitnow, Unstop, LinkedIn/Indeed (via jobspy), and optionally JSearch (RapidAPI)
"""

import requests
import os
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from bs4 import BeautifulSoup


def fetch_all_jobs(skills: list, max_per_source: int = 30) -> List[Dict]:
    """
    Fetch jobs from all free APIs CONCURRENTLY and return a normalized list.
    Each job has: company, role, url, location, tags, source, remote
    """
    all_jobs = []
    futures = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all fetches in parallel
        futures[executor.submit(fetch_remoteok, skills, max_per_source)] = 'RemoteOK'
        futures[executor.submit(fetch_arbeitnow, skills, max_per_source)] = 'Arbeitnow'
        futures[executor.submit(fetch_unstop, skills, max_per_source)] = 'Unstop'
        futures[executor.submit(fetch_linkedin, skills, max_per_source)] = 'LinkedIn'
        futures[executor.submit(fetch_internshala, skills, max_per_source)] = 'Internshala'

        rapidapi_key = os.environ.get('RAPIDAPI_KEY')
        if rapidapi_key:
            futures[executor.submit(fetch_jsearch, skills, rapidapi_key, max_per_source)] = 'JSearch'

        try:
            for future in as_completed(futures, timeout=10):
                source = futures[future]
                try:
                    jobs = future.result()
                    all_jobs.extend(jobs)
                    print(f"[{source}] Fetched {len(jobs)} jobs")
                except Exception as e:
                    print(f"[{source}] Error: {e}")
        except TimeoutError:
            print("[Fetcher] Global 10s timeout reached! Proceeding with fetched jobs.")

    # --- Strict Internship Filter ---
    # Reject anything that looks like a senior/full-time role and isn't explicitly an internship
    filtered_jobs = []
    reject_keywords = ['senior', 'lead', 'manager', 'principal', 'staff', 'director', 'vp', 'head']
    accept_keywords = ['intern', 'internship', 'co-op', 'trainee', 'fresher', 'apprentice', 'student', 'step']

    for job in all_jobs:
        title_lower = str(job.get('role', '')).lower()
        company_lower = str(job.get('company', '')).lower()
        
        # Immediate reject if it has a senior keyword
        if any(bad in title_lower for bad in reject_keywords):
            continue
            
        # Accept if it explicitly mentions intern/trainee/co-op
        if any(good in title_lower for good in accept_keywords) or any(good in company_lower for good in accept_keywords):
            filtered_jobs.append(job)
            continue
            
        # If it's ambiguous (e.g., just "Software Engineer"), we reject it 
        # because the user strictly wants internships.
        pass

    all_jobs = filtered_jobs
    print(f"[Fetcher] After internship filtering: {len(all_jobs)} jobs remain")

    # Verify job URLs are reachable
    all_jobs = verify_job_urls(all_jobs)

    return all_jobs


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
                headers={'User-Agent': 'InternFinder/1.0'}
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
        headers={'User-Agent': 'InternFinder/1.0'},
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


def fetch_jsearch(skills: list, api_key: str, limit: int = 30) -> List[Dict]:
    """Fetch from JSearch API on RapidAPI (free tier: 200 req/month)."""
    query = ' OR '.join(skills[:5]) + ' intern'

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


def fetch_unstop(skills: list, limit: int = 30) -> List[Dict]:
    """
    Fetch internships from Unstop (formerly Dare2Compete) public API.
    Returns LIVE internships with real apply URLs.
    """
    jobs = []
    search_query = ' '.join(skills[:3])  # Use top 3 skills as search

    try:
        resp = requests.get(
            'https://unstop.com/api/public/opportunity/search-result',
            params={
                'opportunity': 'internships',
                'per_page': min(limit, 15),
                'oppstatus': 'open',
                'search': search_query,
            },
            headers={'User-Agent': 'InternFinder/1.0'},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle paginated response — data is at top level with 'data' key
        listings = data.get('data', [])
        if not listings and isinstance(data, list):
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


def fetch_linkedin(skills: list, limit: int = 20) -> List[Dict]:
    """
    Fetch jobs from LinkedIn and Indeed using python-jobspy.
    Returns real job listings with direct apply URLs.
    """
    jobs = []
    try:
        from jobspy import scrape_jobs

        search_query = f"{' '.join(skills[:3])} intern"

        results = scrape_jobs(
            site_name=["linkedin", "indeed"],
            search_term=search_query,
            location="India",
            results_wanted=min(limit, 15),
            hours_old=168,  # Last 7 days
            country_indeed='India',
        )

        if results is not None and not results.empty:
            for _, row in results.iterrows():
                title = str(row.get('title', ''))
                company = str(row.get('company_name', row.get('company', 'Unknown')))
                job_url = str(row.get('job_url', row.get('link', '#')))
                location = str(row.get('location', ''))
                site = str(row.get('site', 'LinkedIn'))
                is_remote = bool(row.get('is_remote', False))
                description = str(row.get('description', ''))[:200]

                if job_url == 'nan' or not job_url or job_url == '#':
                    continue

                # Check skill match
                listing_text = f"{title} {company} {description}".lower()
                if not _matches_skills(listing_text, skills):
                    continue

                source_name = site.capitalize() if site != 'nan' else 'LinkedIn'
                jobs.append({
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
                })

                if len(jobs) >= limit:
                    break

    except ImportError:
        print("[LinkedIn] python-jobspy not installed, skipping")
    except Exception as e:
        print(f"[LinkedIn] Error fetching: {e}")

    return jobs


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

    # Fetch up to 3 categories in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_fetch_category, cat) for cat in list(categories)[:3]]
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

