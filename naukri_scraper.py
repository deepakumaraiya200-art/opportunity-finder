"""
naukri_scraper.py — Background Naukri scraper using ScraperAPI

Scrapes Naukri.com job listings every 6 hours via ScraperAPI (free tier: 1000 calls/month).
Stores results in a local JSON cache. Live user requests read from cache instantly.

Usage:
    - As a module: import and call get_cached_naukri_jobs(skills)
    - Standalone:  python naukri_scraper.py   (runs one scrape cycle)
    - Background:  start_background_scraper() (runs in a daemon thread every 6 hrs)

Requires: SCRAPERAPI_KEY env variable (get free key at scraperapi.com)
"""

import os
import json
import time
import re
import threading
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup

# ---------- Configuration ----------

CACHE_DIR = Path(__file__).parent / "naukri_cache"
CACHE_FILE = CACHE_DIR / "jobs.json"
SCRAPE_INTERVAL = 24 * 60 * 60  # 24 hrs (16 categories × 2 types/day × 30 = 960 calls/mo)

# Skill categories to scrape (8 categories × 4 scrapes/day × 30 days = 960 calls/month)
SKILL_CATEGORIES = [
    # Tech
    "python-developer",
    "backend-developer",
    "frontend-developer",
    "data-science",
    "machine-learning",
    "java-developer",
    "react-developer",
    "devops",
    # Non-Tech / Business
    "management-consulting",
    "finance",
    "accounting",
    "marketing",
    "human-resources",
    "sales",
    "business-analyst",
    "content-writer",
]

# ---------- Scraper Fallback Chain ----------
# Providers tried in order: ScraperAPI → ScrapingBee → ScrapingDog → Direct

SCRAPER_PROVIDERS = []  # Populated at module load based on env keys


def _init_scraper_providers():
    """Initialize the scraper fallback chain based on available API keys."""
    global SCRAPER_PROVIDERS
    SCRAPER_PROVIDERS = []

    scraperapi_key = os.environ.get("SCRAPERAPI_KEY", "")
    scrapingbee_key = os.environ.get("SCRAPINGBEE_KEY", "")
    scrapingdog_key = os.environ.get("SCRAPINGDOG_KEY", "")

    if scraperapi_key:
        SCRAPER_PROVIDERS.append({
            "name": "ScraperAPI",
            "key": scraperapi_key,
            "fetch": lambda url, key: _fetch_via_scraperapi(url, key),
        })
    if scrapingbee_key:
        SCRAPER_PROVIDERS.append({
            "name": "ScrapingBee",
            "key": scrapingbee_key,
            "fetch": lambda url, key: _fetch_via_scrapingbee(url, key),
        })
    if scrapingdog_key:
        SCRAPER_PROVIDERS.append({
            "name": "ScrapingDog",
            "key": scrapingdog_key,
            "fetch": lambda url, key: _fetch_via_scrapingdog(url, key),
        })

    # Always add direct request as last resort
    SCRAPER_PROVIDERS.append({
        "name": "Direct",
        "key": "",
        "fetch": lambda url, key: _fetch_direct(url),
    })

    names = [p["name"] for p in SCRAPER_PROVIDERS]
    print(f"[Scraper Chain] Initialized: {' → '.join(names)}")


def _fetch_via_scraperapi(url: str, api_key: str) -> str:
    """Fetch URL via ScraperAPI (handles JS + CAPTCHAs)."""
    resp = requests.get("https://api.scraperapi.com", params={
        "api_key": api_key,
        "url": url,
        "render": "true",
        "country_code": "in",
    }, timeout=60)
    resp.raise_for_status()
    return resp.text


def _fetch_via_scrapingbee(url: str, api_key: str) -> str:
    """Fetch URL via ScrapingBee (JS rendering + stealth proxy)."""
    resp = requests.get("https://app.scrapingbee.com/api/v1/", params={
        "api_key": api_key,
        "url": url,
        "render_js": "true",
        "country_code": "in",
    }, timeout=60)
    resp.raise_for_status()
    return resp.text


def _fetch_via_scrapingdog(url: str, api_key: str) -> str:
    """Fetch URL via ScrapingDog (JS rendering + proxy rotation)."""
    resp = requests.get("https://api.scrapingdog.com/scrape", params={
        "api_key": api_key,
        "url": url,
        "dynamic": "true",
    }, timeout=60)
    resp.raise_for_status()
    return resp.text


def _fetch_direct(url: str) -> str:
    """Direct request with realistic browser headers (last resort)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text


def _scrape_naukri_page(skill_slug: str, api_key: str, page_type: str = "internship") -> List[Dict]:
    """
    Scrape a single Naukri search page using fallback chain.
    Tries each provider in order until one succeeds.
    """
    naukri_url = f"https://www.naukri.com/{skill_slug}-{page_type}-jobs"
    print(f"[Naukri Scraper] Fetching: {naukri_url}")

    # Initialize providers if not done yet
    if not SCRAPER_PROVIDERS:
        _init_scraper_providers()

    for provider in SCRAPER_PROVIDERS:
        try:
            html = provider["fetch"](naukri_url, provider["key"])
            jobs = _parse_naukri_html(html, skill_slug)
            if jobs:
                print(f"[Naukri Scraper] ✅ {provider['name']} returned {len(jobs)} jobs for '{skill_slug}'")
                return jobs
            # Provider returned HTML but no jobs parsed — try next
            print(f"[Naukri Scraper] {provider['name']} returned HTML but 0 jobs parsed, trying next...")
        except requests.Timeout:
            print(f"[Naukri Scraper] {provider['name']} timed out for '{skill_slug}', trying next...")
        except requests.HTTPError as e:
            print(f"[Naukri Scraper] {provider['name']} HTTP {e.response.status_code} for '{skill_slug}', trying next...")
        except Exception as e:
            print(f"[Naukri Scraper] {provider['name']} error for '{skill_slug}': {e}, trying next...")

    print(f"[Naukri Scraper] All providers failed for '{skill_slug}'")
    return []


def _parse_naukri_html(html: str, skill_slug: str) -> List[Dict]:
    """
    Parse Naukri search results HTML from ScraperAPI.
    Naukri renders job cards with multiple possible class patterns.
    """
    jobs = []
    soup = BeautifulSoup(html, "html.parser")

    # --- Strategy 1: Modern Naukri cards ---
    cards = soup.select(
        "[class*='srp-jobtuple-wrapper'], "
        "[class*='jobTuple'], "
        "[class*='cust-job-tuple'], "
        "article[class*='job']"
    )

    for card in cards:
        job = _extract_from_card(card, skill_slug)
        if job:
            jobs.append(job)

    # --- Strategy 2: Fallback to any links pointing to job pages ---
    if not cards:
        links = soup.find_all("a", href=re.compile(r"/job-listings-|/job/"))
        seen = set()
        for link in links:
            href = link.get("href", "")
            if not href or href in seen:
                continue
            seen.add(href)

            full_url = f"https://www.naukri.com{href}" if href.startswith("/") else href
            text = link.get_text(strip=True)
            if not text or len(text) < 5:
                continue

            jobs.append({
                "company": "Unknown",
                "role": text[:100],
                "apply_url": full_url,
                "location": "India",
                "tags": [skill_slug.replace("-", " ")],
                "source": "Naukri",
                "source_text": "✅ Naukri (Cached)",
                "remote": False,
                "description": "",
                "salary": "",
                "date": "",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            })

    # --- Strategy 3: Parse LD+JSON structured data if available ---
    if not jobs:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "JobPosting":
                    jobs.append(_ld_json_to_job(data, skill_slug))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") == "JobPosting":
                            jobs.append(_ld_json_to_job(item, skill_slug))
            except (json.JSONDecodeError, TypeError):
                continue

    print(f"[Naukri Scraper] Parsed {len(jobs)} jobs from '{skill_slug}'")
    return jobs


def _extract_from_card(card, skill_slug: str) -> Dict[str, Any] | None:
    """Extract job details from a single Naukri card element."""
    title_el = card.select_one("[class*='title'], a[class*='title'], h2 a")
    company_el = card.select_one("[class*='comp-name'], [class*='subTitle'], [class*='company']")
    location_el = card.select_one("[class*='loc'], [class*='location']")
    salary_el = card.select_one("[class*='sal'], [class*='salary']")
    link_el = card.select_one("a[href*='/job-listings'], a[href*='/job/']") or card.select_one("a[href]")

    if not link_el:
        return None

    href = link_el.get("href", "")
    full_url = f"https://www.naukri.com{href}" if href.startswith("/") else href

    role = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
    company = company_el.get_text(strip=True) if company_el else "Unknown"
    location = location_el.get_text(strip=True) if location_el else "India"
    salary = salary_el.get_text(strip=True) if salary_el else ""

    if not role or len(role) < 3:
        return None

    return {
        "company": company,
        "role": role[:100],
        "apply_url": full_url,
        "location": location,
        "tags": [skill_slug.replace("-", " ")],
        "source": "Naukri",
        "source_text": "✅ Naukri (Cached)",
        "remote": any(kw in location.lower() for kw in ["remote", "work from home", "wfh"]),
        "description": "",
        "salary": salary,
        "date": "",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _ld_json_to_job(data: dict, skill_slug: str) -> Dict[str, Any]:
    """Convert a JobPosting LD+JSON schema to our job dict format."""
    org = data.get("hiringOrganization", {})
    loc = data.get("jobLocation", {})
    address = loc.get("address", {}) if isinstance(loc, dict) else {}

    return {
        "company": org.get("name", "Unknown") if isinstance(org, dict) else "Unknown",
        "role": data.get("title", "Unknown Role")[:100],
        "apply_url": data.get("url", "#"),
        "location": address.get("addressLocality", "India") if isinstance(address, dict) else "India",
        "tags": [skill_slug.replace("-", " ")],
        "source": "Naukri",
        "source_text": "✅ Naukri (Cached)",
        "remote": False,
        "description": data.get("description", "")[:200],
        "salary": "",
        "date": data.get("datePosted", ""),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------- On-Demand Scraping (No Background Daemon) ----------

# Map user skills → Naukri category slugs
SKILL_TO_SLUG = {
    "python": "python-developer",
    "django": "python-developer",
    "flask": "python-developer",
    "fastapi": "python-developer",
    "java": "java-developer",
    "spring": "java-developer",
    "react": "react-developer",
    "reactjs": "react-developer",
    "react.js": "react-developer",
    "angular": "frontend-developer",
    "vue": "frontend-developer",
    "html": "frontend-developer",
    "css": "frontend-developer",
    "javascript": "frontend-developer",
    "typescript": "frontend-developer",
    "node": "backend-developer",
    "nodejs": "backend-developer",
    "express": "backend-developer",
    "sql": "data-science",
    "mysql": "data-science",
    "postgresql": "data-science",
    "mongodb": "backend-developer",
    "machine learning": "machine-learning",
    "deep learning": "machine-learning",
    "tensorflow": "machine-learning",
    "pytorch": "machine-learning",
    "data science": "data-science",
    "pandas": "data-science",
    "numpy": "data-science",
    "data analysis": "data-science",
    "power bi": "data-science",
    "tableau": "data-science",
    "excel": "data-science",
    "docker": "devops",
    "kubernetes": "devops",
    "aws": "devops",
    "azure": "devops",
    "gcp": "devops",
    "ci/cd": "devops",
    "terraform": "devops",
    "marketing": "marketing",
    "seo": "marketing",
    "digital marketing": "marketing",
    "content writing": "content-writer",
    "copywriting": "content-writer",
    "finance": "finance",
    "accounting": "accounting",
    "sales": "sales",
    "hr": "human-resources",
    "human resources": "human-resources",
    "business analysis": "business-analyst",
    "business analyst": "business-analyst",
    "consulting": "management-consulting",
}

CACHE_TTL_SECONDS = 12 * 60 * 60  # 12 hours per category


def _get_category_cache_file(slug: str) -> Path:
    """Get the cache file path for a specific category slug."""
    return CACHE_DIR / f"{slug}.json"


def _is_cache_fresh(slug: str) -> bool:
    """Check if the cache for a category is still fresh (< 24 hours old)."""
    cache_file = _get_category_cache_file(slug)
    if not cache_file.exists():
        return False
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        scraped_at = data.get("scraped_at", "")
        if not scraped_at:
            return False
        scraped_dt = datetime.fromisoformat(scraped_at)
        age_seconds = (datetime.now(timezone.utc) - scraped_dt).total_seconds()
        return age_seconds < CACHE_TTL_SECONDS
    except (json.JSONDecodeError, IOError, ValueError):
        return False


def _read_category_cache(slug: str) -> List[Dict]:
    """Read cached jobs for a specific category slug."""
    cache_file = _get_category_cache_file(slug)
    if not cache_file.exists():
        return []
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data.get("jobs", [])
    except (json.JSONDecodeError, IOError):
        return []


def _scrape_and_cache_category(slug: str, search_type: str = "intern") -> List[Dict]:
    """Scrape a single category and save to per-category cache file."""
    api_key = os.environ.get("SCRAPERAPI_KEY", "")
    all_jobs: List[Dict] = []

    # Determine page types based on search_type
    if search_type == "intern":
        page_types = ["internship", "jobs"]
    else:
        page_types = ["jobs", "internship"]

    for page_type in page_types:
        jobs = _scrape_naukri_page(slug, api_key, page_type=page_type)
        all_jobs.extend(jobs)

    # Deduplicate by URL
    seen = set()
    unique = []
    for job in all_jobs:
        url = job.get("apply_url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(job)

    # Save to per-category cache
    CACHE_DIR.mkdir(exist_ok=True)
    cache_data = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "slug": slug,
        "total_jobs": len(unique),
        "jobs": unique,
    }
    cache_file = _get_category_cache_file(slug)
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"[Naukri] Cache write error for {slug}: {e}")

    return unique


def _skills_to_slugs(skills: list) -> List[str]:
    """Map user skills to relevant Naukri category slugs. Exact matches only."""
    slugs = set()
    for skill in skills:
        skill_lower = skill.lower().strip()
        # Direct match only — no fuzzy/partial matching
        if skill_lower in SKILL_TO_SLUG:
            slugs.add(SKILL_TO_SLUG[skill_lower])

    # If no matches, infer from common dev skills
    if not slugs:
        slugs = {"python-developer", "frontend-developer"}

    # Cap at 3 categories max to avoid excessive API calls
    return list(slugs)[:3]


def fetch_naukri_on_demand(skills: list, limit: int = 30, search_type: str = "intern") -> List[Dict]:
    """
    On-demand Naukri scraper: only scrapes categories relevant to the user's skills.
    Uses per-category cache with 12-hour TTL to avoid redundant API calls.
    Scrapes categories IN PARALLEL for speed.
    """
    # Initialize scraper providers if not done
    if not SCRAPER_PROVIDERS:
        _init_scraper_providers()

    slugs = _skills_to_slugs(skills)
    print(f"[Naukri On-Demand] Skills {skills[:5]} → Slugs: {slugs}")

    all_jobs: List[Dict] = []

    # Separate fresh (cached) vs stale (need scraping)
    fresh_slugs = []
    stale_slugs = []
    for slug in slugs:
        if _is_cache_fresh(slug):
            fresh_slugs.append(slug)
        else:
            stale_slugs.append(slug)

    # Serve cached immediately
    for slug in fresh_slugs:
        cached = _read_category_cache(slug)
        print(f"[Naukri On-Demand] ✅ '{slug}' from cache ({len(cached)} jobs, <12h old)")
        all_jobs.extend(cached)

    # Scrape stale categories IN PARALLEL (not sequentially!)
    if stale_slugs:
        print(f"[Naukri On-Demand] 🔄 Scraping {len(stale_slugs)} categories in parallel: {stale_slugs}")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(stale_slugs)) as executor:
            future_to_slug = {
                executor.submit(_scrape_and_cache_category, slug, search_type): slug
                for slug in stale_slugs
            }
            for future in as_completed(future_to_slug, timeout=30):
                slug = future_to_slug[future]
                try:
                    jobs = future.result()
                    all_jobs.extend(jobs)
                    print(f"[Naukri On-Demand] ✅ '{slug}' scraped ({len(jobs)} jobs)")
                except Exception as e:
                    print(f"[Naukri On-Demand] ❌ '{slug}' failed: {e}")

    # Filter by skill match
    skills_lower = [s.lower() for s in skills]
    matched = []
    for job in all_jobs:
        searchable = f"{job.get('role', '')} {job.get('company', '')} {' '.join(job.get('tags', []))} {job.get('description', '')}".lower()
        if any(
            any(word in searchable for word in skill.split())
            for skill in skills_lower
        ):
            matched.append(job)
        if len(matched) >= limit:
            break

    print(f"[Naukri On-Demand] Returning {len(matched)}/{len(all_jobs)} jobs matching user skills")
    return matched


# ---------- Standalone Execution ----------

if __name__ == "__main__":
    print("Running standalone on-demand scrape test...")
    test_skills = ["Python", "React", "Data Science"]
    results = fetch_naukri_on_demand(test_skills, limit=20)
    print(f"\nTotal results: {len(results)}")
    for job in results[:5]:
        print(f"  - {job.get('company', '?')} | {job.get('role', '?')}")

