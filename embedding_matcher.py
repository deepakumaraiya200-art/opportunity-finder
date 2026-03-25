"""
OpportunityFinder — Embedding-Based Vector Matcher
Uses Gemini text-embedding-004 to rank jobs by semantic similarity to resume.
"""

import numpy as np
import google.generativeai as genai
import time


def get_embeddings(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY", max_retries: int = 3) -> list[list[float]]:
    """
    Batch-embed a list of texts using Gemini text-embedding-004.
    Returns a list of 768-dim vectors (one per input text).
    Falls back to retry on rate limits.
    """
    # Gemini embed_content supports batch via list input
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type=task_type,
            )
            # result['embedding'] is a list of vectors when input is a list
            embeddings = result['embedding']
            return embeddings
        except Exception as e:
            err = str(e).lower()
            if '429' in err or 'quota' in err or 'exhausted' in err:
                wait = 5 * (attempt + 1)
                print(f"[Embedding] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"[Embedding] Error: {e}")
                raise
    raise Exception("Embedding API failed after retries")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _job_to_text(job: dict) -> str:
    """Convert a job dict to a searchable text string for embedding."""
    parts = []
    if job.get('role'):
        parts.append(job['role'])
    if job.get('company'):
        parts.append(f"at {job['company']}")
    if job.get('tags'):
        parts.append(f"Skills: {', '.join(job['tags'])}")
    if job.get('location'):
        parts.append(f"Location: {job['location']}")
    if job.get('description'):
        # Use first 200 chars of description if available
        parts.append(job['description'][:200])
    return '. '.join(parts)


def rank_jobs_by_similarity(resume_text: str, jobs: list[dict], top_n: int = 25) -> list[dict]:
    """
    Rank all fetched jobs against the resume using embedding cosine similarity.
    Returns the top_n most relevant jobs, each with a 'similarity_score' field added.
    
    Pipeline:
    1. Embed the resume text
    2. Embed all job descriptions (batch)
    3. Compute cosine similarity for each
    4. Sort descending, return top_n
    """
    if not jobs:
        return []

    t0 = time.time()

    # Prepare texts
    job_texts = [_job_to_text(job) for job in jobs]
    
    # Truncate resume to ~2000 chars to stay within embedding limits
    resume_truncated = resume_text[:2000]

    # Batch embed: resume + all jobs in one API call (saves quota)
    # Gemini batch limit is ~100 texts, split if needed
    BATCH_SIZE = 90
    all_texts = [resume_truncated] + job_texts
    
    all_embeddings = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings(batch, task_type="SEMANTIC_SIMILARITY")
        all_embeddings.extend(batch_embeddings)

    resume_embedding = all_embeddings[0]
    job_embeddings = all_embeddings[1:]

    # Compute similarities
    scored_jobs = []
    for job, job_emb in zip(jobs, job_embeddings):
        sim = cosine_similarity(resume_embedding, job_emb)
        job_copy = dict(job)
        job_copy['similarity_score'] = round(sim, 4)
        scored_jobs.append(job_copy)

    # Sort by similarity descending
    scored_jobs.sort(key=lambda x: x['similarity_score'], reverse=True)

    top_jobs = scored_jobs[:top_n]
    
    print(f"[Embedding] Ranked {len(jobs)} jobs → top {len(top_jobs)} selected ({time.time()-t0:.1f}s)")
    if top_jobs:
        print(f"[Embedding] Best match: {top_jobs[0].get('role', '?')} at {top_jobs[0].get('company', '?')} (sim={top_jobs[0]['similarity_score']:.4f})")
        print(f"[Embedding] Worst kept: {top_jobs[-1].get('role', '?')} at {top_jobs[-1].get('company', '?')} (sim={top_jobs[-1]['similarity_score']:.4f})")

    return top_jobs
