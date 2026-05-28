
import re
import socket
import ipaddress
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


APP_NAME = "UKRI Funding Scout Agent"

HEADERS = {
    "User-Agent": "UKRI-Funding-Scout-Agent/1.0 research prototype; contact: example@example.com"
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "almost", "along", "also", "although",
    "always", "among", "another", "around", "because", "before", "being", "between",
    "both", "could", "does", "doing", "during", "each", "either", "from", "further",
    "have", "having", "here", "into", "itself", "just", "more", "most", "other",
    "over", "same", "should", "some", "such", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "under", "until", "very", "were",
    "what", "when", "where", "which", "while", "with", "within", "would", "research",
    "university", "professor", "school", "department", "publications", "profile",
    "project", "projects", "paper", "papers", "using", "based", "work", "working"
}

DOMAIN_PHRASES = [
    "artificial intelligence", "machine learning", "deep learning", "data science",
    "financial technology", "fintech", "digital finance", "sustainable finance",
    "green finance", "climate finance", "esg", "blockchain", "crypto", "digital assets",
    "tokenisation", "risk management", "forecasting", "banking", "financial regulation",
    "consumer protection", "accountability", "responsible ai", "ai governance",
    "digital literacy", "skills", "workforce", "healthcare", "nhs", "one health",
    "public health", "water energy nexus", "sustainability", "climate risk",
    "creative economy", "cultural assets", "knowledge exchange", "commercialisation",
    "local authority", "social science", "interdisciplinary", "policy", "innovation"
]


class MatchRequest(BaseModel):
    profile_url: str | None = Field(default=None, description="Public profile URL, for example a university profile, ORCID, personal page, or Google Scholar page.")
    profile_text: str | None = Field(default=None, description="Optional pasted profile text. Useful for LinkedIn, which may block automated access.")
    extra_keywords: str | None = Field(default=None, description="Optional extra research keywords, separated by commas.")
    max_results: int = Field(default=10, ge=1, le=25)


app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For prototype only. Restrict this to your GitHub Pages URL in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def home():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Frontend file not found at {index_file}"
        )
    return FileResponse(index_file)

@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.post("/api/match")
def match_funding(req: MatchRequest):
    profile_text = ""

    if req.profile_text and req.profile_text.strip():
        profile_text += "\n" + req.profile_text.strip()

    if req.profile_url and req.profile_url.strip():
        fetched = fetch_profile_text(req.profile_url.strip())
        profile_text += "\n" + fetched

    if not profile_text.strip():
        raise HTTPException(status_code=400, detail="Please provide either a profile URL or pasted profile text.")

    if req.extra_keywords:
        profile_text += "\n" + req.extra_keywords

    keywords = extract_keywords(profile_text, extra_keywords=req.extra_keywords)
    queries = build_search_queries(keywords)

    opportunities = []
    seen = set()

    for query in queries:
        results = search_ukri_opportunities(query, max_pages=2)
        for item in results:
            if item["url"] not in seen:
                seen.add(item["url"])
                opportunities.append(item)

    if not opportunities:
        return {
            "profile_keywords": keywords,
            "queries_used": queries,
            "matches": [],
            "message": "No UKRI opportunities were found. Try adding broader keywords such as AI, finance, sustainability, health, policy or innovation."
        }

    ranked = rank_opportunities(profile_text, keywords, opportunities)
    ranked = ranked[: req.max_results]

    return {
        "profile_keywords": keywords,
        "queries_used": queries,
        "matches": ranked,
        "message": f"Found {len(ranked)} relevant opportunities from UKRI Funding Finder."
    }


def is_safe_public_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if not parsed.hostname:
        return False

    hostname = parsed.hostname.lower()

    if hostname in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return False

    try:
        addresses = socket.getaddrinfo(hostname, None)
        for addr in addresses:
            ip = ipaddress.ip_address(addr[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
    except Exception:
        return False

    return True


def fetch_profile_text(url: str) -> str:
    if not is_safe_public_url(url):
        raise HTTPException(status_code=400, detail="The profile URL must be a public http or https URL.")

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch the profile URL. Some sites, especially LinkedIn, block automated access. Paste profile text instead. Error: {exc}"
        )

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.extract()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = meta["content"]

    body = soup.get_text(" ", strip=True)
    body = re.sub(r"\s+", " ", body)

    return (title + "\n" + meta_desc + "\n" + body)[:50000]


def extract_keywords(text: str, extra_keywords: str | None = None, limit: int = 16) -> list[str]:
    text_lower = text.lower()
    scores = {}

    for phrase in DOMAIN_PHRASES:
        count = text_lower.count(phrase)
        if count:
            scores[phrase] = scores.get(phrase, 0) + count * 6

    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text_lower)
    clean_tokens = [
        t.replace("-", " ")
        for t in tokens
        if t not in STOPWORDS and len(t) > 3 and not t.isdigit()
    ]

    for token in clean_tokens:
        scores[token] = scores.get(token, 0) + 1

    for i in range(len(clean_tokens) - 1):
        bigram = f"{clean_tokens[i]} {clean_tokens[i+1]}"
        if all(part not in STOPWORDS for part in bigram.split()):
            scores[bigram] = scores.get(bigram, 0) + 2

    if extra_keywords:
        for kw in re.split(r"[,;\n]", extra_keywords):
            kw = kw.strip().lower()
            if kw:
                scores[kw] = scores.get(kw, 0) + 10

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    keywords = []

    for kw, _ in ranked:
        if len(keywords) >= limit:
            break
        if len(kw) < 3:
            continue
        if not any(kw in existing or existing in kw for existing in keywords):
            keywords.append(kw)

    return keywords


def build_search_queries(keywords: list[str]) -> list[str]:
    if not keywords:
        return ["interdisciplinary research"]

    queries = []

    # A broader combined query
    queries.append(" ".join(keywords[:4]))

    # Focused searches
    for kw in keywords[:6]:
        queries.append(kw)

    # Extra thematic query for broad UKRI opportunities
    if any(k in " ".join(keywords) for k in ["ai", "artificial intelligence", "machine learning", "data science"]):
        queries.append("artificial intelligence data innovation")
    if any(k in " ".join(keywords) for k in ["finance", "fintech", "banking", "financial"]):
        queries.append("finance innovation regulation")
    if any(k in " ".join(keywords) for k in ["health", "nhs", "wellbeing"]):
        queries.append("health data innovation")
    if any(k in " ".join(keywords) for k in ["sustainability", "climate", "energy", "water"]):
        queries.append("sustainability climate transition")

    # De-duplicate while preserving order
    unique = []
    for q in queries:
        q = q.strip()
        if q and q not in unique:
            unique.append(q)

    return unique[:8]


def search_ukri_opportunities(query: str, max_pages: int = 2) -> list[dict]:
    base = "https://www.ukri.org/opportunity/"
    found = []

    session = requests.Session()

    for page in range(1, max_pages + 1):
        url = base if page == 1 else f"{base}page/{page}/"
        params = [
            ("keywords", query),
            ("filter_status[]", "open"),
            ("filter_status[]", "upcoming"),
            ("filter_submitted", "true"),
            ("filter_order", "closing_date"),
        ]

        try:
            response = session.get(url, params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        page_items = parse_ukri_results_page(soup)

        if not page_items:
            break

        found.extend(page_items)

    return found


def parse_ukri_results_page(soup: BeautifulSoup) -> list[dict]:
    items = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = urljoin("https://www.ukri.org", a["href"])
        title = a.get_text(" ", strip=True)

        if not title or len(title) < 8:
            continue

        if "/opportunity/" not in href:
            continue

        normalised = href.split("#")[0].split("?")[0].rstrip("/")

        if normalised == "https://www.ukri.org/opportunity":
            continue

        if normalised in seen:
            continue

        seen.add(normalised)

        container = a
        for _ in range(5):
            if container.parent:
                container = container.parent
            text = container.get_text(" ", strip=True)
            if len(text) > 220:
                break

        text = re.sub(r"\s+", " ", container.get_text(" ", strip=True))

        item = {
            "title": title,
            "url": normalised + "/",
            "summary": text[:900],
            "status": extract_field(text, r"Opportunity status:\s*(Open|Upcoming|Closed)"),
            "funders": extract_field(text, r"Funders?:\s*(.*?)(?:Funding type:|Co-funders:|Total fund:|Maximum award:|Publication date:|Opening date:|Closing date:|$)"),
            "funding_type": extract_field(text, r"Funding type:\s*(.*?)(?:Total fund:|Maximum award:|Award:|Publication date:|Opening date:|Closing date:|$)"),
            "award": extract_field(text, r"(?:Total fund|Maximum award|Award):\s*(.*?)(?:Publication date:|Opening date:|Closing date:|$)"),
            "closing_date": extract_field(text, r"Closing date:\s*(.*?)(?:$|Publication date:|Opening date:)"),
        }

        items.append(item)

    return items


def extract_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return ""
    value = m.group(1)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:220]


def rank_opportunities(profile_text: str, keywords: list[str], opportunities: list[dict]) -> list[dict]:
    docs = [
        f"{o.get('title','')} {o.get('summary','')} {o.get('funders','')} {o.get('funding_type','')}"
        for o in opportunities
    ]

    if TfidfVectorizer is not None and cosine_similarity is not None:
        corpus = [profile_text] + docs
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    else:
        sims = [keyword_overlap_score(profile_text, d, keywords) for d in docs]

    ranked = []

    for item, sim, doc in zip(opportunities, sims, docs):
        overlap_terms = overlapping_terms(keywords, doc)
        overlap_bonus = min(len(overlap_terms) / 8, 1)
        score = round(float((0.75 * sim + 0.25 * overlap_bonus) * 100), 1)

        item = dict(item)
        item["match_score"] = score
        item["why_match"] = make_why_match(overlap_terms, item)
        item["bid_angle"] = make_bid_angle(keywords, item, overlap_terms)
        item["matched_terms"] = overlap_terms[:8]
        ranked.append(item)

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked


def keyword_overlap_score(profile_text: str, opportunity_text: str, keywords: list[str]) -> float:
    opp = opportunity_text.lower()
    hits = sum(1 for k in keywords if k.lower() in opp)
    return min(hits / max(len(keywords), 1), 1)


def overlapping_terms(keywords: list[str], opportunity_text: str) -> list[str]:
    opp = opportunity_text.lower()
    return [k for k in keywords if k.lower() in opp]


def make_why_match(overlap_terms: list[str], item: dict) -> str:
    if overlap_terms:
        return f"The opportunity text overlaps with your profile themes: {', '.join(overlap_terms[:6])}."
    funders = item.get("funders") or "the listed funder"
    return f"The opportunity appears potentially relevant based on semantic similarity and the priorities of {funders}."


def make_bid_angle(keywords: list[str], item: dict, overlap_terms: list[str]) -> str:
    terms = overlap_terms[:4] if overlap_terms else keywords[:4]
    terms_text = ", ".join(terms) if terms else "your research profile"

    return (
        f"Possible application angle: position the bid around {terms_text}, with a clear route from research excellence "
        f"to measurable impact, stakeholder engagement, and deliverable outputs aligned with the call scope. "
        f"Before applying, check eligibility, closing date, funding limits, and whether the call requires specific partners."
    )
