import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import json
import re
import os
import sys
import logging
from collections import deque
import urllib.robotparser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SEEDS = {
    "drugs.com": "https://www.drugs.com",
    "medscape": "https://reference.medscape.com/drugs",
    "dailymed": "https://dailymed.nlm.nih.gov/dailymed"
}
OUTPUT_FILE = "drug_dataset.jsonl"
USER_AGENT = "PharmaIntelCrawler/1.0 (+https://example.com/) - student project - contact: you@example.com"
MAX_PAGES_PER_SITE = 100 
REQUEST_TIMEOUT = 15
MIN_DELAY = 1.0
MAX_DELAY = 3.0

DRUG_URL_PATTERNS = [
    re.compile(r"/drug", re.I),
    re.compile(r"/drugs/", re.I),
    re.compile(r"drug-information", re.I),
    re.compile(r"drugmonograph", re.I),
    re.compile(r"drug\_info", re.I),
    re.compile(r"drugInfo", re.I),
    re.compile(r"setid=", re.I),              # dailymed setid param
    re.compile(r"/drug/"),                    # medscape/drugs
]

SECTION_KEYWORDS = {
    "indications": ["indications", "indication", "uses", "clinical uses"],
    "mechanism": ["mechanism", "mechanism of action", "pharmacology"],
    "dosage": ["dosage", "dosage and administration", "administration", "dosage & administration"],
    "side_effects": ["side effects", "adverse reactions", "adverse effects", "adverse events"],
    "interactions": ["interactions", "drug interactions", "drug–drug interactions", "drug–food interactions"],
    "brand_names": ["brand names", "brands", "trade names"]
}

HEADINGS = ["h1", "h2", "h3", "h4", "h5"]
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def is_allowed_by_robots(seed_url, path):
    """Check robots.txt for the given domain and path (basic)."""
    try:
        parsed = urlparse(seed_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, path)
    except Exception:
        return True

def polite_get(url):
    """GET with timeout and polite delay + error handling."""
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logging.warning(f"Request failed for {url}: {e}")
        return None

def is_internal_link(base_netloc, link):
    """Check if link is internal to base netloc."""
    try:
        p = urlparse(link)
        if not p.netloc:
            return True
        return p.netloc.endswith(base_netloc)
    except Exception:
        return False

def clean_text(txt):
    if not txt:
        return ""
    return re.sub(r"\s+", " ", txt).strip()

def extract_field_by_section(soup, keywords):
    """
    Find headings that match any keyword and return following text until next heading.
    Returns concatenated text (first match).
    """
    for h in soup.find_all(HEADINGS):
        h_text = clean_text(h.get_text())
        for kw in keywords:
            if kw.lower() in h_text.lower():
                pieces = []
                for sib in h.next_siblings:
                    if getattr(sib, "name", None) and sib.name in HEADINGS:
                        break
                    if getattr(sib, "get_text", None):
                        t = clean_text(sib.get_text())
                        if t:
                            pieces.append(t)
                if pieces:
                    return " ".join(pieces)
    for p in soup.find_all("p"):
        if any(kw.lower() in p.get_text().lower() for kw in keywords):
            return clean_text(p.get_text())
    return ""

def extract_brand_names(soup):
    text = extract_field_by_section(soup, SECTION_KEYWORDS["brand_names"])
    if text:
        return text
    body_text = soup.get_text(separator="\n")
    m = re.search(r"Brand(?:\s|-)names?:\s*(.+)", body_text, re.I)
    if m:
        return clean_text(m.group(1).split("\n")[0])
    return ""

def extract_generic_name(soup):
    h1 = soup.find("h1")
    if h1:
        name = clean_text(h1.get_text())
        name = re.split(r"[-–|:]", name)[0].strip()
        return name
    og = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
    if og and og.get("content"):
        return clean_text(og.get("content"))
    return ""

def extract_structured_fields(url, html, site_key):
    """Heuristic extraction from BeautifulSoup object and return dict."""
    soup = BeautifulSoup(html, "html.parser")
    data = {
        "source_url": url,
        "site": site_key,
        "generic_name": None,
        "brand_names": None,
        "indications": None,
        "mechanism": None,
        "dosage": None,
        "side_effects": None,
        "interactions": None,
        "raw_excerpt": None
    }

    try:
        data["generic_name"] = extract_generic_name(soup)
        data["brand_names"] = extract_brand_names(soup)
        data["indications"] = extract_field_by_section(soup, SECTION_KEYWORDS["indications"])
        data["mechanism"] = extract_field_by_section(soup, SECTION_KEYWORDS["mechanism"])
        data["dosage"] = extract_field_by_section(soup, SECTION_KEYWORDS["dosage"])
        data["side_effects"] = extract_field_by_section(soup, SECTION_KEYWORDS["side_effects"])
        data["interactions"] = extract_field_by_section(soup, SECTION_KEYWORDS["interactions"])

        visible = soup.get_text(separator=" ")
        data["raw_excerpt"] = clean_text(visible)[:1000]
    except Exception as e:
        logging.warning(f"Extraction error for {url}: {e}")

    for k, v in list(data.items()):
        if isinstance(v, str):
            v2 = v.strip()
            if v2 == "":
                data[k] = None
            else:
                data[k] = v2
    return data

def gather_links_from_page(base_url, html, base_netloc):
    """Return set of absolute internal links found on page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a['href'].split('#')[0].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        
        if absolute.startswith("mailto:") or absolute.startswith("javascript:"):
            continue
        if is_internal_link(base_netloc, absolute):
            links.add(absolute)
    return links

def looks_like_drug_page(url):
    """Heuristic: return True if URL likely points to a drug page."""
    for pat in DRUG_URL_PATTERNS:
        if pat.search(url):
            return True
    p = urlparse(url)
    if p.path and len(p.path.split("/")) >= 2 and p.path.endswith((".html", ".htm", "/")):
        return True
    return False

def crawl_site(seed_url, site_key, max_pages=100):
    logging.info(f"Starting crawl for {site_key} -> {seed_url}")
    parsed = urlparse(seed_url)
    base_netloc = parsed.netloc
    visited = set()
    to_visit = deque()
    to_visit.append((seed_url, 0))
    pages_visited = 0
    found_drug_pages = set()
    results = []

    while to_visit and pages_visited < max_pages:
        url, depth = to_visit.popleft()
        if url in visited:
            continue
        # check robots
        path_for_robots = urlparse(url).path or "/"
        if not is_allowed_by_robots(seed_url, path_for_robots):
            logging.info(f"Blocked by robots.txt: {url}")
            visited.add(url)
            continue
        resp = polite_get(url)
        visited.add(url)
        if resp is None:
            continue
        pages_visited += 1
        html = resp.text
        logging.info(f"[{site_key}] Visited ({pages_visited}/{max_pages}) depth={depth}: {url}")

        links = gather_links_from_page(url, html, base_netloc)

        if depth < 1:  
            for link in links:
                if link not in visited:
                    to_visit.append((link, depth + 1))

        if looks_like_drug_page(url):
            logging.info(f"Candidate drug page found: {url}")
            data = extract_structured_fields(url, html, site_key)
            if data.get("generic_name") or data.get("indications") or (data.get("raw_excerpt") and len(data["raw_excerpt"]) > 200):
         
                if url not in found_drug_pages:
                    results.append(data)
                    found_drug_pages.add(url)
                    logging.info(f"Extracted {data.get('generic_name') or url} from {site_key}")
            else:
                logging.info(f"Rejected candidate (insufficient data): {url}")

    logging.info(f"Finished crawling {site_key}. Extracted {len(results)} candidate drug pages.")
    return results

def main(max_pages_per_site=MAX_PAGES_PER_SITE, output_file=OUTPUT_FILE):
    all_results = []
    for site_key, seed in SEEDS.items():
        try:
            res = crawl_site(seed, site_key, max_pages=max_pages_per_site)
            all_results.extend(res)
        except Exception as e:
            logging.error(f"Error crawling {site_key}: {e}")

    # Save to JSONL
    if not all_results:
        logging.warning("No results extracted — check seeds and network/robots.")
    else:
        with open(output_file, "w", encoding="utf-8") as fh:
            for r in all_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(all_results)} records to {output_file}")

if __name__ == "__main__":
    # Allow optional CLI override for pages per site
    arg = None
    if len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1])
        except Exception:
            arg = None
    if arg:
        main(max_pages_per_site=arg)
    else:
        main()
