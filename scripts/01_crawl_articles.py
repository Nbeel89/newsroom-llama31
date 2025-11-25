#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Al Jazeera Full Crawl (v4)
- Deep discovery via sitemaps (fully recursive), section listings, front page hubs,
  and shallow in-page link expansion (depth<=2) to catch stragglers.
- Resume-safe, strong dedup, robust extraction tuned for AlJazeera layouts.
- Target-based stop: --target-new (default 1000).
"""

import argparse, os, re, json, time, random, sys, math, hashlib, threading
import concurrent.futures as cf
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE = "https://www.aljazeera.com"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
]

# Core RSS (good for freshness, not for depth, but we include anyway)
RSS_FEEDS = [
    f"{BASE}/xml/rss/all.xml",
    f"{BASE}/xml/rss/world.xml",
    f"{BASE}/xml/rss/middleeast.xml",
    f"{BASE}/xml/rss/africa.xml",
    f"{BASE}/xml/rss/asia.xml",
    f"{BASE}/xml/rss/europe.xml",
    f"{BASE}/xml/rss/americas.xml",
    f"{BASE}/xml/rss/business.xml",
    f"{BASE}/xml/rss/opinion.xml",
    f"{BASE}/xml/rss/sports.xml",
    f"{BASE}/xml/rss/features.xml",
]

# Section hubs that have classic ?page= pagination
SECTION_SLUGS = [
    "news", "features", "opinion", "world", "middle-east", "africa", "asia",
    "europe", "americas", "business", "economy", "sports", "explainer", "human-rights"
]

VALID_PATH_PREFIXES = (
    "/news/", "/features/", "/opinion/", "/economy/", "/business/",
    "/sports/", "/human-rights/", "/science/", "/technology/", "/culture/",
    "/politics/", "/explainer/", "/gallery/"
)

LIVE_EXCLUDE_SUBSTR = (
    "/liveblog/", "/live/", "/live-b", "/tag/liveblog", "/tag/live"
)

# ----------------------------------------------------------------------

def log(msg):
    ts = datetime.utcnow().strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def session():
    s = requests.Session()
    s.headers.update({"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "en"})
    return s

def fetch_text(sess, url, timeout=20, retries=3, backoff=1.6):
    for attempt in range(1, retries+1):
        try:
            r = sess.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and r.text:
                return r.text
            if r.status_code in (403, 429, 500, 502, 503, 520, 521):
                time.sleep(backoff * attempt + random.uniform(0.2, 0.8))
                sess.headers.update({"User-Agent": random.choice(USER_AGENTS)})
            else:
                return None
        except requests.RequestException:
            time.sleep(backoff * attempt + random.uniform(0.2, 0.8))
            sess.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    return None

# ------------- URL filters ----------------------------------------------------

def normalize_url(u: str) -> str:
    u = u.split("#")[0].split("?")[0].rstrip("/")
    return u

def aj_domain(url: str) -> bool:
    try:
        return urlparse(url).netloc.endswith("aljazeera.com")
    except Exception:
        return False

def is_article_path(path: str) -> bool:
    if not path:
        return False
    p = path.split("?")[0].rstrip("/")
    if any(x in p for x in LIVE_EXCLUDE_SUBSTR):
        return False
    # require one of the valid prefixes and at least one trailing slug
    if p.startswith(VALID_PATH_PREFIXES):
        # filter out obvious hub pages like /news/, /features/ with no slug/date
        return len(p.strip("/").split("/")) >= 2
    return False

def looks_like_article(url: str) -> bool:
    if not aj_domain(url):
        return False
    return is_article_path(urlparse(url).path)

# ------------- Discovery ------------------------------------------------------

def parse_links_from_html(html: str, base: str = BASE):
    links = set()
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http"):
            u = href
        else:
            u = urljoin(base, href)
        u = normalize_url(u)
        if aj_domain(u) and looks_like_article(u):
            links.add(u)
    return links

def discover_via_rss(sess):
    urls = set()
    for feed in RSS_FEEDS:
        txt = fetch_text(sess, feed, timeout=15, retries=2)
        if not txt:
            continue
        for m in re.finditer(r"<link>(https?://[^<]+)</link>", txt):
            u = normalize_url(m.group(1).strip())
            if looks_like_article(u):
                urls.add(u)
        time.sleep(random.uniform(0.25, 0.6))
    log(f"[DISCOVER] RSS -> {len(urls)}")
    return urls

SITEMAP_ROOTS = [
    f"{BASE}/sitemap.xml",
    f"{BASE}/sitemap_index.xml",
    f"{BASE}/sitemap-news.xml",
    f"{BASE}/news-sitemap.xml",
    f"{BASE}/sitemap_news.xml",
]

SITEMAP_LOC_RE = re.compile(r"<loc>(https?://[^<]+)</loc>", re.I)
SITEMAP_LASTMOD_RE = re.compile(r"<lastmod>([^<]+)</lastmod>", re.I)
URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.S | re.I)
SITEMAP_BLOCK_RE = re.compile(r"<sitemap>(.*?)</sitemap>", re.S | re.I)

def iso_to_dt(s: str):
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None

def discover_via_sitemaps(sess, days: int, deep: bool):
    """Fully recursive sitemap traversal. Filters by lastmod if present and within --days."""
    cutoff = datetime.utcnow() - timedelta(days=days) if days > 0 else None
    to_visit = list(SITEMAP_ROOTS)
    visited_xml = set()
    urls = set()
    children_seen = 0

    while to_visit:
        sm = to_visit.pop()
        sm = normalize_url(sm)
        if sm in visited_xml:
            continue
        visited_xml.add(sm)

        txt = fetch_text(sess, sm, timeout=25, retries=3)
        if not txt:
            continue

        # collect child sitemap locations (if this is an index)
        sitemap_blocks = SITEMAP_BLOCK_RE.findall(txt)
        if sitemap_blocks:
            for block in sitemap_blocks:
                locm = SITEMAP_LOC_RE.search(block)
                if not locm:
                    continue
                child = normalize_url(locm.group(1).strip())
                if aj_domain(child) and child.endswith(".xml"):
                    to_visit.append(child)
                    children_seen += 1
            if children_seen and children_seen % 100 == 0:
                log(f"[DISCOVER] sitemap children visited ~{children_seen}")
        else:
            # otherwise, parse URLSET (<url> blocks)
            for ub in URL_BLOCK_RE.findall(txt):
                locm = SITEMAP_LOC_RE.search(ub)
                if not locm:
                    continue
                u = normalize_url(locm.group(1).strip())
                if not aj_domain(u):
                    continue
                if not looks_like_article(u):
                    continue
                if cutoff:
                    lm = SITEMAP_LASTMOD_RE.search(ub)
                    if lm:
                        dt = iso_to_dt(lm.group(1).strip())
                        if dt and dt < cutoff:
                            continue
                urls.add(u)

        # be polite
        time.sleep(random.uniform(0.15, 0.45) if deep else random.uniform(0.05, 0.2))

    log(f"[DISCOVER] SITEMAPS -> {len(urls)} (xml visited={len(visited_xml)}, children~{children_seen})")
    return urls

def discover_via_sections(sess, max_pages: int, deep: bool):
    """Paginate classic sections and collect article links."""
    urls = set()
    for sec in SECTION_SLUGS:
        no_gain_streak = 0
        total_before = len(urls)
        for page in range(1, max_pages + 1):
            url = f"{BASE}/{sec}/?page={page}"
            html = fetch_text(sess, url, timeout=20, retries=2)
            if not html:
                no_gain_streak += 1
                if no_gain_streak >= 8:
                    break
                continue
            found = parse_links_from_html(html)
            before = len(urls)
            urls |= found
            gained = len(urls) - before
            if gained == 0:
                no_gain_streak += 1
                if no_gain_streak >= 8:
                    break
            else:
                no_gain_streak = 0
            if page % 8 == 0:
                log(f"[DISCOVER] section {sec:>12} page {page:>3} -> +{gained} (total {len(urls)})")
            time.sleep(random.uniform(0.10, 0.35) if deep else random.uniform(0.05, 0.15))
        gained_total = len(urls) - total_before
        if gained_total == 0:
            pass
    log(f"[DISCOVER] SECTIONS -> {len(urls)}")
    return urls

def discover_frontpage_and_hubs(sess):
    """Grab the homepage and key hubs listed under the top navbar ('More', etc.)."""
    seeds = {
        BASE,
        f"{BASE}/news/",
        f"{BASE}/features/",
        f"{BASE}/opinion/",
        f"{BASE}/middle-east/",
        f"{BASE}/world/",
        f"{BASE}/business/",
        f"{BASE}/economy/",
        f"{BASE}/sports/",
        f"{BASE}/explainer/",
        f"{BASE}/human-rights/",
    }
    urls = set()
    for u in seeds:
        html = fetch_text(sess, u, timeout=20, retries=2)
        if not html: 
            continue
        urls |= parse_links_from_html(html, base=u)
        time.sleep(random.uniform(0.1, 0.3))
    log(f"[DISCOVER] FRONT/HUBS -> {len(urls)}")
    return urls

def expand_inpage_links(sess, urls_in: set, depth=2, cap_per_page=50):
    """Expand links found inside already-discovered article pages (depth-limited)."""
    if depth <= 0 or not urls_in:
        return set()
    out = set()
    for i, u in enumerate(list(urls_in)[:1500]):  # safety cap
        html = fetch_text(sess, u, timeout=20, retries=2)
        if not html:
            continue
        found = list(parse_links_from_html(html))
        if cap_per_page:
            found = found[:cap_per_page]
        out |= set(found)
        if (i + 1) % 50 == 0:
            log(f"[DISCOVER] expand depth pass -> processed {i+1} pages")
        time.sleep(random.uniform(0.06, 0.2))
    if depth > 1:
        out |= expand_inpage_links(sess, out, depth=depth-1, cap_per_page=cap_per_page)
    return out

# ------------- Existing / Dedup ----------------------------------------------

def load_existing_urls(out_path: str):
    seen = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    seen.add(json.loads(line)["url"])
                except Exception:
                    pass
    return seen

# ------------- Extraction -----------------------------------------------------

CANDIDATE_BODY_SELECTORS = [
    # Newer article bodies
    {"name": "div", "attrs": {"data-component": "article-body"}},
    {"name": "div", "attrs": {"data-component": "wysiwyg"}},
    {"name": "div", "attrs": {"class": lambda v: v and ("wysiwyg" in v or "article-p-wrapper" in v)}},
    {"name": "article"},
    # Fallback: page content
    {"name": "div", "attrs": {"class": lambda v: v and ("main-article" in v or "post-content" in v)}},
]

def extract_text_from_container(el):
    # AlJazeera often uses <p>, <li>, headers; we capture p & li
    chunks = []
    for p in el.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if t:
            chunks.append(t)
    # collapse duplicate whitespace, keep newlines
    text = "\n".join(chunks)
    return text

def extract_article(sess, url, min_words):
    url = normalize_url(url)
    html = fetch_text(sess, url, timeout=25, retries=3)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # canonical normalization (sometimes queryless canonical differs)
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        canon = normalize_url(canonical["href"])
        if aj_domain(canon) and looks_like_article(canon):
            url = canon

    # title
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()

    # time / published
    published = None
    t = soup.find("time")
    if t and t.get("datetime"):
        published = t.get("datetime")
    if not published:
        m = soup.find("meta", attrs={"property": "article:published_time"})
        if m and m.get("content"):
            published = m.get("content")

    # body
    content = ""
    for sel in CANDIDATE_BODY_SELECTORS:
        el = soup.find(sel["name"], attrs=sel.get("attrs"))
        if not el:
            continue
        content = extract_text_from_container(el)
        if content and len(content.split()) >= max(60, min_words // 2):
            break

    # Fallback: collect all <p> if specific containers not found
    if not content:
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        content = "\n".join([p for p in ps if p])

    words = len(content.split())
    if not title or words < min_words:
        return None

    return {
        "url": url,
        "title": title,
        "published": published,
        "content": content,
        "word_count": words,
        "scraped_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

# ------------- Main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Al Jazeera deep full-site crawler (v4)")
    ap.add_argument("--days", type=int, default=3650, help="Only include sitemap URLs roughly within this window (0 = no filter)")
    ap.add_argument("--max-pages", type=int, default=120, help="Listing pages per section")
    ap.add_argument("--workers", type=int, default=12, help="Parallel fetch workers")
    ap.add_argument("--min-words", type=int, default=80, help="Minimum words to accept article")
    ap.add_argument("--target-new", type=int, default=1000, help="Stop after saving this many NEW articles (0 = unlimited)")
    ap.add_argument("--out", type=str, default="data/raw/aljazeera_articles_full.jsonl", help="Output JSONL (append/resume-safe)")
    ap.add_argument("--deep", action="store_true", help="Deeper discovery (slower, broader), plus in-page expansion depth=2")
    ap.add_argument("--expand-depth", type=int, default=2, help="In-page expansion depth when --deep")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    log(f"Start crawl: days={args.days} max_pages={args.max_pages} workers={args.workers} "
        f"min_words={args.min_words} target_new={args.target_new}")

    seen = load_existing_urls(args.out)
    log(f"Existing articles in {args.out}: {len(seen)}")

    sess = session()

    # Heartbeat thread
    stop_hb = False
    def heartbeat():
        while not stop_hb:
            log("HEARTBEAT crawler alive…")
            time.sleep(60)
    threading.Thread(target=heartbeat, daemon=True).start()

    # ----------- Discovery mix
    discovered = set()

    # 1) RSS (quick wins)
    try:
        discovered |= discover_via_rss(sess)
    except Exception as e:
        log(f"[WARN] RSS discovery error: {e}")

    # 2) Sitemaps (full recursion)
    try:
        discovered |= discover_via_sitemaps(sess, args.days, args.deep)
    except Exception as e:
        log(f"[WARN] Sitemaps discovery error: {e}")

    # 3) Sections pagination
    try:
        discovered |= discover_via_sections(sess, args.max_pages, args.deep)
    except Exception as e:
        log(f"[WARN] Sections discovery error: {e}")

    # 4) Front page + nav hubs
    try:
        discovered |= discover_frontpage_and_hubs(sess)
    except Exception as e:
        log(f"[WARN] Front/hubs discovery error: {e}")

    # 5) Optional: in-page expansion to catch uncategorized/“explained” content
    if args.deep:
        try:
            expanded = expand_inpage_links(sess, discovered, depth=max(1, args.expand_depth), cap_per_page=60)
            discovered |= expanded
            log(f"[DISCOVER] EXPANSION -> total {len(discovered)} after depth expand")
        except Exception as e:
            log(f"[WARN] Expansion error: {e}")

    # Deduplicate & filter already-seen
    discovered = {normalize_url(u) for u in discovered if looks_like_article(u)}
    todo = [u for u in discovered if u not in seen]
    random.shuffle(todo)

    log(f"Discovered unique candidate URLs: {len(discovered)}")
    log(f"New URLs to process (not in output): {len(todo)}")

    if not todo:
        stop_hb = True
        log("No new URLs found; exiting.")
        return

    saved = 0
    errors = 0
    processed = 0
    last_flush = 0

    with open(args.out, "a", encoding="utf-8") as out_f:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2url = {ex.submit(extract_article, sess, u, args.min_words): u for u in todo}
            for fut in cf.as_completed(fut2url):
                processed += 1
                u = fut2url[fut]
                try:
                    art = fut.result()
                except Exception as e:
                    log(f"[ERROR] fetch {u}: {e}")
                    errors += 1
                    continue

                if art:
                    url_norm = art["url"]
                    if url_norm in seen:
                        # ultra-safe guard
                        continue
                    json.dump(art, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    saved += 1
                    seen.add(url_norm)
                    if saved % 10 == 0:
                        log(f"[SAVE] {saved} of {len(todo)}  —  {art['title'][:80]} ({art['word_count']}w)")
                    # periodic flush
                    if saved - last_flush >= 25:
                        out_f.flush()
                        last_flush = saved
                    if args.target_new and saved >= args.target_new:
                        log(f"Target reached ({args.target_new}). Stopping.")
                        stop_hb = True
                        log(f"Processed={processed} Saved={saved} Errors={errors}")
                        return
                else:
                    # either short, non-article, or extraction failed
                    pass

    stop_hb = True
    log(f"Completed. saved={saved} errors={errors} discovered={len(discovered)} new_todo={len(todo)}")
    log("Done.")

if __name__ == "__main__":
    main()
