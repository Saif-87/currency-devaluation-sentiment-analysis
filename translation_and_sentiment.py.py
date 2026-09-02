# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import requests
from readability import Document
from bs4 import BeautifulSoup
from transformers import pipeline
import spacy
from langdetect import detect, DetectorFactory
from langcodes import Language
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import threading
import json
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Make langdetect deterministic (same text -> same result every run)
DetectorFactory.seed = 0

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== EMERGENCY FAST MODE SETTINGS ==================
MAX_WORKERS        = 10
REQUEST_DELAY      = 0.5      # min seconds between outbound requests (across all threads)
CONNECT_TIMEOUT    = 10
READ_TIMEOUT       = 15
CHECKPOINT_EVERY   = 50
CHECKPOINT_PATH    = "emergency_checkpoint.json"
MAX_ARTICLE_CHARS  = 20_000   # chars of body text fed to translation + sentiment

_rate_lock = threading.Lock()
_next_slot = 0.0

def rate_limit():
    """Thread-safe global throttle: each caller reserves a time slot, then sleeps outside the lock."""
    global _next_slot
    with _rate_lock:
        now = time.time()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + REQUEST_DELAY
    if wait:
        time.sleep(wait)

def save_checkpoint(processed, results):
    payload = {"processed_count": processed, "results": results, "timestamp": time.time()}
    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"✅ Checkpoint saved: {processed} items processed")
    except Exception as e:
        print(f"⚠️ Checkpoint save failed: {e}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== OPTIMISED HTTP SESSION ==================
def make_session(total_retries=2, backoff=0.3):
    sess = requests.Session()
    # GET is retryable by default; allowed_methods/method_whitelist naming differs by urllib3 version.
    retry = Retry(
        total           = total_retries,
        backoff_factor  = backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    sess.mount("http://",  adapter)
    sess.mount("https://", adapter)
    sess.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    return sess

SESSION = make_session()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== LOAD NLP MODELS ONCE ==================
print("🔄 Loading NLP models for emergency fast mode…")
try:
    # truncation/max_length set at construction so they're always applied,
    # regardless of the installed transformers version.
    sentiment = pipeline(
        "sentiment-analysis",
        model="tabularisai/ModernFinBERT",
        truncation=True,
        max_length=8192,
    )
    nlp       = spacy.load("en_core_web_sm")
    print("✅ All NLP models loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise

# ModernFinBERT sentiment ───────────
def analyse_sentiment(text: str):
    """Analyse sentiment with ModernFinBERT (tokenizer trims beyond 8 192 tokens).

    Returns (label, score) or a failure pair.
    """
    if not text or not text.strip():
        return "[SENTIMENT FAILED]", None
    try:
        res = sentiment(text)[0]
        return res["label"], round(float(res["score"]), 2)
    except Exception:
        return "[SENTIMENT FAILED]", None

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== UTILITY FUNCTIONS ==================
def detect_lang_code(text: str) -> str:
    """ISO code from langdetect, or 'unknown'."""
    if not text or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"

def lang_display_name(code: str) -> str:
    try:
        return Language.get(code).display_name()
    except Exception:
        return "Unknown"

def translate_text_safe(text: str) -> str:
    """Translate a short string (e.g. a title) to English; return the original on failure."""
    if not text:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        return text

def translate_to_english(text: str, source_lang="auto", retries=2) -> str:
    if not text:
        return ""
    CHUNK_SIZE = 4_500
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_chunks = []

    for chunk in chunks:
        for attempt in range(retries):
            try:
                translated = GoogleTranslator(source=source_lang, target="en").translate(chunk)
                translated_chunks.append(translated or chunk)   # some versions return None on failure
                break
            except Exception as e:
                print(f"   ⚠️ Translation failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(0.5)
        else:
            translated_chunks.append(chunk)

    return " ".join(translated_chunks)

def fetch(url):
    """Single GET; the session's HTTPAdapter already handles connection/5xx retries."""
    try:
        return SESSION.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        print(f"   ⚠️ Request failed: {e}")
        return None

def extract_main_text(html: str) -> str:
    try:
        doc  = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        return soup.get_text().strip()
    except Exception:
        return ""

def extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception:
        return ""

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== ARTICLE PROCESSING ==================
# Phase 1 (thread-safe, I/O bound): fetch + translate.
# Phase 2 (sequential): sentiment + NER, because spaCy / HF pipelines are not
# safe to call concurrently from multiple threads.

_KEYWORD_LABELS = ("GPE", "ORG", "MONEY")

def fetch_and_translate(row: dict, max_chars: int = MAX_ARTICLE_CHARS):
    """Return a partial record (no sentiment/keywords yet) or None if the row is skipped."""
    title = row.get("title", row.get("Title", "")) or ""
    url   = row.get("url",   row.get("URL",  "")) or ""
    title = title if isinstance(title, str) else str(title)
    url   = url   if isinstance(url,   str) else str(url)

    if not url.startswith("http") or "video" in url.lower() or len(url) < 10:
        return None

    try:
        rate_limit()
        domain           = extract_domain(url)
        translated_title = translate_text_safe(title)

        resp = fetch(url)
        if not resp or not getattr(resp, "ok", False):
            return {
                "URL"             : url,
                "Translated Title": translated_title or title,
                "Translated Text" : translated_title or title,
                "Language"        : lang_display_name(detect_lang_code(title)),
                "Domain"          : domain,
                "_status"         : "no_content",
            }

        text_full = extract_main_text(resp.text)
        basis     = text_full if text_full and len(text_full) > 50 else title
        lang_code = detect_lang_code(basis)
        is_en     = lang_code == "en"

        if not text_full or len(text_full.split()) < 50:
            src = text_full or ""
            translated_text = (src if (src and is_en) else translate_to_english(src)) \
                              or translated_title or title
        else:
            capped = text_full[:max_chars]
            translated_text = capped if is_en else translate_to_english(capped)

        return {
            "URL"             : url,
            "Translated Title": translated_title or title,
            "Translated Text" : translated_text or (translated_title or title),
            "Language"        : lang_display_name(lang_code),
            "Domain"          : domain,
            "_status"         : "ok",
        }

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def enrich_sentiment(partial: dict) -> dict:
    """Add sentiment label/score and keyword entities to a phase-1 partial record."""
    text = partial["Translated Text"] or ""

    if partial.get("_status") == "no_content":
        label, score = "[FAILED - NO CONTENT]", None
    else:
        label, score = analyse_sentiment(text)

    try:
        ents = nlp(text).ents
        keywords_found = ", ".join(
            sorted({ent.text for ent in ents if ent.label_ in _KEYWORD_LABELS})
        )
    except Exception:
        keywords_found = ""

    return {
        "URL"                  : partial["URL"],
        "Translated Title"     : partial["Translated Title"],
        "Translated Text"      : text,
        "Language"             : partial["Language"],
        "Model Sentiment Label": label,
        "Model Sentiment Score": score,
        "Keywords"             : keywords_found,
        "Domain"               : partial["Domain"],
    }

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== EMERGENCY MODE MAIN PROCESSOR ==================
def translate_summarize_enrich_df_emergency(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "URL", "Translated Title", "Translated Text", "Language",
        "Model Sentiment Label", "Model Sentiment Score", "Keywords", "Domain"
    ]
    rows_list = df.to_dict(orient="records")
    total     = len(rows_list)
    if total == 0:
        print("⚠️ No rows to process")
        return pd.DataFrame(columns=cols)

    print("🚀 EMERGENCY FAST MODE ACTIVATED")
    print(f"📊 Processing {total} articles | 👥 workers: {MAX_WORKERS} | ⏱️ {REQUEST_DELAY}s spacing")
    print(f"💾 Checkpointing every {CHECKPOINT_EVERY} items → {CHECKPOINT_PATH}")

    start_time = time.time()

    # ── Phase 1: parallel fetch + translation (I/O bound) ──
    partials = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_and_translate, row) for row in rows_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetch+translate"):
            p = fut.result()
            if p:
                partials.append(p)

    # ── Phase 2: sequential sentiment + NER (models are not thread-safe) ──
    results = []
    for i, p in enumerate(tqdm(partials, desc="Sentiment+NER"), start=1):
        results.append(enrich_sentiment(p))
        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(i, results)
    save_checkpoint(len(results), results)

    duration = time.time() - start_time
    print("\n✅ EMERGENCY MODE COMPLETED!")
    print(f"⏱️ Total time: {duration/60:.1f} min")
    print(f"📈 Successfully processed: {len(results)}/{total} ({len(results)/max(total, 1)*100:.1f} %)")
    if duration > 0:
        print(f"🚀 Speed: {len(results)/duration*60:.1f} articles/min")

    return pd.DataFrame(results, columns=cols)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================== MAIN EXECUTION ==================
print("🔄 Starting Emergency Fast Mode Processing…")

currency_devaluation_main = dataiku.Dataset("currency_devaluation_main")
base_df = currency_devaluation_main.get_dataframe()
print(f"📥 Loaded dataset with {len(base_df)} rows")

base_cols = ["URL", "Date", "Title", "Country", "Domain", "Keywords"]
available = [c for c in base_cols if c in base_df.columns]
base_df_min = base_df[available].copy()
print(f"📋 Using columns: {available}")

if "URL" in base_df_min.columns:
    before = len(base_df_min)
    base_df_min = base_df_min.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    if before != len(base_df_min):
        print(f"🧹 Dropped {before - len(base_df_min)} duplicate URLs")

translation_summary_df = translate_summarize_enrich_df_emergency(base_df_min)

enrich_renamed = translation_summary_df.rename(
    columns={"Keywords": "Enriched Keywords", "Domain": "Enriched Domain"}
)

final_df = base_df_min.merge(
    enrich_renamed[
        [
            "URL", "Translated Title", "Translated Text", "Language",
            "Model Sentiment Label", "Model Sentiment Score",
            "Enriched Keywords", "Enriched Domain"
        ]
    ],
    on="URL",
    how="left"
)

print(f"📤 Final dataset shape: {final_df.shape}")

testing_new = dataiku.Dataset("translation_summary")
testing_new.write_with_schema(final_df)

print("🎉 EMERGENCY FAST MODE COMPLETE!")
print("✅ Features preserved: Translation ✓  Sentiment ✓  Keywords ✓")
print(f"⚡ Sentiment on up to ~{MAX_ARTICLE_CHARS:,} chars/article (tokenizer trims beyond 8 192 tokens)")