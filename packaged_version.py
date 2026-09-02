#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
packaged-version — currency-devaluation news pipeline (class-based, single file).

Combines the two former recipes into one module with two stages that can run
together or independently:

    stage 1  "collect"   GDELT DOC API            -> dataset `currency_devaluation_main`
    stage 2  "enrich"    translation + FinBERT + spaCy NER
                                                  -> dataset `translation_summary`

Entry points
------------
    run_pipeline()                 # both stages (default)
    collect(config)                # stage 1 only  -> DataFrame
    enrich(config, base_df=None)   # stage 2 only  -> DataFrame

Notes
-----
* The GDELT DOC API has no pagination (`startrecord` is ignored and it caps at
  `maxrecords=250`).  To get past 250 hits/country/day we split each day into
  `window_hours`-hour sub-windows and issue one request per window.
* spaCy / HF pipelines are not safe to call from multiple threads, so enrichment
  runs in two phases: parallel fetch+translate, then single-threaded model
  inference.
* Stage-2 dependencies (transformers, spacy, readability, bs4, langdetect,
  deep_translator, langcodes) are imported lazily so stage 1 can run without them.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse

import dataiku
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

__all__ = ["PipelineConfig", "collect", "enrich", "run_pipeline"]


# ============================================================================ #
# Configuration
# ============================================================================ #
DEFAULT_KEYWORDS = (
    "Currency Devaluation", "Exchange Rate", "Foreign Reserves", "Trade Balance",
    "Central Bank Intervention", "Speculative Attack", "Capital Flight",
    "External Debt", "Market Volatility", "Currency Depreciation",
)


@dataclass
class PipelineConfig:
    # ---- stage 1: collection ----
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    keywords: tuple = DEFAULT_KEYWORDS
    max_records: int = 250
    window_hours: int = 6                 # intra-day sub-window size
    collect_concurrency: int = 12
    flow_var_name: str = "DKU_DST_run_date"
    raw_dataset: str = "currency_devaluation_main"

    # ---- stage 2: enrichment ----
    enrich_workers: int = 10
    request_delay: float = 0.5            # min seconds between outbound requests
    connect_timeout: int = 10
    read_timeout: int = 15
    max_article_chars: int = 20_000       # body chars fed to translation + sentiment
    short_text_word_threshold: int = 50
    checkpoint_every: int = 50
    checkpoint_path: str = "enrichment_checkpoint.json"
    sentiment_model: str = "tabularisai/ModernFinBERT"
    sentiment_max_tokens: int = 8192
    spacy_model: str = "en_core_web_sm"
    keyword_labels: tuple = ("GPE", "ORG", "MONEY")
    enriched_dataset: str = "translation_summary"
    base_columns: tuple = ("URL", "Date", "Title", "Country", "Domain", "Keywords")

    # ---- shared HTTP ----
    http_retries: int = 3
    http_backoff: float = 0.5


# ============================================================================ #
# HTTP
# ============================================================================ #
class HttpClient:
    """Pooled ``requests.Session`` with adapter-level retries and an optional
    process-wide rate limit (thread-safe)."""

    def __init__(self, *, retries=3, backoff=0.5, timeout=15, pool=50,
                 user_agent="Mozilla/5.0", min_interval=0.0):
        self.timeout = timeout
        self._min_interval = float(min_interval)
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0

        self.session = requests.Session()
        # GET is retried by default; allowed_methods/method_whitelist naming
        # differs across urllib3 versions, so we don't pass it.
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

    def _throttle(self):
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait:
            time.sleep(wait)

    def get(self, url):
        """Single GET (the adapter already retries connection errors + 429/5xx)."""
        self._throttle()
        try:
            return self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            print(f"   ⚠️ request failed: {exc}")
            return None

    def get_json(self, url):
        resp = self.get(url)
        if resp is None:
            return None
        if resp.status_code != 200:
            print(f"   ⚠️ HTTP {resp.status_code} for {url[:100]}")
            return None
        try:
            return resp.json()
        except ValueError:
            print(f"   ⚠️ non-JSON response for {url[:100]}")
            return None

    @staticmethod
    def domain(url, strip_www=False):
        try:
            host = urlparse(url).netloc.lower()
            return host[4:] if strip_www and host.startswith("www.") else host
        except Exception:
            return ""


# ============================================================================ #
# Stage 1 — collection
# ============================================================================ #
class CountryManager:
    """FIPS 10-4 country codes used with GDELT ``sourcecountry:``."""

    _REGIONS = {
        "americas": ["AC", "AR", "AA", "BF", "BB", "BH", "BD", "BL", "BR", "CA", "CJ", "CI", "CO",
                     "CS", "CU", "DO", "DR", "EC", "ES", "GJ", "GT", "GY", "HA", "HO", "JM", "MX",
                     "NU", "PM", "PA", "PE", "RQ", "NS", "TD", "US", "UY", "VE"],
        "europe":   ["UK", "AL", "AN", "AU", "BO", "BE", "BK", "BU", "HR", "CY", "EZ", "DA", "EN",
                     "EU", "FO", "FI", "FR", "GM", "GR", "HU", "IC", "EI", "IM", "IT", "KV", "LG",
                     "LS", "LH", "LT", "LU", "MT", "MD", "MN", "MJ", "MW", "NL", "MK", "NO", "PL",
                     "PO", "RO", "RS", "LO", "SI", "SP", "SW", "SZ", "TU", "UP"],
        "africa":   ["AG", "AO", "BN", "BC", "UV", "BY", "CM", "CV", "CT", "CD", "CN", "CF", "DJ",
                     "EG", "EK", "ER", "ET", "GB", "GA", "GH", "GV", "PU", "IV", "KE", "LT", "LI",
                     "LY", "MA", "MI", "ML", "MR", "MP", "MO", "MZ", "WA", "NG", "NI", "CG", "RW",
                     "TP", "SG", "SE", "SL", "SO", "SF", "OD", "SU", "WZ", "TZ", "TG", "TS", "UG",
                     "ZA", "ZI"],
        "asia":     ["AF", "AM", "AJ", "BA", "BG", "BT", "BX", "CB", "CH", "TT", "GG", "HK", "IN",
                     "ID", "IR", "IZ", "IS", "JA", "JO", "KZ", "KU", "KG", "LA", "LE", "MC", "MY",
                     "MV", "MG", "BM", "NP", "KN", "MU", "WE", "PK", "RP", "QA", "SA", "SN", "KS",
                     "CE", "SY", "TW", "TI", "TH", "TX", "AE", "UZ", "VM", "YM"],
        "oceania":  ["AS", "FJ", "KR", "NC", "NZ", "PP", "WS", "BP", "TN", "NH"],
    }

    def __init__(self):
        self.codes = sorted({c for region in self._REGIONS.values() for c in region})
        print(f"🌍 Total country codes: {len(self.codes)}")


class DateWindow:
    """Current Mon–Sun week, split into ``window_hours``-hour sub-windows per day."""

    def __init__(self, config: PipelineConfig, today: datetime = None):
        if today is None:
            raw = dataiku.dku_flow_variables.get(config.flow_var_name) \
                  or datetime.utcnow().strftime("%Y-%m-%d")
            today = datetime.strptime(raw, "%Y-%m-%d")
        self.today = today
        self.start_of_week = today - timedelta(days=today.weekday())
        self.end_of_week = self.start_of_week + timedelta(days=6)
        self.window_hours = config.window_hours
        print(f"📅 Weekly window: {self.start_of_week.date()} → {self.end_of_week.date()} "
              f"({self.window_hours}h sub-windows)")

    def iter_days(self):
        d = self.start_of_week
        while d <= self.end_of_week:
            yield d
            d += timedelta(days=1)

    def day_windows(self, day: datetime):
        base = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = base + timedelta(hours=23, minutes=59, seconds=59)
        for h in range(0, 24, self.window_hours):
            w_start = base + timedelta(hours=h)
            w_end = min(w_start + timedelta(hours=self.window_hours) - timedelta(seconds=1), day_end)
            yield w_start.strftime("%Y%m%d%H%M%S"), w_end.strftime("%Y%m%d%H%M%S")


class GDELTClient:
    def __init__(self, http: HttpClient, config: PipelineConfig):
        self.http = http
        self.base_url = config.gdelt_base_url
        self.max_records = config.max_records
        self.keyword_query = "(" + " OR ".join(f'"{kw}"' for kw in config.keywords) + ")"

    def _url(self, code: str, start_str: str, end_str: str) -> str:
        query = quote_plus(f"{self.keyword_query} sourcecountry:{code}")
        return (
            f"{self.base_url}?query={query}"
            f"&startdatetime={start_str}&enddatetime={end_str}"
            f"&format=json&maxrecords={self.max_records}&mode=artlist"
        )

    def fetch_country_day(self, day, windows, code, seen_urls: set, lock: threading.Lock):
        """One country across every sub-window of ``day``. Returns (rows, api_calls)."""
        rows, calls = [], 0
        iso_date = day.strftime("%Y-%m-%dT00:00:00.000Z")

        for start_str, end_str in windows:
            data = self.http.get_json(self._url(code, start_str, end_str))
            calls += 1
            if not isinstance(data, dict):
                continue

            for art in data.get("articles") or []:
                url = (art.get("url") or "").strip()
                if not url:
                    continue
                with lock:                       # thread-safe check-and-add
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                rows.append({
                    "Date": iso_date,
                    "Title": art.get("title") or "",
                    "URL": url,
                    "Domain": self.http.domain(url),
                    "Country": art.get("sourcecountry") or "Unknown",
                    "Keywords": self._themes(art.get("themes")),
                })

        return rows, calls

    @staticmethod
    def _themes(themes):
        if isinstance(themes, list):
            return ", ".join(sorted({t for t in themes if isinstance(t, str) and t}))
        if isinstance(themes, str):
            return themes
        return ""


class ArticleCollector:
    _COLS = ["Date", "Title", "URL", "Domain", "Country", "Keywords"]

    def __init__(self, gdelt: GDELTClient, countries: CountryManager,
                 date_window: DateWindow, config: PipelineConfig):
        self.gdelt = gdelt
        self.countries = countries.codes
        self.date_window = date_window
        self.concurrency = config.collect_concurrency

    def run(self) -> pd.DataFrame:
        rows, total_calls = [], 0
        seen_urls: set = set()
        lock = threading.Lock()

        for day in tqdm(list(self.date_window.iter_days()), desc="Days", unit="day"):
            windows = list(self.date_window.day_windows(day))
            print(f"📆 {day.date()} — {len(windows)} windows × {len(self.countries)} countries")

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self.gdelt.fetch_country_day, day, windows, code, seen_urls, lock): code
                    for code in self.countries
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"Countries {day.date()}", leave=False, unit="cty"):
                    code = futures[fut]
                    try:
                        day_rows, calls = fut.result()
                        rows.extend(day_rows)
                        total_calls += calls
                    except Exception as exc:
                        print(f"[{day.date()}][{code}] ❌ {exc}")

        print(f"✅ Collected {len(rows)} rows in {total_calls} API calls")
        df = pd.DataFrame(rows, columns=self._COLS)
        return df.drop_duplicates(subset=["URL"]).reset_index(drop=True)


# ============================================================================ #
# Dataset I/O
# ============================================================================ #
class DatasetIO:
    @staticmethod
    def read(name: str, columns=None, dedupe_on=None) -> pd.DataFrame:
        df = dataiku.Dataset(name).get_dataframe()
        print(f"📥 '{name}': {len(df)} rows")
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep].copy()
            print(f"📋 columns: {keep}")
        if dedupe_on and dedupe_on in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[dedupe_on]).reset_index(drop=True)
            if before != len(df):
                print(f"🧹 dropped {before - len(df)} duplicate {dedupe_on}")
        return df

    @staticmethod
    def write(name: str, df: pd.DataFrame):
        if df is None or df.empty:
            print(f"⚠️ '{name}': nothing to write — skipping")
            return
        dataiku.Dataset(name).write_with_schema(df)
        print(f"💾 '{name}': wrote {len(df)} rows")


# ============================================================================ #
# Stage 2 — enrichment
# ============================================================================ #
class HtmlExtractor:
    def __init__(self):
        from readability import Document
        from bs4 import BeautifulSoup
        self._Document = Document
        self._BeautifulSoup = BeautifulSoup

    def main_text(self, html: str) -> str:
        try:
            doc = self._Document(html)
            soup = self._BeautifulSoup(doc.summary(), "lxml")
            return soup.get_text().strip()
        except Exception:
            return ""


class NLPProcessor:
    """Loads the heavy models once. Model calls (`sentiment_of`, `keywords_of`)
    must be made from a single thread."""

    def __init__(self, config: PipelineConfig):
        from transformers import pipeline
        import spacy
        from langdetect import detect, DetectorFactory
        from deep_translator import GoogleTranslator
        from langcodes import Language

        DetectorFactory.seed = 0            # deterministic language detection
        self._detect = detect
        self._Translator = GoogleTranslator
        self._Language = Language
        self._chunk_size = 4_500
        self._keyword_labels = tuple(config.keyword_labels)

        print("🔄 loading NLP models …")
        # truncation/max_length at construction so they apply on every transformers version
        self.sentiment = pipeline(
            "sentiment-analysis",
            model=config.sentiment_model,
            truncation=True,
            max_length=config.sentiment_max_tokens,
        )
        self.nlp = spacy.load(config.spacy_model)
        print("✅ NLP models ready")

    # ---- language ----
    def lang_code(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"
        try:
            return self._detect(text)
        except Exception:
            return "unknown"

    def lang_name(self, code: str) -> str:
        try:
            return self._Language.get(code).display_name()
        except Exception:
            return "Unknown"

    # ---- translation (I/O, thread-safe) ----
    def translate_short(self, text: str) -> str:
        if not text:
            return text
        try:
            return self._Translator(source="auto", target="en").translate(text) or text
        except Exception:
            return text

    def translate_long(self, text: str, retries: int = 2) -> str:
        if not text:
            return ""
        chunks = [text[i:i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]
        out = []
        for chunk in chunks:
            for attempt in range(retries):
                try:
                    # some deep_translator versions return None on a "successful" call
                    out.append(self._Translator(source="auto", target="en").translate(chunk) or chunk)
                    break
                except Exception as exc:
                    print(f"   ⚠️ translation attempt {attempt + 1} failed: {exc}")
                    if attempt < retries - 1:
                        time.sleep(0.5)
            else:
                out.append(chunk)
        return " ".join(out)

    # ---- models (single-thread only) ----
    def sentiment_of(self, text: str):
        if not text or not text.strip():
            return "[SENTIMENT FAILED]", None
        try:
            res = self.sentiment(text)[0]
            return str(res["label"]).strip().lower(), round(float(res["score"]), 2)
        except Exception:
            return "[SENTIMENT FAILED]", None

    def keywords_of(self, text: str) -> str:
        if not text:
            return ""
        try:
            ents = self.nlp(text).ents
            return ", ".join(sorted({e.text for e in ents if e.label_ in self._keyword_labels}))
        except Exception:
            return ""


class ArticleEnricher:
    """Phase 1 (thread-safe): fetch + translate.
    Phase 2 (single-thread): sentiment + NER."""

    def __init__(self, http: HttpClient, extractor: HtmlExtractor,
                 nlp: NLPProcessor, config: PipelineConfig):
        self.http = http
        self.extractor = extractor
        self.nlp = nlp
        self.max_chars = config.max_article_chars
        self.word_threshold = config.short_text_word_threshold

    # ---------- phase 1 ----------
    def fetch_and_translate(self, row: dict):
        title = str(row.get("title") or row.get("Title") or "")
        url = str(row.get("url") or row.get("URL") or "")
        if not url.startswith("http") or "video" in url.lower() or len(url) < 10:
            return None

        try:
            domain = self.http.domain(url, strip_www=True)
            translated_title = self.nlp.translate_short(title)

            resp = self.http.get(url)
            if resp is None or not getattr(resp, "ok", False):
                return {
                    "URL": url,
                    "Translated Title": translated_title or title,
                    "Translated Text": translated_title or title,
                    "Language": self.nlp.lang_name(self.nlp.lang_code(title)),
                    "Domain": domain,
                    "_status": "no_content",
                }

            text_full = self.extractor.main_text(resp.text)
            basis = text_full if text_full and len(text_full) > 50 else title
            code = self.nlp.lang_code(basis)
            is_en = code == "en"

            if not text_full or len(text_full.split()) < self.word_threshold:
                src = text_full or ""
                body = (src if (src and is_en) else self.nlp.translate_long(src)) \
                       or translated_title or title
            else:
                capped = text_full[:self.max_chars]
                body = capped if is_en else self.nlp.translate_long(capped)

            return {
                "URL": url,
                "Translated Title": translated_title or title,
                "Translated Text": body or translated_title or title,
                "Language": self.nlp.lang_name(code),
                "Domain": domain,
                "_status": "ok",
            }
        except Exception as exc:
            print(f"❌ fetch/translate failed for {url}: {exc}")
            return None

    # ---------- phase 2 ----------
    def add_sentiment(self, partial: dict) -> dict:
        text = partial["Translated Text"] or ""
        if partial.get("_status") == "no_content":
            label, score = "[failed - no content]", None
        else:
            label, score = self.nlp.sentiment_of(text)
        return {
            "URL": partial["URL"],
            "Translated Title": partial["Translated Title"],
            "Translated Text": text,
            "Language": partial["Language"],
            "Model Sentiment Label": label,
            "Model Sentiment Score": score,
            "Keywords": self.nlp.keywords_of(text),
            "Domain": partial["Domain"],
        }


class EnrichmentPipeline:
    OUTPUT_COLS = [
        "URL", "Translated Title", "Translated Text", "Language",
        "Model Sentiment Label", "Model Sentiment Score", "Keywords", "Domain",
    ]

    def __init__(self, enricher: ArticleEnricher, config: PipelineConfig):
        self.enricher = enricher
        self.workers = config.enrich_workers
        self.checkpoint_every = config.checkpoint_every
        self.checkpoint_path = config.checkpoint_path

    def _checkpoint(self, processed: int, results: list):
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as fh:
                json.dump({"processed": processed, "results": results, "ts": time.time()},
                          fh, ensure_ascii=False)
            print(f"✅ checkpoint: {processed} items")
        except Exception as exc:
            print(f"⚠️ checkpoint failed: {exc}")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = df.to_dict(orient="records")
        total = len(rows)
        if total == 0:
            print("⚠️ no rows to enrich")
            return pd.DataFrame(columns=self.OUTPUT_COLS)

        print(f"🚀 enriching {total} articles | workers={self.workers} "
              f"| checkpoint → {self.checkpoint_path}")
        start = time.time()

        # phase 1 — parallel fetch + translation (I/O bound)
        partials = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.enricher.fetch_and_translate, r) for r in rows]
            for fut in tqdm(as_completed(futures), total=total, desc="Fetch+translate"):
                p = fut.result()
                if p:
                    partials.append(p)

        # phase 2 — sequential sentiment + NER (models are not thread-safe)
        results = []
        for i, p in enumerate(tqdm(partials, desc="Sentiment+NER"), start=1):
            results.append(self.enricher.add_sentiment(p))
            if i % self.checkpoint_every == 0:
                self._checkpoint(i, results)
        self._checkpoint(len(results), results)

        elapsed = time.time() - start
        print(f"\n✅ enriched {len(results)}/{total} "
              f"({len(results) / max(total, 1) * 100:.1f}%) in {elapsed / 60:.1f} min")
        if elapsed > 0:
            print(f"🚀 {len(results) / elapsed * 60:.1f} articles/min")
        return pd.DataFrame(results, columns=self.OUTPUT_COLS)


# ============================================================================ #
# Orchestration
# ============================================================================ #
def collect(config: PipelineConfig = None, today: datetime = None) -> pd.DataFrame:
    """Stage 1: pull articles from GDELT and write `raw_dataset`."""
    config = config or PipelineConfig()
    http = HttpClient(retries=config.http_retries, backoff=config.http_backoff,
                      timeout=config.read_timeout, pool=50)
    collector = ArticleCollector(
        gdelt=GDELTClient(http, config),
        countries=CountryManager(),
        date_window=DateWindow(config, today=today),
        config=config,
    )
    df = collector.run()
    DatasetIO.write(config.raw_dataset, df)
    return df


def enrich(config: PipelineConfig = None, base_df: pd.DataFrame = None) -> pd.DataFrame:
    """Stage 2: translate + score + tag, then write `enriched_dataset`."""
    config = config or PipelineConfig()

    if base_df is None:
        base_df = DatasetIO.read(config.raw_dataset,
                                 columns=config.base_columns, dedupe_on="URL")
    else:
        keep = [c for c in config.base_columns if c in base_df.columns]
        base_df = base_df[keep].copy()
        if "URL" in base_df.columns:
            before = len(base_df)
            base_df = base_df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
            if before != len(base_df):
                print(f"🧹 dropped {before - len(base_df)} duplicate URLs")

    http = HttpClient(
        retries=2, backoff=0.3,
        timeout=(config.connect_timeout, config.read_timeout),
        pool=20, min_interval=config.request_delay,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    enricher = ArticleEnricher(http, HtmlExtractor(), NLPProcessor(config), config)
    enriched = EnrichmentPipeline(enricher, config).run(base_df)

    if enriched.empty:
        print("⚠️ enrichment produced no rows — writing base columns only")
        DatasetIO.write(config.enriched_dataset, base_df)
        return base_df

    enriched = enriched.rename(columns={"Keywords": "Enriched Keywords",
                                        "Domain": "Enriched Domain"})
    merge_cols = ["URL", "Translated Title", "Translated Text", "Language",
                  "Model Sentiment Label", "Model Sentiment Score",
                  "Enriched Keywords", "Enriched Domain"]
    final_df = base_df.merge(enriched[merge_cols], on="URL", how="left")
    print(f"📤 final dataset shape: {final_df.shape}")
    DatasetIO.write(config.enriched_dataset, final_df)
    return final_df


def run_pipeline(config: PipelineConfig = None, today: datetime = None,
                 stages=("collect", "enrich")):
    config = config or PipelineConfig()
    print("🔄 Starting currency-devaluation pipeline …")

    base_df = None
    if "collect" in stages:
        base_df = collect(config, today=today)
    if "enrich" in stages:
        enrich(config, base_df=base_df)

    print("🎉 Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
