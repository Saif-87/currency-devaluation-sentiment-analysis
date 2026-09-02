# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote_plus
import time
import random
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def make_session(total_retries=3, backoff=0.5):
    """Create a Session with connection pooling + retries (idempotent GET)."""
    session = requests.Session()
    # GET is in urllib3's default retryable-method set, so no need to pass
    # allowed_methods / method_whitelist (the name differs across urllib3 versions).
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

def safe_request(url, retries=2, backoff=2, timeout=15, session=None):
    """GET with a few manual retries layered on top of the session's adapter-level Retry.

    The HTTPAdapter already retries connection errors and 429/5xx responses;
    this loop only adds a light second layer and never sleeps after the final try.
    """
    client = session or requests
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"⚠️ Status {resp.status_code}, attempt {attempt+1}/{retries}")
        except requests.RequestException as e:
            print(f"⚠️ Request error: {e}, attempt {attempt+1}/{retries}")
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1) * (1 + 0.2 * random.random()))
    return None

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Countries (edit as needed)
americas = ["AC","AR","AA","BF","BB","BH","BD","BL","BR","CA","CJ","CI","CO","CS","CU","DO","DR","EC","ES","GJ","GT","GY","HA","HO","JM","MX","NU","PM","PA","PE","RQ","NS","TD","US","UY","VE"]
europe   = ["UK","AL","AN","AU","BO","BE","BK","BU","HR","CY","EZ","DA","EN","EU","FO","FI","FR","GM","GR","HU","IC","EI","IM","IT","KV","LG","LS","LH","LT","LU","MT","MD","MN","MJ","MW","NL","MK","NO","PL","PO","RO","RS","LO","SI","SP","SW","SZ","TU","UP"]
africa   = ["AG","AO","BN","BC","UV","BY","CM","CV","CT","CD","CN","CF","DJ","EG","EK","ER","ET","GB","GA","GH","GV","PU","IV","KE","LT","LI","LY","MA","MI","ML","MR","MP","MO","MZ","WA","NG","NI","CG","RW","TP","SG","SE","SL","SO","SF","OD","SU","WZ","TZ","TG","TS","UG","ZA","ZI"]
asia     = ["AF","AM","AJ","BA","BG","BT","BX","CB","CH","TT","GG","HK","IN","ID","IR","IZ","IS","JA","JO","KZ","KU","KG","LA","LE","MC","MY","MV","MG","BM","NP","KN","MU","WE","PK","RP","QA","SA","SN","KS","CE","SY","TW","TI","TH","TX","AE","UZ","VM","YM"]
oceania  = ["AS","FJ","KR","NC","NZ","PP","WS","BP","TN","NH"]
country_codes_raw = americas + europe + africa + asia + oceania
country_codes = sorted(set(country_codes_raw))
print(f"🌍 Total country codes: {len(country_codes)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Weekly window from Dataiku flow variable (fall back to current UTC date if unset)
today = dataiku.dku_flow_variables.get("DKU_DST_run_date") or datetime.utcnow().strftime("%Y-%m-%d")
today = datetime.strptime(today, "%Y-%m-%d")
start_of_week = today - timedelta(days=today.weekday())   # Monday
end_of_week   = start_of_week + timedelta(days=6)         # Sunday
print(f"📅 Weekly window: {start_of_week.date()} → {end_of_week.date()}")

# Precompute all days in the weekly window
days = []
d = start_of_week
while d <= end_of_week:
    days.append(d)
    d += timedelta(days=1)

# GDELT's DOC API caps a query at maxrecords=250 and has no pagination
# (startrecord is ignored). To get past 250 hits/country/day we instead split
# each day into intra-day windows and issue one request per window.
WINDOW_HOURS = 6

def day_windows(day):
    """Yield (startdatetime, enddatetime) GDELT strings covering one calendar day."""
    base = day.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = base + timedelta(hours=23, minutes=59, seconds=59)
    for h in range(0, 24, WINDOW_HOURS):
        w_start = base + timedelta(hours=h)
        w_end = min(w_start + timedelta(hours=WINDOW_HOURS) - timedelta(seconds=1), day_end)
        yield w_start.strftime('%Y%m%d%H%M%S'), w_end.strftime('%Y%m%d%H%M%S')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Keywords and API parameters
keywords = [
    "Currency Devaluation",
    "Exchange Rate",
    "Foreign Reserves",
    "Trade Balance",
    "Central Bank Intervention",
    "Speculative Attack",
    "Capital Flight",
    "External Debt",
    "Market Volatility",
    "Currency Depreciation"
]

keyword_query = "(" + " OR ".join([f'"{keyword}"' for keyword in keywords]) + ")"
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250

# Tunables
CONCURRENCY = 12          # parallel countries per day
REQUEST_RETRIES = 2       # manual retries on top of the adapter-level Retry
REQUEST_BACKOFF = 2       # base backoff seconds
REQUEST_TIMEOUT = 15      # per-request timeout

# Early dedupe across all days/countries (guarded for thread-safe check-and-add)
seen_urls = set()
seen_urls_lock = threading.Lock()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def fetch_country_for_day(current_day, code, session):
    """Fetch all intra-day windows for a single country; returns list of rows + request count."""
    rows_local = []
    total_calls = 0
    raw_query = f"{keyword_query} sourcecountry:{code}"
    encoded_query = quote_plus(raw_query)

    for start_str, end_str in day_windows(current_day):
        gdelt_url = (
            f"{BASE_URL}"
            f"?query={encoded_query}"
            f"&startdatetime={start_str}"
            f"&enddatetime={end_str}"
            f"&format=json"
            f"&maxrecords={MAX_RECORDS}"
            f"&mode=artlist"
        )

        resp = safe_request(
            gdelt_url,
            retries=REQUEST_RETRIES, backoff=REQUEST_BACKOFF, timeout=REQUEST_TIMEOUT, session=session
        )
        total_calls += 1
        if resp is None:
            continue

        try:
            data = resp.json()
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue

        articles = data.get("articles") or []
        if not articles:
            continue

        for a in articles:
            url = a.get("url", "") or ""
            if not url:
                continue
            # Global dedupe (thread-safe check-and-add)
            with seen_urls_lock:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            title = a.get("title", "") or ""
            domain = get_domain(url)
            country_name = a.get("sourcecountry", "Unknown") or "Unknown"

            themes = a.get("themes", [])
            if isinstance(themes, list):
                keywords_joined = ", ".join(sorted({t for t in themes if isinstance(t, str) and t}))
            elif isinstance(themes, str):
                keywords_joined = themes
            else:
                keywords_joined = ""

            rows_local.append({
                "Date": current_day.strftime("%Y-%m-%dT00:00:00.000Z"),
                "Title": title,
                "URL": url,
                "Domain": domain,
                "Country": country_name,
                "Keywords": keywords_joined
            })

    return rows_local, total_calls

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
session = make_session(total_retries=3, backoff=0.5)
rows = []
total_requests = 0

# Loop per day, then parallelize per country inside
for current_day in tqdm(days, desc="Days", unit="day"):
    print(f"📆 Running for {current_day.date()}")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(fetch_country_for_day, current_day, code, session): code
            for code in country_codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Countries {current_day.date()}", leave=False, unit="cty"):
            code = futures[fut]
            try:
                rows_local, calls = fut.result()
                if rows_local:
                    rows.extend(rows_local)
                total_requests += calls
            except Exception as e:
                print(f"[{current_day.date()}][{code}] ❌ Error: {e}")

print(f"✅ Completed {len(rows)} rows, {total_requests} API calls "
      f"(WINDOW_HOURS={WINDOW_HOURS}, TIMEOUT={REQUEST_TIMEOUT}s)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.DataFrame(rows, columns=["Date","Title","URL","Domain","Country","Keywords"])
df.drop_duplicates(subset=["URL"], inplace=True)

currency_devalue_main = dataiku.Dataset("currency_devaluation_main")
currency_devalue_main.write_with_schema(df)

print(f"💾 Saved {len(df)} rows for week {start_of_week.date()} → {end_of_week.date()} into 'currency_devaluation_main'")
