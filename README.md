# Currency Devaluation Sentiment Analysis

**Can news sentiment predict currency devaluation before it happens?** This project explores whether media sentiment around financial and political events can help predict or explain currency devaluation risk in frontier markets — and whether that signal could support early-warning decisions.

## Pipeline Overview

1. **News Collection** — Scrapes global news articles (GDELT API) mentioning currency-related keywords (devaluation, exchange rate, capital flight, etc.) across frontier-market countries. Runs on a weekly cycle — triggered with a single click and automatically populates the dataset, so results are ready for stakeholders to review without manual intervention
2. **Translation & Sentiment** — Translates non-English articles, runs financial sentiment analysis (ModernFinBERT) on article text up to 8,192 tokens, and extracts key entities (locations, organizations, monetary figures)
3. **Scoring & Aggregation** — Converts sentiment into a signed score, filters for high-confidence predictions, and aggregates by country and month
4. **Macroeconomic Integration** — Combines sentiment signals with macroeconomic indicators, forex rates, and financial data (different team) to assess devaluation risk

## My Contribution

I built the news collection, sentiment analysis, and scoring pipeline:

- `data_collection.py` — scrapes news articles from the GDELT API across frontier-market countries using currency-related keywords
- `translation_and_sentiment.py` — translation pipeline, ModernFinBERT sentiment scoring, and entity extraction from scraped articles
- `currency_score_groupby.py` — converts sentiment labels into a signed score, filters for high-confidence results, and aggregates by country/month for downstream analysis

## Automation

The collection step is built around a weekly run cycle, using a flow-managed date variable to automatically compute the correct Monday–Sunday window each time it runs. A single click kicks off the full week's scrape across all countries in parallel, and results are written directly into the dataset — no manual pulling or reformatting needed, so leadership and non-technical stakeholders can view up-to-date results on demand.

## Team

This was a collaborative project:

- **News collection, sentiment analysis & scoring** — built by me
- **Macroeconomic indicators, forex, and financial data integration** — built by the team
- The collection and sentiment pipeline was later packaged into an object-oriented version, worked on together with an intern from a different department (see `packaged-version/`)

## Packaged Version

The `packaged-version/` folder contains an OOP-refactored version of the pipeline, built by me and a teammate. I identified its potential for reuse across different asset classes and decided to turn it into a reusable package for production use. The functional scripts above reflect my original implementation and reasoning.

## Tech Stack

- Python, Dataiku (workflow orchestration)
- `transformers` (ModernFinBERT — `tabularisai/ModernFinBERT`)
- `spaCy` (named entity recognition)
- `deep-translator`, `langdetect`, `langcodes` (translation & language detection)
- `readability-lxml`, `BeautifulSoup` + `lxml` (main article text extraction from raw HTML)
- `pandas`, `numpy`
- `requests` with retry/backoff logic (`urllib3`), threaded with `concurrent.futures` for parallel scraping
- `tqdm` (progress tracking)

## Notes

Built as part of a team project focused on frontier-market economic risk signals. Dataset names and some configuration details have been generalized/simplified for this public repo.
