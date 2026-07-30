# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Academic project (Introduction to Data Science course) building a recipe recommendation system from Food.com-style data. Combines exploratory Jupyter notebooks (preprocessing, graphing, model iteration) with a Flask web app that serves recommendations.

- `app.py` — the Flask app. Loads `data/interactions_processed.csv`, `data/recipes_improved.csv`, and `data/recipes_processed_key.json`, builds per-user preference vectors, and serves recommendations via `/` and `/process` on port 5002.
- `optimized models/` — iterative recommender notebooks (`recommendation system1.ipynb` through `1.3.1.1.ipynb`).
- `preprocessing/` — notebooks that generate the processed CSV/JSON data files.
- `graphs/` — exploratory charts and the notebook that produced them.

## Environment

No `requirements.txt` or environment file exists. Run with plain Python 3 and pip-installed `numpy`, `pandas`, and `flask` — no version pinning in use.

## Gotchas

- **Which notebook version is canonical is unclear** — there are several iterative versions of the recommendation model (`recommendation system1.ipynb` → `1.3.1.1.ipynb`) and two website notebooks, and it's been a while since this was actively worked on. Don't assume the highest-numbered file is the one in use — check with the user or compare against `app.py` before treating any notebook as authoritative.
- **Don't casually regenerate or overwrite the committed data files** (`data/interactions_processed.csv`, `data/recipes_improved.csv`, `data/recipes_processed_key.json`). They're preprocessed outputs of the notebooks in `preprocessing/`; only rerun that pipeline if explicitly asked.
- No test suite or lint config exists in this repo.
