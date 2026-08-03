# Roadmap: getting food-for-thought portfolio-ready

Goal: turn this from a two-year-old university project into something a stranger
(recruiter, other dev) can open, understand in five minutes, and actually try.

Current state, for reference:
- `app.py` is the live app: loads three data files, builds
  per-user ingredient/tag preference vectors, exposes `/` (a raw inline HTML
  form) and `/process` (returns JSON) on port 5002.
- Data files are large and committed straight into git with no `.gitignore`:
  `data/interactions_processed.csv` (18MB, ~1.07M rows), `data/recipes_improved.csv`
  (40MB, ~232k rows), `data/recipes_processed_key.json` (76KB).
- ~~Five near-duplicate notebooks in `optimized models/` (`recommendation
  system1.ipynb` → `1.3.1.1.ipynb`), plus two `Food_for_thought_website*.ipynb`
  notebooks~~ — resolved, see §0/§1 below. `preprocessing/` and `graphs/`
  notebooks remain.
- `README.md` now has rough working notes on what the recommender does and
  how (`app.py`'s pipeline), still needs a proper pass. No licence, no
  `requirements.txt`, no tests, no CI.

---

## 0. Re-orient yourself first

Do this before touching anything else — you can't clean up or document code you
don't remember.

- [x] Open each notebook in `optimized models/` and note, per file: what it
      changed vs. the previous version, and whether it still runs top-to-bottom.
      Done: `1` (ingredients only, unnormalised) → `1.2` (adds tags, slow
      loop-based) → `1.3` (dense NumPy matrix, vectorised, fast) → `1.3.1`
      (float32 dtype tweak) → `1.3.1.1` (transposed matrix layout). All five
      deleted afterward — `app.py` supersedes them (see §1).
- [x] Diff `Food_for_thought_website.ipynb` vs `Food_for_thought_website_v2.ipynb`
      vs `app.py` — confirm the `.py` file really is the
      most recent/correct logic (CLAUDE.md flags this as unverified).
      Confirmed: v1 had a tag-count off-by-one, `uint8` overflow risk, no
      already-rated exclusion, and a redundant separate `RAW_recipes.csv`
      load; v2 fixed all of that and matches `app.py`'s logic almost
      exactly except v2 still used dense matrices where `app.py` uses
      `scipy.sparse`. Both notebooks deleted (see §1).
- [x] Trace the `preprocessing/` notebooks against the three committed data
      files — confirm which notebook produces which output file, so the
      pipeline is reconstructable later.
      Done: `Interactions_processing.ipynb` → `data/interactions_processed.csv`
      directly (confirmed, then deleted — output already committed).
      `preprocessing/tags_preprocessing.ipynb` → very likely feeds `data/recipes_improved.csv`
      (empirically verified: none of its 49 flagged-for-removal tag ids
      appear anywhere in the committed data) — kept, moved to repo root.
      `ingredients_first_word.ipynb` → NOT used (verified: `data/recipes_improved.csv`
      uses the original 3,122-entry ingredient key, not this notebook's
      consolidated vocabulary) — deleted. `preprocessing/` folder removed
      since empty.
- [x] Write down (even just in a scratch file) what the recommender actually
      does: per-user mean rating per ingredient/tag, recipes scored by
      average of (ingredient-affinity, tag-affinity), cosine-free dot product
      — whatever it turns out to be. You'll need this for the README and any
      write-up.
      Done — rough notes now in `README.md` (still needs a polish pass, see §2).

## 1. File clean-up

- [x] Decide which notebook in `optimized models/` is canonical. Delete or
      move the rest into an `archive/` folder (or a git tag/branch) so the
      repo doesn't look like five abandoned attempts sitting side by side.
      Done — `app.py` is canonical, all five deleted, `optimized models/`
      folder gone.
- [x] Same decision for `Food_for_thought_website.ipynb` vs `_v2.ipynb` —
      likely delete both once the logic is confirmed to live in the `.py`
      file, or keep one as "exploration" and clearly label it as such.
      Done — both deleted, logic confirmed to live in `app.py`.
- [x] Rename `app.py` → `app.py` (or `server.py`). Keeping
      a typo'd filename as "the actual name" is a fine team-only shrug; it's
      not a great first impression for an external visitor. Update CLAUDE.md
      and any references after renaming.
      Done — `website.py` renamed to `app.py` (staged rename), and CLAUDE.md's
      stale "note the typo in the filename" line removed since it no longer
      applies.
- [x] Add a `.gitignore`: `__pycache__/`, `*.pyc`, `.ruff_cache/`, `.idea/`,
      `.ipynb_checkpoints/`. `.idea/` and `__pycache__/` are currently
      untracked-but-present in your working tree — get them out before they
      get committed by accident.
      Done — `.gitignore` added at repo root with those five patterns.
- [x] Decide what happens to the two large CSVs (58MB combined) — see
      section 4 (Data) before deciding; don't just leave them sitting in git
      history unaddressed.
      Done — see §4: keeping them committed as plain tracked files, licence
      confirmed to permit it.
- [x] Group loose top-level notebooks/scripts into folders consistent with
      the existing `preprocessing/`, `graphs/`, `optimized models/` pattern
      (e.g. `app/` for the Flask code once it grows beyond one file).
      Deferred for now — `app.py` and `preprocessing/tags_preprocessing.ipynb` are each
      single files at the root; folders for one file apiece (especially
      recreating the `preprocessing/` folder just removed) add no value.
      Revisit if either grows.

## 2. Documentation

- [ ] Replace the one-line `README.md` with:
  - What the project does (one paragraph) and a link to the live demo once
    it exists.
  - A screenshot or short GIF of the app in use.
  - The dataset it's built on and a link/citation to the source.
  - How the recommender works, in plain language (2-4 sentences) — this is
    the single most valuable paragraph for a recruiter skimming the repo.
  - Repo layout: what `preprocessing/`, `optimized models/`, `graphs/`, and
    the app file each contain.
  - Local setup instructions (once `requirements.txt` exists — see below).
  - Known limitations (e.g. cold-start for users with no interactions,
    dataset is static/not live food.com data).
- [ ] Add a `requirements.txt` or `pyproject.toml` pinning `numpy`, `pandas`,
      `flask`, and anything the frontend build needs. CLAUDE.md notes none
      exists today.
- [ ] Add a `LICENSE` for your own code (MIT is the standard default for a
      portfolio project) — separate from the dataset's licence (see below).
- [ ] Add module/function docstrings to the recommender functions
      (`parseReviews` and `generateRecommendations` in `app.py`,
      `vectorizeRecipes` in `recipe_vectors.py`) — short, explaining the
      *why* (e.g. why incremental mean rather than a stored sum) not just
      restating the code.
- [ ] Show a couple of the `graphs/` charts in the README as "what we found
  in EDA" — they're already made, just not surfaced anywhere outside the
  notebook.
- [ ] A short "how the model was built" write-up (even a paragraph) linking
  to the preprocessing/model-iteration notebooks, for anyone who wants to
  go deeper than the README summary.

## 3. Backend / code quality

Small but real correctness and robustness gaps found while reading
`app.py`:

- [x] `generateRecommendations` divides by `nIngredients` with no
      zero-guard (only `nTags` is protected via `np.maximum(nTags, 1)`) — a
      recipe with zero listed ingredients would produce `NaN`/`inf` and could
      poison the ranking.
      Done — `nIngredients` now guarded the same way as `nTags`.
- [x] Flask is started unconditionally at import time in a background thread
      (`Thread(target=run_app).start()` at module scope) rather than behind
      `if __name__ == "__main__":` — looks like it was written to run in a
      notebook/Colab cell. Fine there, awkward for a normal script/deploy
      target; worth switching to a standard entrypoint.
      Done — replaced with `if __name__ == "__main__": run_app()`; unused
      `Thread` import removed.
- [x] Leftover `print(data["name"].head(15))` debug line in `parseReviews`.
      Done — removed.

## 4. Data

- [x] Check the licensing terms of the underlying Food.com dataset (this
      looks like the Kaggle "Food.com Recipes and Interactions" dataset) —
      confirm redistribution of the processed CSVs in a public repo is
      permitted before making the repo public, or before hosting them as a
      downloadable/servable artefact. If redistribution isn't clearly
      allowed, keep the raw/processed data out of the repo and document how
      to regenerate it instead.
      Done — confirmed the licence permits redistribution.
- [x] Given the 58MB combined size, decide: keep the files in the repo (fine
      up to GitHub's 100MB/file soft limits, but bloats every clone), move
      them to Git LFS, or exclude them from git entirely and document a
      download/generation step in the README. Whichever you pick, don't
      leave the current "large binary-ish files with no `.gitignore`, no
      LFS, no explanation" state as-is.
      Done — keeping the files committed as plain tracked files (not LFS,
      not excluded). Priority is a visitor being able to `git clone` and run
      immediately with zero extra tooling/steps; the ~58MB clone cost is an
      accepted tradeoff at this scale.
- [x] If you do trim/republish the data, note that it's already in git
      history from earlier commits — removing it going forward doesn't
      remove it from history. Only worth rewriting history if repo size or
      the licensing question above forces it.
      N/A — decided above to keep the data committed as-is, not trim or
      republish it, so there's nothing to rewrite history for.

## 5. Frontend & hosting

**Decision made:** single hosted app on **Render** — one Flask app serves
both the frontend and the recommendation logic from the same origin
(no GitHub Pages, no split frontend/backend, no CORS needed). Render
connects directly to the GitHub repo and redeploys on push. Free tier is
~512MB RAM and spins down after inactivity (30-60s cold start on the first
request after idling) — acceptable for a portfolio project, revisit
(paid tier, or Fly.io which doesn't sleep) only if that becomes a problem.

- [x] Build a real frontend: a form to enter/pick a user ID, a results view
      (recipe name, image if available, score, link out to food.com) instead
      of raw JSON. Doesn't need to be fancy — clean and legible beats
      elaborate for a CV project.
      Done — `templates/index.html` + `static/{style.css,app.js}`. Form
      posts to `/process` via `fetch`, results render as a card grid (rank,
      image, star rating, name, link out). `/process` scrapes each
      recommended recipe's food.com page for its `og:image` meta tag
      (in-memory cache keyed by recipe id, since recipes recur across users'
      recommendations) and falls back to a plain icon when a recipe has no
      photo of its own (food.com serves a generic share-card graphic in that
      case — detected and treated as "no image" rather than shown).
- [x] Add basic loading/error states in the frontend (backend cold-start,
      invalid/unknown user ID, no recommendations available).
      Done — spinner while `/process` is in flight (message upgrades to a
      cold-start notice after 4s), red banner for server-side errors, and an
      empty-state message when a user has zero recommendations.
- [x] Set up the actual Render deployment (see §5b below for steps).
- [x] Confirm the recommender's current in-memory, load-everything-at-import
      approach fits comfortably in Render's free-tier RAM limit, or switch to
      precomputed/cached vectors if it doesn't (see the caching item in §5c).
      Done — `vectorizeRecipes` and the `url` column are now precomputed
      offline by `build_recipe_vectors.py` into `data/recipes_vectors.pkl`,
      which `app.py` just loads at import. Startup dropped from ~30s+ to
      ~3.5s; RAM footprint at import is now dominated by the CSVs and the
      pickled sparse matrices rather than a transient Python-list build.

### 5b. Render deployment steps

- [x] Add a `requirements.txt` (tracked separately in §2) — Render needs this
      to install dependencies.
      Done — `flask`, `numpy`, `pandas`, `scipy`, `gunicorn`.
- [x] Make sure `app.py` binds to Render's `$PORT` env var rather than the
      hardcoded `5002`, e.g. `app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))`.
      Done.
- [x] Push the repo to GitHub (already done) — Render deploys straight from
      a connected GitHub repo, no separate CI config needed.
- [x] In the Render dashboard: New → Web Service → connect the GitHub repo.
- [x] Set the **Build Command** to `pip install -r requirements.txt`.
- [x] Set the **Start Command** to
      `gunicorn app:app --bind 0.0.0.0:$PORT` (production WSGI server —
      don't rely on Flask's dev server / `app.run()` in production; gunicorn
      needs the explicit `--bind` since it doesn't read `$PORT` on its own).
      A `--timeout` bump is no longer required for worker boot: module-level
      import used to do a 40MB CSV read plus a plain Python loop over ~232k
      recipes in `vectorizeRecipes`, which could take well past gunicorn's
      default 30s worker-boot timeout on free-tier CPU (Render would report
      "no open ports detected" forever because every worker got killed
      mid-boot before it bound). That work is now precomputed offline into
      `data/recipes_vectors.pkl` (see §5a), so import is down to ~3.5s —
      comfortably under the default 30s. If free-tier CPU still runs slower
      than expected in practice, `--timeout 60` is a cheap safety margin to
      add back, but it shouldn't be needed.
- [x] Choose the **Free** instance type to start.
- [x] Deploy, then verify the live URL loads `/` and that submitting a user
      ID on `/process` returns recommendations (watch for the cold-start
      delay on the first request).
- [x] Once confirmed working, link the live URL from `README.md`.
      Done — https://food-for-thought-b8ur.onrender.com/

### 5c. `/` and `/process` items to revisit once the frontend changes

Moved out of §3 — these were noted against the *current* `/` and `/process`
handlers, and may be moot or need re-evaluating once the real frontend
(above) replaces them.

- [x] `/process` does zero input validation: a non-numeric `user_id`, a
      `user_id` with no interactions, or a user with fewer than 10 candidate
      recommendations will throw an unhandled exception (the `result` loop
      hardcodes `range(10)` against a list that isn't guaranteed to have 10
      items) and return a raw Flask 500 page.
      Done — non-numeric/missing/out-of-range `user_id` returns a 400 with a
      JSON `{error}` body instead of a 500; the top-10 slice is now
      `personalRecommendations[:10]`, which degrades to fewer (or zero)
      items instead of raising `IndexError`.
- [x] `/` returns a hardcoded HTML string instead of `render_template` (a
      `templates/` folder is imported via `render_template` but doesn't
      exist) — becomes moot once there's a real frontend, but worth knowing
      the current handler is dead-code-adjacent.
      Done — `/` now renders `templates/index.html`.
- [x] `/process` returns JSON only, with recommendation scores and URLs
      baked into a single formatted string per item — fine for a JSON API
      consumed by a new frontend, but will need restructuring into a proper
      JSON schema (list of `{id, name, score, url}` objects) for anything to
      consume it cleanly.
      Done — now `{"recommendations": [{id, name, score, url, image}, ...]}`.
- [x] Consider whether `recipesV` (built once at import time over all 232k
      recipes) and the two large DataFrames should be lazily loaded / cached
      rather than loaded at import — matters more once this is deployed
      somewhere with limited memory (see hosting section above).
      Done for `recipesV` and the `url` column — see §5a. The two source
      DataFrames (`interactions`, `recipes`) still load fully at import from
      CSV; that's a smaller, separate cost (~1-2s) and wasn't part of this
      pass.

## 6. Nice-to-haves (once the above is solid)

- [x] A handful of automated tests around `parseReviews` /
      `generateRecommendations` with small synthetic data — mainly to catch
      the zero-division and missing-user edge cases from section 3.
      Done — `tests/test_recommender.py` (pytest), with synthetic
      DataFrames/sparse matrices so the tests don't depend on real user data.
      Covers: incremental-mean correctness in `parseReviews`, a user with no
      interactions returning zero vectors instead of crashing, recipes with
      zero ingredients/zero tags not producing NaN/inf in
      `generateRecommendations` (regression test for the §3 zero-division
      fix), already-rated recipes being excluded, results being ranked
      highest-first, and the 25-recommendation cap. `tests/conftest.py`
      imports the real `app.py` once per session (so it's exercised against
      the actual committed data files) and individual tests call the two
      functions directly with their own small synthetic inputs.
- [x] A minimal CI workflow (lint + tests) via GitHub Actions — also doubles
      as something recruiters glance at on the repo's Actions tab.
      Done — `.github/workflows/ci.yml` runs `ruff check .` then
      `pytest tests/ -v` on every push/PR to `main`. Added
      `requirements-dev.txt` (extends `requirements.txt` with `pytest` and
      `ruff`) so prod deploys don't pull test tooling. Along the way, fixed
      the pre-existing `ruff check .` failures in `app.py` (line length,
      an ambiguous `l` variable name) and `build_recipe_vectors.py` (line
      length) so the lint step is actually green rather than immediately
      broken by lint debt unrelated to this task; excluded `graphs/` and
      `preprocessing/` (exploratory notebooks) from ruff's scope in
      `ruff.toml` since they weren't part of this cleanup pass.

