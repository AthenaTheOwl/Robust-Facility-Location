# Deploy — Streamlit Community Cloud

The app is in deployable shape. To put it on a public URL:

1. Sign in at https://share.streamlit.io with the GitHub account that owns `AthenaTheOwl/facility-location` (one-time browser auth).
2. Click **New app** → **From existing repo**.
3. Fill in:
   - **Repository**: `AthenaTheOwl/facility-location`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Advanced settings → Python version**: `3.11`
   - **Secrets**: leave empty — the app has no API-key dependencies.
4. Click **Deploy**.

First build takes ~3 minutes (installs `cvxpy`, `pulp`, `plotly`,
`scipy`). Subsequent deploys take ~30 seconds via Streamlit Cloud's
incremental cache.

## Local dev

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

Interactive Robust Facility Location Explorer:
- Choose a region (regions defined in `core/`).
- Pick uncertainty assumptions (demand bounds, robustness budget).
- See nominal vs robust facility placements side by side.
- Compare expected and worst-case cost.

The optimizer uses CVXPY (LP/MILP) with PuLP as the backend. No
external services required.

## Notes for future maintainers

- `requirements.txt` is at repo root next to `app.py` — Streamlit Cloud's expected location.
- No `.streamlit/config.toml` needed; the defaults work.
- If you ever add a secret, put it in **Advanced settings → Secrets** in
  the dashboard, not in the repo.
