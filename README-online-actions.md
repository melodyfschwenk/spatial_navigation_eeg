# Alternative: Deploy via GitHub Actions (if "Deploy from a branch" isn't available)

If you don't see the "Deploy from a branch" option under Settings > Pages, you can deploy the site using GitHub Actions. This repository includes a workflow that publishes the `docs/` folder to GitHub Pages automatically.

What it does
- On every push to `main`, it uploads the `docs/` folder as a Pages artifact and deploys it.
- You can also trigger it manually via the Actions tab (Run workflow).

How to use it
1. Ensure GitHub Actions are enabled for this repository (Settings > Actions > General > Allow GitHub Actions to run).
2. Push/merge to `main` (or use Actions > Deploy GitHub Pages (docs/) > Run workflow).
3. Wait for the workflow to complete (~1–3 minutes). The site URL will appear in the job summary and under Settings > Pages.

Expected site URL
- https://melodyfschwenk.github.io/spatial_navigation_eeg/

Notes
- The site serves the static files in `docs/` (including `index.html`).
- This method works in both public and private repositories and avoids needing the "Deploy from a branch" UI.
- Data collection still depends on `docs/config.js` having your Google Apps Script Web App URL set as `DATA_ENDPOINT`.