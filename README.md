
# UKRI Funding Scout Agent

This is a prototype AI-style research funding agent.

It accepts a public academic profile URL or pasted profile text, extracts research themes, searches UKRI Funding Finder, ranks relevant opportunities, and produces a suggested bid angle.

## What it does

1. Reads a public profile page, for example a university profile, ORCID profile or personal webpage.
2. Extracts research themes and keywords.
3. Searches UKRI Funding Finder for open and upcoming opportunities.
4. Ranks opportunities using TF-IDF similarity and keyword overlap.
5. Produces a practical opportunity summary and possible bid angle.

## Important limitations

LinkedIn often blocks automated scraping. For LinkedIn, paste the profile text manually.

This is a prototype. Before applying, always check the official UKRI opportunity page for eligibility, deadlines, funding level and partner requirements.

## Run in Google Colab

Upload this project folder or unzip the project in Colab, then run:

```python
!pip install -r requirements.txt

import subprocess, time
from google.colab import output

server = subprocess.Popen([
    "uvicorn", "app:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

time.sleep(3)
print(output.eval_js("google.colab.kernel.proxyPort(8000)"))
```

Open the printed link.

## Deploy from GitHub to Render

GitHub Pages cannot run Python. For the full live agent, deploy this repository as a FastAPI web service.

1. Create a GitHub repository.
2. Upload all files in this folder.
3. Go to Render.
4. New Web Service.
5. Connect the GitHub repository.
6. Build command:

```bash
pip install -r requirements.txt
```

7. Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

8. Deploy.

Render will give you a public HTTPS URL.

## Optional GitHub Pages front end

The front end is in `docs/index.html`. You may publish it with GitHub Pages and enter your Render backend URL in the `Backend API URL` field.

GitHub Pages settings:

* Source: Deploy from a branch
* Branch: main
* Folder: /docs

## Suggested improvements

* Add OpenAI API on the backend for richer bid summaries.
* Save user searches.
* Add email alerts for new UKRI opportunities.
* Add UKRI council filters, for example ESRC, EPSRC, AHRC, MRC, NERC, BBSRC, Innovate UK.
* Add downloadable PDF or Word opportunity reports.
* Add collaborator recommendation from university staff profiles.
