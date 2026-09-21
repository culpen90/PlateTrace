# PlateTrace

A local autonomous agent for public vehicle research. Give it a plate, its issuing jurisdiction, and an objective. An OpenRouter or Ollama model chooses its own searches, websites, shell commands, and scripts, observes the results, and continues until it produces a sourced report or reaches its run limit.

**This is an agent, not a plate database.** There is no fixed OSINT provider or source allowlist. The agent can explore public sources through general web tools and an internet-enabled Linux terminal. No application can guarantee a match for every plate. Plate reassignment, regional access restrictions, ambiguous records, and missing public data all affect results.

The supported scope is public vehicle specifications, recalls, and authorized vehicle records. Private owner identification, personal contact information, surveillance, location tracking, leaked data, and access-control bypass are outside the scope.

## Run locally

Requires Python 3.11 or newer. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
platetrace
```

Open [PlateTrace](http://127.0.0.1:8741). The server binds only to loopback. Use `platetrace --port 8742` if needed. Configuration is read from `.env` when using the `platetrace` or `python -m platetrace` entry point; restart after changing it.

### Choose a model

**OpenRouter:** set `OPENROUTER_API_KEY` in `.env` or enter a key for a single run in the UI. Refresh the model list to discover available models with tool support. Model calls use your account and may incur charges. Request-scoped keys are held in memory for that run; they are not written into case history or browser storage.

**Ollama:** install and start [Ollama](https://ollama.com/), then pull a model that supports tools, for example:

```sh
ollama pull qwen3:8b
ollama serve
```

Skip `ollama serve` if Ollama is already running. Refresh models in PlateTrace and select your installed model. The default API URL is `http://127.0.0.1:11434`; override `OLLAMA_BASE_URL` for your own server. The model's reliability and memory requirements vary; small models may fail to complete multi-step tool work. Local inference still sends research queries to websites when web tools are used.

**Demo:** a clearly marked synthetic fixture exercises the UI, event log, report, and exports without contacting an LLM, terminal, or website. It is not a real plate lookup.

### Enable automatic GitHub issue reports

The agent can report an actionable PlateTrace bug when it judges a report necessary. Reporting uses the server's GitHub API connection and works with terminal access disabled or Docker unavailable. Demo runs never publish issues.

To enable reporting, create a [fine-grained GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for the selected repository with **Issues: read and write** permission. Set these values in `.env`, then restart PlateTrace:

```dotenv
PLATETRACE_GITHUB_TOKEN=your-token
PLATETRACE_GITHUB_REPOSITORY=culpen90/PlateTrace
```

Set your own `owner/repository` explicitly when using a fork. The default tracker, `culpen90/PlateTrace`, is public; a custom tracker's reports are visible to people who have access to that repository. **Configuring the token authorizes automatic issue creation without per-run confirmation.** Remove the token and restart the server to disable reporting. The token stays on the server and is not provided to the model, browser, or terminal container. The UI shows reporting availability next to the terminal control.

Reports describe the defect, reproduction steps, and expected and actual behavior. The agent is instructed to report actionable application defects, not an expected lack of public vehicle matches, normal access controls, or routine setup problems. The server checks for earlier agent reports with the same normalized title, including closed issues, and permits at most three creation attempts per run. This does not detect every differently worded or manually filed duplicate. Created or existing issue links appear in the activity log, research brief, and exports. Reporting failures remain visible as tool errors so the agent can continue research.

Reports should contain only the minimum technical detail needed to reproduce a bug. Raw case logs are not attached automatically. The server applies best-effort redaction to known credentials, supplied vehicle identifiers, and common sensitive text patterns; this cannot guarantee removal of every sensitive detail from model-written text. Do not include private information in issue reports.

### Enable the autonomous terminal

Install and start a Linux-container Docker runtime, then prepare the default image:

```sh
docker pull python:3.12-alpine
```

Reload PlateTrace and enable terminal access. Each run receives a separate container with a shell, Python, BusyBox utilities such as `wget`, and Docker bridge internet access. The model may execute arbitrary shell commands in that container, create scripts, fetch websites, and install Python packages into `/work`. Working files persist between commands for the duration of the run. `evidence.json` is refreshed before each command.

This does **not** give the model access to your Mac's terminal, files, Docker socket, or provider credentials. Containers run as an unprivileged user with a read-only root filesystem, dropped capabilities, 256 MB RAM, one CPU, 64 processes, a 64 MB workspace, and a 16 MB temporary directory. Commands time out after 30 seconds; a timeout attempts to destroy that workspace. Output is capped at 16,000 characters per stream. Cancellation and normal completion attempt to remove the container; Docker daemon failures can prevent immediate cleanup and are logged. Containers also have a 20-minute lifetime. After a hard server/process crash, inspect containers labeled `app=platetrace` if recovery is needed.

Docker bridge networking permits outbound traffic, potentially including services reachable on your LAN or Docker host. It is **not** a public-internet-only network firewall. Use a separate VM/network if your environment needs strict egress isolation. The app's public-web fetch tool separately blocks private addresses and validates every redirect. Research scope and prompt-injection instructions guide the model; they are not a complete enforcement boundary for arbitrary networked shell code.

Set `PLATETRACE_TERMINAL_IMAGE` to a trusted, prebuilt image if you need extra tools such as `curl`, a PDF parser, or a browser. The image must have `python` and `/bin/sh`, work under UID 65534 with a read-only root, and fit the resource limits. The app never pulls images automatically. The default image does not include an interactive browser; JavaScript-heavy sites may need an appropriate custom image. Containers receive no host bind mounts. Put no credentials in custom images.

### Internet research

- `search_web`: model-chosen search queries. Uses Brave Search when `BRAVE_SEARCH_API_KEY` is configured; otherwise uses best-effort DuckDuckGo HTML search, which may block automation. A blocked search is reported as a tool error for the agent to handle.
- `fetch_url`: reads public HTTP(S) HTML, text, JSON, and XML, records source URLs and retrieval times, and returns page links for further exploration. Supports ports 80/443, up to five redirects, 1 MB responses, and a bounded text excerpt. Unsupported formats can be explored with terminal utilities.
- `terminal`: arbitrary shell in the run's internet-enabled temporary container. The model chooses commands; there is no fixed command allowlist or database pipeline.
- `read_records`: optional exact plate/jurisdiction lookup in the vehicle records you supply.
- `report_issue`: files a necessary PlateTrace bug report in the configured GitHub tracker or returns an existing report. Available when server-side GitHub reporting is configured, independently of terminal access.
- `finish_report`: validates report structure and checks that every finding references collected source IDs. It does not prove that a source supports the model's interpretation; review important claims against originals.

The agent can use any public site within the research scope. It is instructed to respect access controls and treat external text as untrusted data. Sites requiring authentication, payment, CAPTCHAs, or special access can remain unavailable. Network errors and inconclusive research are explicit; no fictitious lookup result is substituted.

## Research workflow

1. Enter a plate and issuing country/state/province, or upload a plate photo and review the detected text. The same text may occur in multiple jurisdictions.
2. Describe the public vehicle question. Optionally add a VIN, make/model/year, or starting URLs you already have.
3. Select OpenRouter or Ollama, a tool-capable model, and the run's turn limit. Enable the terminal if Docker is ready. Leave **Use past research memory** on to recall and save research lessons, or turn it off for this run.
4. Choose the research purpose that describes your situation (see below), then confirm that you are authorized to conduct the research and use any supplied records.
5. Start the agent. Inspect its actual tool requests/results in the live activity log. Stop it at any time.
6. Review findings, citations, and limitations. Export JSON (including events and sources) or a Markdown report.

A supplied VIN or record is labeled user-provided, not independently verified. Model/year recalls alone do not establish whether a particular VIN is affected or repaired. No result does not establish that a vehicle does not exist or has no recalls.

### Research memory

**Use past research memory** is enabled by default for live runs. The agent keeps compact lessons from earlier runs, including useful source leads, observed tool successes and failures, and optional research lessons supplied when the model finishes its report. Later runs recall up to five relevant entries from the past 90 days, matched by jurisdiction, vehicle clues, or research objective. This gives the model context for choosing its next steps; it does not train or change the model, and creating memory requires no extra model calls.

Recalled lessons and source leads are sent to your selected model provider with the new run. They are untrusted hints, not verified facts about the current vehicle. The agent must collect fresh evidence before using a remembered source in a finding. Demo runs never read or save memory. Turning the checkbox off skips both recall and saving for that run.

The **Research memory** panel shows the saved entry count and lets you inspect lessons, source leads, and tool outcomes. **Clear memory** removes all saved entries without deleting case history. Older case logs are not automatically imported again, and runs already in progress when you clear memory cannot add it back when they finish. Clearing cannot remove context already sent to a provider or erase a run's audit trail. Memory events in the activity log and the JSON export show recall and save activity.

On the first startup with no memory file, PlateTrace imports observed tool outcomes and source website origins from eligible finished live runs in the last 90 days. This one-time import excludes demo runs and legacy report prose or model-written lessons. Clearing memory leaves an empty memory file, so restarting does not reimport the old case history.

Memory is stored locally in `.platetrace/runs/memory/entries.json` and retains at most 200 entries within a 2 MB storage limit across restarts. Entries omit raw plates, VINs, supplied records, terminal logs, credentials, and report findings; source leads are sanitized. Model-written lessons receive best-effort redaction, which cannot guarantee removal of every sensitive detail. Inspect and clear saved entries as needed. Full case history has its own retention policy below.

### Start from a plate photo

Choose a JPEG, PNG, or WebP photo (up to 8 MiB) in the photo section beneath the plate fields. A preview appears before anything is sent to a model. Select OpenRouter or Ollama as the research engine, then choose **Read plate from photo**. The selected model must support images. You can enter a separate **Photo model** to read the image while keeping your tool-capable research model for the investigation. Demo mode does not read real photos.

Review the detected text, choose **Use detected plate**, and correct the plate or issuing jurisdiction as needed before starting research. If the jurisdiction cannot be read, enter it yourself. For a blurry photo or multiple plates, use a clearer photo cropped to one plate. Reading a photo does not start a research run.

OpenRouter sends the photo to the selected cloud model service and may incur charges. Ollama sends it to your configured Ollama server. Photos are validated, oriented, resized when needed, and stripped of metadata before model processing; images over 24 megapixels and animated images are rejected. PlateTrace does not save uploaded photos in case history or exports. Only the text you apply and submit becomes part of the research input. Your provider's own data policies still apply.

### Research purpose

Choose the option that best explains why you are researching the vehicle:

| Purpose | When to choose it | Example |
| --- | --- | --- |
| **Public vehicle research** | You want publicly available vehicle facts without relying on ownership or fleet-management authority. This is the default for general specifications and recall research. | Research published specifications and model-year recall notices for a vehicle you are considering buying. |
| **My own vehicle** | You are researching a vehicle you own and can supply details from your own records, such as its VIN, make, model, or year. | Check public recall information using the VIN from your registration. |
| **Authorized fleet research** | You have permission to research vehicles managed by an organization. Supply the relevant vehicle details or authorized records; each run still targets one plate and jurisdiction. | Research specifications and recall notices for a company van you are responsible for maintaining. |
| **Authorized vehicle dataset** | Your starting point is a set of vehicle records you have permission to use. Paste the JSON array into **Authorized vehicle records** so the agent can look for an exact plate/jurisdiction match and investigate related public facts. | Compare a permitted inventory record with publicly available vehicle specifications. |

These choices describe the research context. The selected purpose is sent to the model and saved with the run; the model may use it to tailor its research. All four choices use the same agent and available tools. A purpose does not switch databases, enable terminal access, verify ownership or permission, or grant access to restricted records. Provider selection and the terminal toggle are separate controls, and the same vehicle-only scope applies to every purpose.

If more than one purpose fits, choose the one that best describes the task. You can supply authorized vehicle records with any purpose. Choosing **Authorized vehicle dataset** does not load records automatically or restrict the agent to that dataset; it can still use the available web and terminal tools.

### Supplying vehicle records

Optional dataset input is a JSON array of vehicle-only records:

```json
[
  {
    "plate": "DEMO123",
    "jurisdiction": "Example jurisdiction",
    "make": "Example Motors",
    "model": "Touring",
    "year": 2020,
    "fuel": "Electric",
    "color": "Blue"
  }
]
```

Only `plate`, `jurisdiction`, `vin`, `make`, `model`, `year`, `fuel`, and `color` fields are accepted. The example is fictional. Do not supply personal information. Web sources and terminal output are sent to your chosen model provider; using OpenRouter sends research context to the selected cloud service.

## Local data and limits

Case history is stored under `.platetrace/runs/` with owner-only file permissions, including input, visible tool logs, source excerpts, and reports. API keys and hidden model reasoning are excluded. Up to 100 recent cases are kept; the oldest inactive case is removed when a new case exceeds that limit. The UI does not persist credentials. Stop the app and remove its data directory to erase history. The default is local single-user use; do not expose the server publicly.

Each run is limited to 2–40 model turns (default 24), eight tool calls per turn, and 15 minutes. The final two turns are reserved within that total to write a sourced report and correct it if needed; a two-turn run reserves one turn for research and one for its report. The agent may finish earlier. More turns allow deeper research and may increase provider costs. At most two runs are active at once. Provider calls have deadlines. Stopping a run cancels active work and attempts container cleanup; a cloud provider may still bill a request already submitted. Server restart marks unfinished runs interrupted; it does not silently restart billable work.

## Development and verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check platetrace tests
node --check platetrace/static/app.js
```

Tests mock provider, GitHub, and Docker responses to cover tool conversations, validation, cancellation, terminal isolation flags, issue reporting, source handling, research memory, and API lifecycle without sending plate queries, publishing issues, or spending credits. A passing mock test is not evidence of a successful live model investigation, authenticated GitHub submission, or live Docker session. Use your configured services to verify that environment separately.

Implementation references: [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling), [Ollama chat API](https://docs.ollama.com/api/chat), [GitHub issue creation](https://docs.github.com/en/rest/issues/issues#create-an-issue), [Docker container run](https://docs.docker.com/engine/containers/run/), and [Brave Search API](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).
