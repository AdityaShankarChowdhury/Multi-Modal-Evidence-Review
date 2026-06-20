# HackerRank Damage Claim Evidence Review System

This Python-based multi-modal damage verification system analyzes claim transcripts, images, user history, and minimum evidence checklists to verify damage claims.

## Prerequisites & Installation

1. Install Python 3.9+ or higher.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Set the `GEMINI_API_KEY` environment variable. You can do this by exporting it in your shell or creating a `.env` file inside the `code` folder (which is gitignored).

**Shell (Linux/macOS):**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**PowerShell (Windows):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Command Prompt (Windows):**
```cmd
set GEMINI_API_KEY="your_api_key_here"
```

**`.env` File:**
Create a file named `.env` in the `code/` folder:
```text
GEMINI_API_KEY=your_api_key_here
```

## Running the Evaluation

To evaluate prompt strategies, run the evaluation script:
```bash
python evaluation/main.py
```
This script runs predictions over `dataset/sample_claims.csv`, evaluates metrics (accuracies and exact-match rates) for both Strategies (Minimal vs. Detailed), selects the best strategy, and outputs the comparison results to `code/evaluation/evaluation_report.md`.

## Running the Pipeline

Once evaluation is completed, run the main prediction script:
```bash
python main.py
```
This script reads `dataset/claims.csv`, executes the pipeline on each claim using the selected prompt strategy, and writes the structured output to `output.csv` in the repository root.

It automatically verifies:
- Output row count matches input claims row count exactly (44 rows).
- Schema structure and ordering match requirements exactly.
- All enum-constrained values are valid.

## Pipeline Highlights

* **Multi-Modal Integration:** Combines verbatim claim transcripts, history context, image checklists, and high-fidelity image bytes into a single request.
* **Resilient Schema Enforcement:** Enforces strict Pydantic model validation on JSON output and performs corrective retry prompts if necessary.
* **Deterministic Configuration:** Uses a temperature of 0.0 to maximize predictability.
* **Disk Caching:** Caches SHA-256 hashes of inputs (claims, requirements, history, and image bytes) locally in `.cache/gemini_cache.json` to prevent duplicate API billing.
* **RPM Throttling & Retries:** Built-in sleep throttle (RPM configured) and exponential backoff with jitter handle API limitations gracefully.
