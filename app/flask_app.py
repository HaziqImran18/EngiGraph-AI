"""Flask web app: upload manuals, describe a problem, view the cited report.

Long tasks (indexing, diagnosis) run in a background thread. The browser
polls /api/jobs/<id> to show live progress, so requests never time out.

Run from the project root:  python main.py serve
"""
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.ingestion import indexer
from src.services.report import SAFETY_NOTES
from src.workflow import graph as workflow_graph

load_dotenv()
log = logging.getLogger("engigraph")

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "raw_pdfs"
REPORT_DIR = ROOT / "reports"
MIN_QUERY, MAX_QUERY = 5, 1000

JOBS: dict[str, dict] = {}
DIAGNOSE_LOCK = threading.Lock()  # one diagnosis at a time (protects API rate limits)
INDEX_LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers
def friendly_error(exc: Exception) -> str:
    """Turn a technical exception into a message a user can act on."""
    low = f"{type(exc).__name__}: {exc}".lower()
    if "index is empty" in low:
        return "No documents are indexed yet. Upload a PDF first."
    if "missing" in low and "key" in low:
        return "An API key is missing. Check your .env file."
    if "malformed structured output" in low or "tool_use_failed" in low:
        return "The AI model returned a badly formatted answer. Please run the diagnosis again."
    if "429" in low or "rate limit" in low or "ratelimit" in low or "resource_exhausted" in low:
        return "The AI provider's rate limit was reached. Wait about a minute and try again."
    if "401" in low or "invalid api key" in low or "api key not valid" in low:
        return "The API key was rejected. Check GROQ_API_KEY / GOOGLE_API_KEY in .env."
    if "404" in low and "model" in low:
        return "Model not found. Check GROQ_MODEL / GOOGLE_EMBEDDING_MODEL in .env."
    if any(w in low for w in ("connection", "timeout", "timed out")):
        return "Could not reach the AI provider. Check your internet connection."
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Unexpected error ({type(exc).__name__}). See the server console for details."


def _new_job(kind: str) -> dict:
    if len(JOBS) > 50:  # keep memory bounded
        for old in sorted(JOBS, key=lambda j: JOBS[j]["created"])[:20]:
            JOBS.pop(old, None)
    job = {
        "id": uuid.uuid4().hex[:10], "kind": kind, "status": "running", "created": time.time(),
        "stage": "", "progress": 0.0, "steps": [], "active": None, "revisions": 0,
        "result": None, "error": None,
    }
    JOBS[job["id"]] = job
    return job


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "created"}


def _serialize(state: dict, report_file: str) -> dict:
    findings, plan = state.get("findings", []), state.get("diagnostic_plan")
    verification = state.get("verification_result")
    if not findings:
        status = "no_evidence"
    elif verification and verification.verified:
        status = "verified"
    else:
        status = "partial"
    return {
        "query": state["user_query"],
        "status": status,
        "findings": [f.model_dump() for f in findings],
        "plan": plan.model_dump() if plan else {"steps": [], "safety_notes": []},
        "verification": verification.model_dump() if verification else None,
        "passages": [
            {"source": c.source, "page": c.page, "score": c.score, "text": c.text[:600]}
            for c in state.get("retrieved_chunks", [])
        ],
        "safety_notes": SAFETY_NOTES,
        "report_markdown": state["final_report"],
        "report_file": report_file,
    }


# ---------------------------------------------------------------- background tasks
def _diagnosis_task(job: dict, query: str) -> None:
    try:
        def on_step(node: str, delta: dict) -> None:
            job["steps"].append(node)
            job["revisions"] = max(0, job["steps"].count("planner") - 1)
            if node == "investigator":
                job["active"] = "planner"
            elif node == "planner":
                job["active"] = "verifier"
            elif node == "verifier":
                v = delta["verification_result"]
                job["active"] = "planner" if (not v.verified and v.attempts < workflow_graph.MAX_VERIFICATIONS) else "report"
            else:
                job["active"] = None

        job["active"] = "investigator"
        state = workflow_graph.run_workflow(query, on_step=on_step)

        REPORT_DIR.mkdir(exist_ok=True)
        report_file = f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        (REPORT_DIR / report_file).write_text(state["final_report"], encoding="utf-8")

        job["result"] = _serialize(state, report_file)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        log.exception("Diagnosis failed")
        job["status"], job["error"] = "error", friendly_error(exc)
    finally:
        job["active"] = None
        DIAGNOSE_LOCK.release()


def _index_task(job: dict, names: list[str]) -> None:
    try:
        for i, name in enumerate(names):
            job["stage"] = f"Indexing {name} ({i + 1}/{len(names)})"

            def on_progress(done: int, total: int, i: int = i) -> None:
                job["progress"] = (i + done / total) / len(names)

            indexer.delete_document(name)  # replace any older version of the same file
            indexer.index_pdf(PDF_DIR / name, on_progress=on_progress)
        job["progress"], job["status"] = 1.0, "done"
    except Exception as exc:  # noqa: BLE001
        log.exception("Indexing failed")
        job["status"], job["error"] = "error", friendly_error(exc)
    finally:
        INDEX_LOCK.release()


# ---------------------------------------------------------------- app factory
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB per request

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify(error="File too large (limit 60 MB)."), 413

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        try:
            chunks = indexer.count_chunks()
        except Exception:  # noqa: BLE001  (e.g. embedding key missing)
            chunks = 0
        configured = lambda name: bool(os.getenv(name)) and not os.getenv(name, "").startswith("your_")  # noqa: E731
        return jsonify(
            llm_ready=configured("GROQ_API_KEY"),
            embeddings_ready=configured("GOOGLE_API_KEY"),
            chunks=chunks,
            model=os.getenv("GROQ_MODEL", "default"),
        )

    @app.get("/api/documents")
    def documents():
        try:
            indexed = indexer.list_documents()
        except Exception:  # noqa: BLE001
            indexed = {}
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        docs = [
            {"name": p.name, "size_kb": round(p.stat().st_size / 1024), "chunks": indexed.get(p.name, 0)}
            for p in sorted(PDF_DIR.glob("*.pdf"))
        ]
        return jsonify(documents=docs)

    @app.delete("/api/documents/<name>")
    def delete_document(name: str):
        name = secure_filename(name)
        path = PDF_DIR / name
        if not path.exists():
            return jsonify(error="Document not found."), 404
        try:
            indexer.delete_document(name)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=friendly_error(exc)), 500
        path.unlink()
        return jsonify(deleted=name)

    @app.post("/api/upload")
    def upload():
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            return jsonify(error="No file received."), 400

        for f in files:  # validate everything before saving anything
            if not secure_filename(f.filename).lower().endswith(".pdf"):
                return jsonify(error=f"{f.filename}: only PDF files are accepted."), 400
            head = f.stream.read(5)
            f.stream.seek(0)
            if head != b"%PDF-":
                return jsonify(error=f"{f.filename} is not a valid PDF."), 400

        if not INDEX_LOCK.acquire(blocking=False):
            return jsonify(error="Another indexing job is still running. Please wait."), 409

        PDF_DIR.mkdir(parents=True, exist_ok=True)
        names = []
        for f in files:
            name = secure_filename(f.filename)
            f.save(PDF_DIR / name)
            names.append(name)

        job = _new_job("index")
        threading.Thread(target=_index_task, args=(job, names), daemon=True).start()
        return jsonify(job_id=job["id"])

    @app.post("/api/diagnose")
    def diagnose():
        query = ((request.get_json(silent=True) or {}).get("query") or "").strip()
        if len(query) < MIN_QUERY:
            return jsonify(error=f"Describe the problem in at least {MIN_QUERY} characters."), 400
        if len(query) > MAX_QUERY:
            return jsonify(error=f"Problem description is too long (max {MAX_QUERY} characters)."), 400
        if not DIAGNOSE_LOCK.acquire(blocking=False):
            return jsonify(error="A diagnosis is already running. Please wait for it to finish."), 409

        job = _new_job("diagnose")
        threading.Thread(target=_diagnosis_task, args=(job, query), daemon=True).start()
        return jsonify(job_id=job["id"])

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            return jsonify(error="Unknown job."), 404
        return jsonify(_public(job))

    @app.get("/reports/<path:filename>")
    def download_report(filename: str):
        return send_from_directory(REPORT_DIR, filename, as_attachment=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False, threaded=True)