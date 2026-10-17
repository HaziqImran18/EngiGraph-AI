"""Command line entry point for testing each piece.

Usage:
    python main.py list-models
    python main.py test-llm
    python main.py extract-pdf [file.pdf]
    python main.py index [file.pdf] [--reset]
    python main.py search "your question" [-k 5]
    python main.py ask "your question" [-k 5]
    python main.py diagnose "equipment problem" [--save]
    python main.py serve [--port 5000]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

PDF_FOLDER = Path("data/raw_pdfs")


def cmd_list_models(_args) -> None:
    from src.services.llm import list_available_models

    for model_id in list_available_models():
        print(model_id)


def cmd_test_llm(_args) -> None:
    from src.services.llm import ask_llm

    question = "In one sentence, what is a bearing in a rotating machine?"
    print(f"Q: {question}")
    print(f"A: {ask_llm(question)}")


def cmd_extract_pdf(args) -> None:
    from src.ingestion.pdf_loader import load_all_pdfs, load_pdf

    pages = load_pdf(PDF_FOLDER / args.file) if args.file else load_all_pdfs(PDF_FOLDER)
    if not pages:
        print("No text extracted. Put PDFs in data/raw_pdfs/ first.")
        return
    for doc, count in Counter(p.source for p in pages).items():
        print(f"{doc}: {count} pages with text")
    sample = pages[min(2, len(pages) - 1)]
    print(f"\nSample -> {sample.source}, page {sample.page}")
    print(sample.text[:500])


def cmd_index(args) -> None:
    from src.ingestion.indexer import count_chunks, index_pdf, reset_index

    if args.reset:
        reset_index()
        print("Index cleared.")
    files = [PDF_FOLDER / args.file] if args.file else sorted(PDF_FOLDER.glob("*.pdf"))
    if not files:
        print("No PDFs found in data/raw_pdfs/")
        return
    for pdf in files:
        index_pdf(pdf)
    print(f"\nTotal chunks in database: {count_chunks()}")


def cmd_search(args) -> None:
    from src.retrieval.search import search

    for i, hit in enumerate(search(args.query, k=args.k), start=1):
        print(f"\n[{i}] {hit.source} | p.{hit.page} | score {hit.score}")
        print(hit.text[:300].replace("\n", " "))


def cmd_ask(args) -> None:
    from src.services.basic_rag import answer_question

    result = answer_question(args.query, k=args.k)
    print(f"\nQ: {result.question}\n\nA: {result.answer}\n\nSources retrieved:")
    for hit in result.sources:
        print(f"  - {hit.source}, p.{hit.page} (score {hit.score})")


def cmd_diagnose(args) -> None:
    from datetime import datetime

    from src.workflow.graph import run_workflow

    state = run_workflow(args.query, verbose=True)
    report = state["final_report"]
    print("\n" + report)
    if args.save:
        Path("reports").mkdir(exist_ok=True)
        path = Path("reports") / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        path.write_text(report, encoding="utf-8")
        print(f"\nSaved: {path}")


def cmd_serve(args) -> None:
    from app.flask_app import create_app

    print(f"EngiGraph running at http://127.0.0.1:{args.port}")
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # PDFs contain symbols Windows may choke on

    parser = argparse.ArgumentParser(description="EngiGraph command line")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models").set_defaults(func=cmd_list_models)
    sub.add_parser("test-llm").set_defaults(func=cmd_test_llm)

    p = sub.add_parser("extract-pdf")
    p.add_argument("file", nargs="?")
    p.set_defaults(func=cmd_extract_pdf)

    p = sub.add_parser("index")
    p.add_argument("file", nargs="?", help="PDF name inside data/raw_pdfs (default: all)")
    p.add_argument("--reset", action="store_true", help="delete the index first")
    p.set_defaults(func=cmd_index)

    for name, func in (("search", cmd_search), ("ask", cmd_ask)):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=5, help="number of chunks to retrieve")
        p.set_defaults(func=func)

    p = sub.add_parser("diagnose")
    p.add_argument("query")
    p.add_argument("--save", action="store_true", help="save report to reports/")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=5000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()