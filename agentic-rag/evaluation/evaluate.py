"""RAGAS evaluation runner for the RAG API pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from api.router import query_book
from api.models import QueryRequest

logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    """Load golden QA dataset from JSON file."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("Failed to load golden dataset: %s", exc)
        raise


def build_eval_rows(samples: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Run pipeline on sample questions and prepare RAGAS dataset rows."""
    try:
        questions, answers, contexts, ground_truths = [], [], [], []
        for sample in samples:
            req = QueryRequest(query=sample["question"])
            response = query_book(req)
            questions.append(sample["question"])
            answers.append(response.answer)
            contexts.append([c.excerpt for c in response.citations] or [""])
            ground_truths.append(sample["ground_truth"])
        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    except Exception as exc:
        logger.exception("Failed to build evaluation rows: %s", exc)
        raise


def run_evaluation(dataset_path: str = "evaluation/golden_dataset.json") -> Any:
    """Execute RAGAS metrics on the golden dataset."""
    try:
        samples = load_golden_dataset(dataset_path)
        rows = build_eval_rows(samples)
        result = evaluate(
            rows,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        logger.info("RAGAS evaluation result: %s", result)
        return result
    except Exception as exc:
        logger.exception("RAGAS evaluation failed: %s", exc)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    run_evaluation()
