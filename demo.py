"""Run the supplied model, standalone clustering, or their ordinary integration."""

import argparse
import json
import math
from pathlib import Path

from comparison_model import IndependentComparisonModel
from clustering import mutual_best_partition


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def train_case(name):
    case = load("training.json")[name]
    model = IndependentComparisonModel.fit(case["rows"], case["dimensions"], alpha=2.0)
    return case, model


def model_demo(name):
    case, model = train_case(name)
    dimensions = case["dimensions"]
    values = [1] * dimensions
    return {"case": name, "probabilities": model.probabilities,
            "score": model.score(values, prior=0.5),
            "all_missing_score": model.score([None] * dimensions, prior=0.3)}


def cluster_demo(name):
    case = load("clustering.json")[name]
    weights = {(a, b): weight for a, b, weight in case["edges"]}
    groups = mutual_best_partition(case["nodes"], weights, case["duplicate_free_sources"])
    return {"case": name, "partition": groups}


def linkage_demo():
    case = load("linkage.json")["ordinary"]
    _, model = train_case(case["training_case"])
    weights, evidence = {}, []
    for pair in case["pairs"]:
        score = model.score(pair["values"], case["prior"])
        weight = math.log(score["probability"] / (1.0 - score["probability"]))
        weights[(pair["left"], pair["right"])] = weight
        evidence.append({**pair, **score, "log_odds": weight})
    groups = mutual_best_partition(case["nodes"], weights, case["duplicate_free_sources"])
    return {"model_probabilities": model.probabilities, "prior": case["prior"],
            "evidence": evidence, "duplicate_free_sources": case["duplicate_free_sources"],
            "partition": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("model", "cluster", "link"), nargs="?", default="link")
    parser.add_argument("--case", default=None)
    args = parser.parse_args()
    if args.mode == "model":
        result = model_demo(args.case or "ordinary")
    elif args.mode == "cluster":
        result = cluster_demo(args.case or "three_sources_positive")
    else:
        result = linkage_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
