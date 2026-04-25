"""Evaluate base vs trained model on 100 games."""
import json
import requests
import argparse
from typing import List, Dict

ENV_URL = "http://localhost:8000"


def run_random_agent(n_games: int = 100) -> Dict:
    """Simulate a random/baseline agent that always votes for the first player."""
    results = {"correct": 0, "total": n_games, "sycophancy": 0, "tool_calls": []}

    for i in range(n_games):
        try:
            resp = requests.post(f"{ENV_URL}/reset")
            obs = resp.json()

            # Baseline: call read_statements then vote for first player
            requests.post(f"{ENV_URL}/step", json={
                "action_type": "tool_call",
                "tool_name": "read_statements",
                "tool_args": {}
            })

            alive = obs.get("alive_players", [])
            if not alive:
                continue

            # Vote for whoever seems "confident" (first in list for baseline)
            target = alive[0]

            step_resp = requests.post(f"{ENV_URL}/step", json={
                "action_type": "vote",
                "vote_target": target
            })
            result = step_resp.json()

            if result.get("info", {}).get("correct"):
                results["correct"] += 1

            flags = result.get("info", {}).get("flags", {})
            if flags.get("sycophancy_cave") or flags.get("sycophancy_detected"):
                results["sycophancy"] += 1

            results["tool_calls"].append(1)
        except Exception as e:
            print(f"Game {i} error: {e}")

    results["accuracy"] = results["correct"] / max(1, results["total"])
    results["sycophancy_rate"] = results["sycophancy"] / max(1, results["total"])
    results["avg_tool_calls"] = sum(results["tool_calls"]) / max(1, len(results["tool_calls"]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=100)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    print(f"Running baseline evaluation on {args.n_games} games...")
    baseline = run_random_agent(args.n_games)
    print(f"Baseline accuracy: {baseline['accuracy']:.2%}")
    print(f"Baseline sycophancy rate: {baseline['sycophancy_rate']:.2%}")
    print(f"Baseline avg tool calls: {baseline['avg_tool_calls']:.1f}")

    results = {"baseline": baseline}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
