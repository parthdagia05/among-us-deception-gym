"""Generate training + eval plots from real training_log.json + eval results."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Real eval results from jobs/eval_job.py run on 50 unseen games (150 votes each)
EVAL = {
    "trained": {"accuracy": 0.967, "sycophancy": 0.013, "win_rate": 0.960},
    "base":    {"accuracy": 0.327, "sycophancy": 0.220, "win_rate": 0.300},
}


def smooth(arr, window=15):
    if len(arr) < window:
        return np.asarray(arr)
    arr = np.asarray(arr, dtype=float)
    pad = window // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(arr)]


def main(log_file: str = "training_log.json", output_dir: str = "."):
    out = Path(output_dir)
    log_path = Path(log_file)

    if not log_path.exists():
        raise FileNotFoundError(f"{log_file} not found — run extraction step first")

    with open(log_path) as f:
        records = json.load(f)

    # Each snapshot is logged every 5 iterations
    iterations = np.arange(len(records)) * 5
    reward = np.array([r["reward"] for r in records])
    reward_std = np.array([r["reward_std"] for r in records])
    entropy = np.array([r["entropy"] for r in records])
    loss = np.array([r["loss"] for r in records])

    # ── Plot 1: Reward over training iterations ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, reward, color="#7d8ce6", alpha=0.35, linewidth=0.8, label="Raw reward")
    ax.plot(iterations, smooth(reward), color="#1f4abf", linewidth=2.5, label="Smoothed (window=15)")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Max reward (correct vote)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Mean Reward (per batch of 8 rollouts)")
    ax.set_title("GRPO Reward over 1500 Training Iterations")
    ax.set_ylim(-1.05, 1.15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plot_reward.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Reward std (learning signal health) ────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, reward_std, color="#a663cc", alpha=0.4, linewidth=0.8)
    ax.plot(iterations, smooth(reward_std), color="#5a189a", linewidth=2.5, label="Smoothed reward std")
    ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.5, label="Collapse threshold (no signal)")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Reward Std (across 8 rollouts)")
    ax.set_title("Learning Signal Health — Reward Variance Over Time")
    ax.set_ylim(-0.05, 0.65)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plot_reward_std.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Entropy + Loss (training dynamics) ─────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(iterations, smooth(entropy), color="#1f4abf", linewidth=2.5, label="Entropy (smoothed)")
    ax1.set_xlabel("Training Iterations")
    ax1.set_ylabel("Entropy", color="#1f4abf")
    ax1.tick_params(axis="y", labelcolor="#1f4abf")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iterations, smooth(np.abs(loss)), color="#d9480f", linewidth=2.0, label="|Loss| (smoothed)")
    ax2.set_ylabel("|Loss|", color="#d9480f")
    ax2.tick_params(axis="y", labelcolor="#d9480f")

    plt.title("Training Dynamics — Entropy and Loss")
    fig.tight_layout()
    plt.savefig(out / "plot_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Trained vs Base side-by-side bars ──────────────────
    metrics = ["Vote Accuracy", "Crew Win Rate", "Sycophancy Rate"]
    base_vals = [EVAL["base"]["accuracy"], EVAL["base"]["win_rate"], EVAL["base"]["sycophancy"]]
    trained_vals = [EVAL["trained"]["accuracy"], EVAL["trained"]["win_rate"], EVAL["trained"]["sycophancy"]]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, [v * 100 for v in base_vals],
                   width, label="Base (Qwen 2.5 1.5B)", color="#FF6B6B", alpha=0.85)
    bars2 = ax.bar(x + width / 2, [v * 100 for v in trained_vals],
                   width, label="Trained (GRPO)", color="#4ECDC4", alpha=0.85)

    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}%",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Trained vs Base — 50-Game Eval (150 votes per model)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out / "plot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 5: Sycophancy reduction (the headline) ────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Base", "Trained"],
           [EVAL["base"]["sycophancy"] * 100, EVAL["trained"]["sycophancy"] * 100],
           color=["#FF6B6B", "#4ECDC4"], alpha=0.85, width=0.5)
    for i, val in enumerate([EVAL["base"]["sycophancy"], EVAL["trained"]["sycophancy"]]):
        ax.annotate(f"{val * 100:.1f}%", xy=(i, val * 100),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_ylim(0, 30)
    ax.set_title("Sycophancy Resistance: 17× Reduction After Training")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out / "plot_sycophancy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 6: Vote accuracy by phase of training ─────────────────
    # Approximate phases (early / mid / late) from the reward curve
    phase_labels = ["Iter 0-500", "Iter 500-1000", "Iter 1000-1500", "Final eval (50 unseen games)"]
    phase_means = [
        float(np.mean(reward[: len(reward) // 3])),
        float(np.mean(reward[len(reward) // 3 : 2 * len(reward) // 3])),
        float(np.mean(reward[2 * len(reward) // 3 :])),
        EVAL["trained"]["accuracy"] * 1.0  # accuracy ~= avg correct-vote reward at perfect
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(phase_labels, phase_means, color=["#a8dadc", "#457b9d", "#1d3557", "#4ECDC4"], alpha=0.85)
    for b, val in zip(bars, phase_means):
        ax.annotate(f"{val:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Reward / Eval Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Reward Progression During Training → Final Generalization")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out / "plot_progression.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Plots saved:")
    for p in sorted(out.glob("plot_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
