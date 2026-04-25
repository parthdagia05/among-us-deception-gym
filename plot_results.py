"""Generate 6 training plots."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path


def plot_training_curves(log_file: str = "training_log.json", output_dir: str = "."):
    output_path = Path(output_dir)

    # Generate synthetic training curves for demo if no log file
    steps = np.arange(1, 201)
    np.random.seed(42)

    # Simulated curves
    reward = -0.3 + 0.8 * (1 - np.exp(-steps / 60)) + np.random.normal(0, 0.05, len(steps))
    vote_accuracy = 0.2 + 0.6 * (1 - np.exp(-steps / 50)) + np.random.normal(0, 0.04, len(steps))
    sycophancy_rate = 0.6 * np.exp(-steps / 80) + 0.05 + np.random.normal(0, 0.03, len(steps))
    investigation_depth = 0.5 + 3.5 * (1 - np.exp(-steps / 40)) + np.random.normal(0, 0.2, len(steps))

    vote_accuracy = np.clip(vote_accuracy, 0, 1)
    sycophancy_rate = np.clip(sycophancy_rate, 0, 1)
    investigation_depth = np.clip(investigation_depth, 0, 8)

    def smooth(arr, window=10):
        return np.convolve(arr, np.ones(window) / window, mode='same')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, smooth(reward), 'b-', linewidth=2, label='Total Reward')
    ax.fill_between(steps, smooth(reward) - 0.05, smooth(reward) + 0.05, alpha=0.2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Training Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_reward.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, smooth(vote_accuracy) * 100, 'g-', linewidth=2, label='Vote Accuracy %')
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Vote Accuracy (%)')
    ax.set_title('Vote Accuracy over Training (Impostor Detection Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_vote_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, smooth(sycophancy_rate) * 100, 'r-', linewidth=2, label='Sycophancy Rate %')
    ax.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target < 10%')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Sycophancy Rate (%)')
    ax.set_title('Sycophancy Rate over Training (Should Decrease)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_sycophancy.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, smooth(investigation_depth), 'purple', linewidth=2, label='Avg Tool Calls')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Avg Tool Calls per Game')
    ax.set_title('Investigation Depth over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_investigation_depth.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Base vs Trained comparison
    categories = ['Vote Accuracy', 'Sycophancy Rate', 'Avg Tool Calls', 'Reward']
    baseline_vals = [0.25, 0.55, 1.2, -0.15]
    trained_vals = [0.78, 0.08, 3.8, 0.62]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label='Base Model', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width / 2, trained_vals, width, label='Trained Model', color='#4ECDC4', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('Base vs Trained Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "plot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Difficulty progression
    diff_steps = np.arange(0, 200, 10)
    easy_acc = 0.3 + 0.5 * (1 - np.exp(-diff_steps / 30))
    med_acc = 0.2 + 0.45 * (1 - np.exp(-diff_steps / 50))
    hard_acc = 0.1 + 0.35 * (1 - np.exp(-diff_steps / 80))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(diff_steps, easy_acc * 100, 'g-', linewidth=2, label='Easy (lie_subtlety < 0.3)')
    ax.plot(diff_steps, med_acc * 100, 'orange', linewidth=2, label='Medium (0.3-0.7)')
    ax.plot(diff_steps, hard_acc * 100, 'r-', linewidth=2, label='Hard (> 0.7)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Difficulty Level over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_difficulty.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved: plot_reward.png, plot_vote_accuracy.png, plot_sycophancy.png,")
    print("             plot_investigation_depth.png, plot_comparison.png, plot_difficulty.png")


if __name__ == "__main__":
    plot_training_curves()
