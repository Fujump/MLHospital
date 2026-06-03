import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlhospital-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlhospital-xdg-cache")

import matplotlib.pyplot as plt


ACCURACY_COLOR = "#dc9b78"
AUC_COLOR = "#638b78"
LINE_WIDTH = 3
MARKER_SIZE = 7
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_LABEL_SIZE = 18


def plot_seed_metrics(output_dir: Path) -> None:
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    auc = [0.533, 0.533, 0.533, 0.530, 0.526, 0.530, 0.535, 0.531, 0.537]
    test_accuracy = [0.6840, 0.6831, 0.6718, 0.6708, 0.6929, 0.6822, 0.6699, 0.6823, 0.6788]

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_acc = plt.subplots(figsize=(8, 6))
    ax_auc = ax_acc.twinx()

    acc_line = ax_acc.plot(
        seeds,
        test_accuracy,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=ACCURACY_COLOR,
        label="Test accuracy",
    )
    auc_line = ax_auc.plot(
        seeds,
        auc,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=AUC_COLOR,
        label="Attack AUC",
    )

    ax_acc.set_xlabel("Seed")
    ax_acc.set_ylabel("Test Accuracy", color=ACCURACY_COLOR)
    ax_auc.set_ylabel("Attack AUC", color=AUC_COLOR)
    ax_acc.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_acc.xaxis.label.set_weight("bold")
    ax_acc.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_acc.yaxis.label.set_weight("bold")
    ax_auc.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_auc.yaxis.label.set_weight("bold")
    ax_acc.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, width=2.5)
    ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=2.5, labelcolor=ACCURACY_COLOR)
    ax_auc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=2.5, labelcolor=AUC_COLOR)
    for tick_label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels() + ax_auc.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax_acc.set_xticks(seeds)
    ax_acc.set_ylim(0.2, 0.8)
    ax_auc.set_ylim(0.2, 0.8)
    ax_acc.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)

    lines = acc_line + auc_line
    labels = [line.get_label() for line in lines]
    ax_acc.legend(lines, labels, loc="upper left", frameon=False, prop={"size": LEGEND_LABEL_SIZE, "weight": "bold"})

    # fig.suptitle("Test Accuracy and Attack AUC under Different Seeds", fontsize=16, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_dir / "seed_accuracy_auc.pdf")
    plt.close(fig)


def plot_sample_size_metrics(output_dir: Path) -> None:
    sample_sizes = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    sample_sizes_k = [sample_size / 1000 for sample_size in sample_sizes]
    sample_size_labels = ["6K", "6.5K", "7K", "7.5K", "8K", "8.5K", "9K", "9.5K", "10K"]
    sample_size_test_accuracy = [0.6960, 0.6934, 0.6978, 0.6889, 0.6902, 0.6937, 0.6921, 0.6864, 0.6738]
    sample_size_auc = [0.565, 0.556, 0.555, 0.549, 0.548, 0.546, 0.543, 0.537, 0.535]

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_acc = plt.subplots(figsize=(8, 6))
    ax_auc = ax_acc.twinx()

    acc_line = ax_acc.plot(
        sample_sizes_k,
        sample_size_test_accuracy,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=ACCURACY_COLOR,
        label="Test accuracy",
    )
    auc_line = ax_auc.plot(
        sample_sizes_k,
        sample_size_auc,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=AUC_COLOR,
        label="Attack AUC",
    )

    ax_acc.set_xlabel("Sample Size")
    ax_acc.set_ylabel("Test Accuracy", color=ACCURACY_COLOR)
    ax_auc.set_ylabel("Attack AUC", color=AUC_COLOR)
    ax_acc.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_acc.xaxis.label.set_weight("bold")
    ax_acc.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_acc.yaxis.label.set_weight("bold")
    ax_auc.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax_auc.yaxis.label.set_weight("bold")
    ax_acc.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, width=2.5)
    ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=2.5, labelcolor=ACCURACY_COLOR)
    ax_auc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=2.5, labelcolor=AUC_COLOR)
    ax_acc.set_xticks(sample_sizes_k)
    ax_acc.set_xticklabels(sample_size_labels)
    ax_acc.set_ylim(0.2, 0.8)
    ax_auc.set_ylim(0.2, 0.8)
    ax_acc.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    for tick_label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels() + ax_auc.get_yticklabels():
        tick_label.set_fontweight("bold")

    lines = acc_line + auc_line
    labels = [line.get_label() for line in lines]
    ax_acc.legend(lines, labels, loc="upper left", frameon=False, prop={"size": LEGEND_LABEL_SIZE, "weight": "bold"})

    fig.tight_layout()

    fig.savefig(output_dir / "sample_size_accuracy_auc.pdf")
    plt.close(fig)


def plot_no_defense_auc_bar(output_dir: Path) -> None:
    labels = ["No Defense", "Ours(Train)", "Ours(Infer)"]
    auc_values = [0.719, 0.537, 0.502]
    bar_colors = ["#cc5a5c", "#4a90b5", "#70a37f"]

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        labels,
        auc_values,
        width=0.52,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_ylabel("AUC")
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, width=2.5)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=2.5)
    ax.set_ylim(0.2, 0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    for bar, value in zip(bars, auc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_SIZE,
            fontweight="bold",
        )

    fig.tight_layout()
    
    fig.savefig(output_dir / "no_defense_auc_bar.pdf")
    plt.close(fig)


def plot_dp_past_grouped_bar(output_dir: Path) -> None:
    metrics = ["Test Accuracy", "Attack AUC"]
    accuracy_percent = [50.630, 69.630]
    dp_values = [accuracy_percent[0] / 100, 0.501]
    past_values = [accuracy_percent[1] / 100, 0.5358]
    dp_color = "#d15b5f"
    past_color = "#4990b8"
    x_positions = [0, 0.68]
    bar_width = 0.22
    axis_label_size = 24
    tick_label_size = 21
    legend_label_size = 22
    value_label_size = 20

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    dp_bars = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        dp_values,
        width=bar_width,
        color=dp_color,
        edgecolor="black",
        linewidth=1.2,
        label="DP",
    )
    past_bars = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        past_values,
        width=bar_width,
        color=past_color,
        edgecolor="black",
        linewidth=1.2,
        label="PAST",
    )

    ax_auc = ax.twinx()

    ax.set_ylabel("Accuracy")
    ax_auc.set_ylabel("AUC")
    ax.yaxis.label.set_size(axis_label_size)
    ax.yaxis.label.set_weight("bold")
    ax_auc.yaxis.label.set_size(axis_label_size)
    ax_auc.yaxis.label.set_weight("bold")
    ax.tick_params(axis="x", labelsize=tick_label_size, width=2.5)
    ax.tick_params(axis="y", labelsize=tick_label_size, width=2.5)
    ax_auc.tick_params(axis="y", labelsize=tick_label_size, width=2.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.2, 0.8)
    ax_auc.set_ylim(0.2, 0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax_auc.get_yticklabels():
        tick_label.set_fontweight("bold")

    for bar, value in zip(dp_bars, dp_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=value_label_size,
            fontweight="bold",
        )

    for bar, value in zip(past_bars, past_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=value_label_size,
            fontweight="bold",
        )

    ax.legend(
        loc="upper right",
        frameon=False,
        prop={"size": legend_label_size, "weight": "bold"},
    )
    fig.tight_layout()

    fig.savefig(output_dir / "dp_past_grouped_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent
    # plot_seed_metrics(output_path)
    # plot_sample_size_metrics(output_path)
    # plot_no_defense_auc_bar(output_path)
    plot_dp_past_grouped_bar(output_path)
