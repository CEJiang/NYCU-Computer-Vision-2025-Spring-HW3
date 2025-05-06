"""
visualization.py

Utilities for visualizing training progress and detection results.
Includes:
- Loss and mAP curve plotting
- Single and batch image prediction visualizations
- Comparison of AP/AR metrics from log files
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# === AP/AR log parsing and plotting ===

metrics_to_plot = [
    "AP@[IoU=0.50:0.95]",
    "AP@[IoU=0.50]",
    "AP@[IoU=0.75]",
    "AP@[IoU=0.50:0.95|area=small]",
    "AP@[IoU=0.50:0.95|area=medium]",
    "AP@[IoU=0.50:0.95|area=large]",
    "AR@[IoU=0.50:0.95|maxDets=1]",
    "AR@[IoU=0.50:0.95|maxDets=10]",
    "AR@[IoU=0.50:0.95|maxDets=100]",
    "AR@[IoU=0.50:0.95|area=small]",
    "AR@[IoU=0.50:0.95|area=medium]",
    "AR@[IoU=0.50:0.95|area=large]"
]

metric_keywords = {
    "AP@[IoU=0.50:0.95]": "AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "AP@[IoU=0.50]": "AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
    "AP@[IoU=0.75]": "AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
    "AP@[IoU=0.50:0.95|area=small]": "AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "AP@[IoU=0.50:0.95|area=medium]": "AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "AP@[IoU=0.50:0.95|area=large]": "AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    "AR@[IoU=0.50:0.95|maxDets=1]": "AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
    "AR@[IoU=0.50:0.95|maxDets=10]": "AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
    "AR@[IoU=0.50:0.95|maxDets=100]": "AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "AR@[IoU=0.50:0.95|area=small]": "AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "AR@[IoU=0.50:0.95|area=medium]": "AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "AR@[IoU=0.50:0.95|area=large]": "AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
}


def parse_log(file_path):
    """
    Parses a COCO-style evaluation log file to extract all metric values per epoch.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: A dictionary where each key is a metric name and values are lists of floats across epochs.
    """
    metrics_dict = {k: [] for k in metrics_to_plot}
    with open(file_path, 'r') as file:
        for line in file:
            for metric in metrics_to_plot:
                if metric_keywords[metric] in line:
                    match = re.search(r"\] = ([0-9.]+)", line)
                    if match:
                        value = float(match.group(1))
                        metrics_dict[metric].append(value)
    return metrics_dict


def plot_metric_comparison(file1, file2, label1, label2):
    """
    Plot side-by-side comparisons of AP/AR metrics between two models.

    Args:
        file1 (str): Path to the first log file.
        file2 (str): Path to the second log file.
        label1 (str): Label for the first model.
        label2 (str): Label for the second model.
    """
    metrics1 = parse_log(file_path=file1)
    metrics2 = parse_log(file_path=file2)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle("PAN vs Non-PAN", fontsize=14)

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        ax.plot(metrics1[metric], label=label1, color='orange')
        ax.plot(metrics2[metric], label=label2, color='brown')
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True)
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./{label1}_vs_{label2}.png")
    plt.show()


def plot_single_log_curve_combined(file_path, save_path="single_combined.png"):
    """
    Plot all AP/AR metrics from a single log file into one combined line chart.

    Args:
        file_path (str): Path to the validation log file.
        save_path (str): Path to save the output combined figure.
    """
    plt.figure(figsize=(16, 10))
    metrics1 = parse_log(file_path=file_path)
    for metric in metrics_to_plot:
        plt.plot(range(1, len(metrics1[metric]) + 1),
                 metrics1[metric], marker='o', label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("All AP/AR Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.show()


def plot_loss_accuracy(train_losses, val_losses,
                       train_accuracies=None, val_accuracies=None,
                       save_fig=True, output_path="training_curve.png"):
    """
    Plot and optionally save training/validation loss and accuracy curves.

    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        train_accuracies (list of float, optional): Training accuracy.
        val_accuracies (list of float, optional): Validation accuracy.
        save_fig (bool): Whether to save the figure.
        output_path (str): Path to save the output plot.
    """
    num_plots = 2 if train_accuracies and val_accuracies else 1
    _, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    axes[0].plot(train_losses, label="Train Loss", marker='o')
    axes[0].plot(val_losses, label="Validation Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid()

    if num_plots == 2:
        axes[1].plot(train_accuracies, label="Train Acc", marker='o')
        axes[1].plot(val_accuracies, label="Val Acc", marker='o')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid()

    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path)
        print(f"Training curve saved as {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_single_log_curve_combined("map_log.txt", "./panandCBAM.png")

    # plot_metric_comparison("map_log_pan.txt", "map_log_0123.txt",
    #                     label1="PAN", label2="Non-PAN")
