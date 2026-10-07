from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def plot_detection(images, real_boxes, real_classes, pred_boxes, pred_classes,
                   class_names, scores=None, max_images=4, save_path=None):
    count = min(max_images, len(images))
    if count == 0:
        raise ValueError("No hay imágenes para visualizar.")
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for i in range(count):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        height, width = image.shape[:2]
        axis = axes[0, i]
        axis.imshow(np.clip(image, 0, 1))
        axis.axis("off")

        def draw_box(box, color, label):
            box = box.detach().cpu().float()
            valid = box.shape == (4,) and torch.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]
            if not valid:
                axis.text(0.02, 0.95, f"{label}: caja inválida", transform=axis.transAxes,
                          color=color, va="top", bbox={"facecolor": "white", "alpha": 0.7})
                return
            x1, y1, x2, y2 = (box * torch.tensor([width, height, width, height])).tolist()
            axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=color, linewidth=2))
            axis.text(x1, y1, label, color=color, backgroundcolor="white")

        draw_box(real_boxes[i], "green", f"Real: {class_names[int(real_classes[i])]}")
        label = f"Pred: {class_names[int(pred_classes[i])]}"
        if scores is not None:
            label += f" ({float(scores[i].detach().cpu()):.2f})"
        draw_box(pred_boxes[i], "red", label)
        axis.legend(handles=[
            patches.Patch(color="green", label="Real"),
            patches.Patch(color="red", label="Predicción"),
        ], loc="lower right", framealpha=0.7)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_ax, lr_ax) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for key, label, color in (
        ("train_loss", "Train total", "#2563eb"),
        ("val_loss", "Val total", "#dc2626"),
        ("train_loss_cls", "Train clasificación", "#7c3aed"),
        ("val_loss_cls", "Val clasificación", "#a855f7"),
        ("train_loss_box", "Train localización", "#0891b2"),
        ("val_loss_box", "Val localización", "#06b6d4"),
    ):
        if key in history:
            loss_ax.plot(epochs, history[key], label=label, linewidth=2)
    loss_ax.set_title("Pérdidas")
    loss_ax.set_xlabel("Época")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, linestyle="--", alpha=0.3)
    loss_ax.legend(frameon=False)
    lr_ax.plot(epochs, history["learning_rate"], label="Learning rate", color="#16a34a", linewidth=2)
    lr_ax.set_yscale("log")
    lr_ax.set_title("Learning rate")
    lr_ax.set_xlabel("Época")
    lr_ax.grid(True, linestyle="--", alpha=0.3)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig, loss_ax
