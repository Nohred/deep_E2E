from pathlib import Path
import json

import torch

from callbacks.early_stopping import EarlyStopping
from datasets.voc import VOC_CLASSES, get_voc_loaders
from engine.trainer import fit
from models.dect import ConvDetector
from models.vgg import VggDetector
from utils.device import get_device
from utils.plotting import plot_detection, plot_history
import time


def main():
    batch_size = 32
    image_size = (256, 256)
    learning_rate = 1e-3
    num_epochs = 100
    patience = 6
    min_delta = 1e-3
    weight_decay = 1e-4
    lambda_box = 1.0
    data_dir = "./data"
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    train_loader, val_loader = get_voc_loaders(data_dir, batch_size, image_size, image_net=True)

    # Model

    # model = ConvDetector(num_classes=len(VOC_CLASSES)).to(device)
    model = VggDetector(num_classes=len(VOC_CLASSES), pretrained=True, freeze_backbone=True).to(device)

    # Losses
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_box = torch.nn.SmoothL1Loss()\

    # Optimizer 
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # Scheduler - Reduce learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Early stopping - Stop training when a monitored metric has stopped improving
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)


    # Training
    start_time = time.time()
    history = fit(model, train_loader, val_loader, criterion_cls, criterion_box, optimizer, device,
                  num_epochs, lambda_box, early_stopping, scheduler)
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.\n")

    # Save the best model checkpoint and metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": len(VOC_CLASSES),
        "classes": list(VOC_CLASSES),
        "box_format": "xyxy",
        "boxes_normalized": True,
        "image_size": image_size,
        "image_range": "[0, 1] from torchvision.transforms.functional.to_tensor",
        "history": history,
        "lambda_box": lambda_box,
        "best_val_loss": early_stopping.best_loss,
    }
    torch.save(checkpoint, artifacts / "best_detector.pth")
    (artifacts / "detector_metadata.json").write_text(json.dumps({
        key: value for key, value in checkpoint.items() if key != "model_state_dict"
    }, indent=2, default=list))

    # Evaluation
    model.eval()
    images, labels, target_boxes = next(iter(val_loader))
    with torch.inference_mode():
        pred_boxes, class_logits = model(images.to(device))
        pred_classes = class_logits.argmax(dim=1)
        scores = class_logits.softmax(dim=1).amax(dim=1)

    # Plotting
    plot_detection(images, target_boxes, labels, pred_boxes, pred_classes, VOC_CLASSES,
                   scores=scores, max_images=min(4, len(images)), save_path=artifacts / "validation_detection.png")
    
    plot_history(history, save_path=artifacts / "history.png")


if __name__ == "__main__":
    main()
