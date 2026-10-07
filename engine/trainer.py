import torch


def _box_iou(pred_boxes, target_boxes):
    """IoU after clipping predictions; inverted boxes receive zero IoU."""
    clipped = pred_boxes.clamp(0.0, 1.0)
    pred_width = clipped[:, 2] - clipped[:, 0]
    pred_height = clipped[:, 3] - clipped[:, 1]
    target_width = target_boxes[:, 2] - target_boxes[:, 0]
    target_height = target_boxes[:, 3] - target_boxes[:, 1]
    valid = (pred_width > 0) & (pred_height > 0) & (target_width > 0) & (target_height > 0)
    inter_left = torch.maximum(clipped[:, 0], target_boxes[:, 0])
    inter_top = torch.maximum(clipped[:, 1], target_boxes[:, 1])
    inter_right = torch.minimum(clipped[:, 2], target_boxes[:, 2])
    inter_bottom = torch.minimum(clipped[:, 3], target_boxes[:, 3])
    intersection = (inter_right - inter_left).clamp(min=0) * (inter_bottom - inter_top).clamp(min=0)
    union = pred_width.clamp(min=0) * pred_height.clamp(min=0)
    union += target_width.clamp(min=0) * target_height.clamp(min=0) - intersection
    return torch.where(valid & (union > 0), intersection / union.clamp(min=1e-12), torch.zeros_like(union)), ~valid


def _metrics(pred_boxes, class_logits, labels, target_boxes):
    iou, invalid = _box_iou(pred_boxes, target_boxes)
    return (
        (class_logits.argmax(dim=1) == labels).sum().item(),
        iou.sum().item(),
        invalid.sum().item(),
    )


def train_one_epoch(model, dataloader, criterion_cls, criterion_box, optimizer, device, lambda_box):
    model.train()
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "correct": 0, "iou": 0.0, "invalid": 0, "samples": 0}
    for images, labels, target_boxes in dataloader:
        images, labels, target_boxes = images.to(device), labels.to(device), target_boxes.to(device)
        optimizer.zero_grad()
        pred_boxes, class_logits = model(images)
        loss_cls = criterion_cls(class_logits, labels)
        loss_box = criterion_box(pred_boxes, target_boxes)
        loss = loss_cls + lambda_box * loss_box
        loss.backward()
        optimizer.step()
        count = images.size(0)
        correct, iou, invalid = _metrics(pred_boxes.detach(), class_logits.detach(), labels, target_boxes)
        totals["loss"] += loss.item() * count
        totals["loss_cls"] += loss_cls.item() * count
        totals["loss_box"] += loss_box.item() * count
        totals["correct"] += correct
        totals["iou"] += iou
        totals["invalid"] += invalid
        totals["samples"] += count
    return _average(totals)


def evaluate(model, dataloader, criterion_cls, criterion_box, device, lambda_box):
    model.eval()
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "correct": 0, "iou": 0.0, "invalid": 0, "samples": 0}
    with torch.no_grad():
        for images, labels, target_boxes in dataloader:
            images, labels, target_boxes = images.to(device), labels.to(device), target_boxes.to(device)
            pred_boxes, class_logits = model(images)
            loss_cls = criterion_cls(class_logits, labels)
            loss_box = criterion_box(pred_boxes, target_boxes)
            loss = loss_cls + lambda_box * loss_box
            count = images.size(0)
            correct, iou, invalid = _metrics(pred_boxes, class_logits, labels, target_boxes)
            totals["loss"] += loss.item() * count
            totals["loss_cls"] += loss_cls.item() * count
            totals["loss_box"] += loss_box.item() * count
            totals["correct"] += correct
            totals["iou"] += iou
            totals["invalid"] += invalid
            totals["samples"] += count
    return _average(totals)


def _average(totals):
    n = totals.pop("samples")
    if n == 0:
        raise ValueError("El dataloader no contiene ejemplos.")
    return {
        "loss": totals["loss"] / n,
        "loss_cls": totals["loss_cls"] / n,
        "loss_box": totals["loss_box"] / n,
        "accuracy": totals["correct"] / n,
        "mean_iou": totals["iou"] / n,
        "invalid_box_rate": totals["invalid"] / n,
    }


def fit(model, train_loader, val_loader, criterion_cls, criterion_box, optimizer, device, epochs,
        lambda_box=1.0, early_stopping=None, scheduler=None):
    history = {key: [] for key in (
        "train_loss", "val_loss", "train_loss_cls", "val_loss_cls",
        "train_loss_box", "val_loss_box", "train_accuracy", "val_accuracy",
        "train_mean_iou", "val_mean_iou", "train_invalid_box_rate",
        "val_invalid_box_rate", "learning_rate",
    )}
    for epoch in range(epochs):
        train = train_one_epoch(model, train_loader, criterion_cls, criterion_box, optimizer, device, lambda_box)
        val = evaluate(model, val_loader, criterion_cls, criterion_box, device, lambda_box)
        history["train_loss"].append(train["loss"])
        history["val_loss"].append(val["loss"])
        for name in ("loss_cls", "loss_box", "accuracy", "mean_iou", "invalid_box_rate"):
            history[f"train_{name}"].append(train[name])
            history[f"val_{name}"].append(val[name])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}: train={train['loss']:.4f}, val={val['loss']:.4f}, "
              f"acc={val['accuracy']:.3f}, IoU={val['mean_iou']:.3f}, lr={history['learning_rate'][-1]:.2e}")
        if scheduler is not None:
            scheduler.step(val["loss"])
        if early_stopping is not None:
            early_stopping(model, val["loss"])
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
    if early_stopping is not None:
        early_stopping.restore_best_model(model)
    return history
