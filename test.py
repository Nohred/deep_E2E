from models.dect import ConvDetector
from utils.device import get_device
from utils.plotting import plot_detection, plot_history
from datasets.voc import VOC_CLASSES, get_voc_loaders
import torch
from pathlib import Path

image_size = (256, 256)
batch_size = 8
data_dir = "./data"
weights_path = Path("artifacts/best_detector.pth")
artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)


device = get_device()
print(f"Using device: {device}")

train_loader, val_loader = get_voc_loaders(data_dir, batch_size, image_size)

model = ConvDetector(num_classes=len(VOC_CLASSES)).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])


# model.eval()
# images, labels, target_boxes = next(iter(val_loader))
# with torch.inference_mode():
#     pred_boxes, class_logits = model(images.to(device))
#     pred_classes = class_logits.argmax(dim=1)
#     scores = class_logits.softmax(dim=1)
#     confidence, predicted_classes = scores.max(dim=1) # Softmax para obtener probabilidades y luego max para obtener la puntuación más alta

# for i in range(min(4, len(images))):
#     print(
#         f"Imagen {i}: "
#         f"clase={predicted_classes[i].item()}, "
#         f"score={confidence[i].item():.8f}, "
#         f"bbox={pred_boxes[i].cpu().tolist()}"
#     )

# print(
#     "Diferencia máxima de logits entre imagen 0 y 1:",
#     (class_logits[0] - class_logits[1]).abs().max().item()
# )


# plot_detection(images, target_boxes, labels, pred_boxes, pred_classes, VOC_CLASSES,
#                 scores=scores.amax(dim=1), max_images=min(4, len(images)), save_path=artifacts / "validation_detection.png")

model.eval()

images, labels, target_boxes = next(iter(val_loader))
images_device = images.to(device)

print("\n--- Entradas ---")
print("Shape:", tuple(images.shape))
print("Tipo:", images.dtype)
print("Todos los valores finitos:", torch.isfinite(images).all().item())

for i in range(min(4, images.size(0))):
    image = images[i]

    print(
        f"Imagen {i}: "
        f"min={image.min().item():.6f}, "
        f"max={image.max().item():.6f}, "
        f"std={image.std().item():.6f}, "
        f"label={labels[i].item()}, "
        f"target_box={target_boxes[i].tolist()}"
    )

if images.size(0) >= 2:
    print(
        "Diferencia máxima entre entradas 0 y 1:",
        (images[0] - images[1]).abs().max().item()
    )

print("\n--- Activaciones ---")

handles = []


def make_hook(name):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        values = output.detach()

        difference = float("nan")
        if values.ndim > 0 and values.size(0) >= 2:
            difference = (
                values[0] - values[1]
            ).abs().max().item()

        zero_fraction = (
            values == 0
        ).float().mean().item()

        print(
            f"{name}: "
            f"shape={tuple(values.shape)}, "
            f"min={values.min().item():.6e}, "
            f"max={values.max().item():.6e}, "
            f"zeros={zero_fraction:.3f}, "
            f"diff01={difference:.6e}"
        )

    return hook


tracked_types = (
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Linear,
)

for name, module in model.named_modules():
    if isinstance(module, tracked_types):
        handles.append(
            module.register_forward_hook(make_hook(name))
        )

try:
    with torch.inference_mode():
        pred_boxes, class_logits = model(images_device)
finally:
    for handle in handles:
        handle.remove()

scores = class_logits.softmax(dim=1)
confidence, predicted_classes = scores.max(dim=1)

print("\n--- Salidas ---")

for i in range(min(4, images.size(0))):
    print(
        f"Imagen {i}: "
        f"clase={predicted_classes[i].item()}, "
        f"score={confidence[i].item():.8f}, "
        f"bbox={pred_boxes[i].cpu().tolist()}"
    )


