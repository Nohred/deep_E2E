from utils.plotting import plot_history, plot_segmentation
import torch
from utils.device import get_device, clear_device_cache
from datasets.voc import SegmentationTransform
from models.unet import UNet
from models.dect import ConvDetector
from engine.trainer import evaluate, fit
from callbacks.early_stopping import EarlyStopping

def main(): 
    ## HIPERPARAMETROS ##
    batch_size = 8 # No elementos a procesar al mismo tiempo
    learning_rate = 1e-3
    num_epochs = 100 # callback - early stopping - regularizar
    # ReduceRLROnPlateau - Cambia el ratio de aprendizaje cuando la métrica de validación deja de mejorar
    patience = 6
    min_delta = 1e-3 # criterio de mejora mínima para considerar que la métrica ha mejorado
    weight_decay = 1e-4

    
    # clear_device_cache()
    device = get_device() 
    print(f"Using device: {device}")

    train_loader, val_loader = SegmentationTransform.get_voc_loaders(data_dir="./data", batch_size=batch_size)

    ## MODEL ##

    model = UNet(num_classes=21)
    model = model.to(device)

    #####
    boxes, class_logits = model(images)

    loss_cls = torch.nn.CrossEntropyLoss()(class_logits, labels)
    loss_box = torch.nn.SmoothL1Loss()(boxes, target_boxes)

    loss = loss_cls + lambda_box * loss_box

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #######

    ## Train
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255) # funcion de perdida

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.5, 
                                                           patience=2) # Reduce el learning rate cuando la métrica de validación deja de mejorar

    history = fit(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    torch.save(model.state_dict(), "artifacts/best_model.pth")  # Guardar el modelo entrenado

    model.eval()  # Cambiar el modelo a modo de evaluación

    images, masks = next(iter(val_loader))  # Obtener un batch de imágenes, máscaras y predicciones del conjunto de validación
    with torch.inference_mode():  # Desactivar el cálculo de gradientes para la evaluación
        outputs = model(images.to(device))  # Obtener las predicciones del modelo
        predictions = torch.argmax(outputs, dim=1)  # Obtener la clase con la mayor probabilidad para cada píxel
    
    plot_segmentation(
        images=images,
        masks=masks,
        predictions=predictions,
    )

    plot_history(history)




if __name__ == "__main__":
    # main()

    train_loader, val_loader = SegmentationTransform.get_voc_loaders(data_dir="./data", batch_size=8)
    # 1. Reconstruir la MISMA arquitectura que usaste en entrenamiento
    model = UNet(num_classes=21)  # <-- aquí va tu clase de modelo, con los mismos args
    # 2. Cargar los pesos
    device = get_device()
    model.load_state_dict(torch.load("artifacts/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

     
    images, masks = next(iter(val_loader))  # Obtener un batch de imágenes, máscaras y predicciones del conjunto de validación
    with torch.inference_mode():  # Desactivar el cálculo de gradientes para la evaluación
        outputs = model(images.to(device))  # Obtener las predicciones del modelo
        predictions = torch.argmax(outputs, dim=1)  # Obtener la clase con la mayor probabilidad para cada píxel
    
    plot_segmentation(
        images=images,
        masks=masks,
        predictions=predictions,
    )