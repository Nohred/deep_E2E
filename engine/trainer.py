import torch

def train_one_epoch(
        model, 
        dataloader,
        criterion,
        optimizer,
        device,
        ):
    
    model.train() # modo entrenamiento

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels) # loss function - funcion de perdida

        optimizer.zero_grad() # limpiar gradientes
        loss.backward() # backpropagation
        optimizer.step() # actualizar pesos

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss  

def evaluate(
        model, 
        dataloader,
        criterion,
        device,
        ):
    
    model.eval() # modo evaluacion

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(): # desactiva el calculo de gradientes
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels) # loss function - funcion de perdida

            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    early_stopping= None,
    scheduler= None
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(
            model, 
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6e}")

        if scheduler is not None:
            scheduler.step(val_loss)  # Update learning rate based on validation loss

        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
    if early_stopping is not None:
        early_stopping.restore_best_model(model)
        print("Restored best model state.")

    return history