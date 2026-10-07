
# Technique Early Stopping

import copy

class EarlyStopping:
    def __init__(self, patience: int=3, min_delta: float=1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

        self.best_model_state = None

    def __call__(self,model, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict()) # lo que hace deepcopy es crear una copia completa del objeto, incluyendo todos los objetos anidados, en lugar de solo copiar las referencias a esos objetos. 
            #Esto significa que cualquier cambio realizado en la copia no afectará al objeto original y viceversa.
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)