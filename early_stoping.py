import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, model, model_name, token_length):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
            print("Saving best model...")
            if not os.path.exists(f'../BERT_fine_tune'):
                os.makedirs(f'../BERT_fine_tune')
            torch.save({"model_state_dict": model.state_dict(), "token_length": token_length}, f"../BERT_fine_tune/{model_name}.pt")
            
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False