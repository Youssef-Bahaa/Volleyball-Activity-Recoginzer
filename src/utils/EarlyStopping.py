class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def state_dict(self):
        return {'counter': self.counter, 'best_loss': self.best_loss}

    def load_state_dict(self, state):
        self.counter   = state['counter']
        self.best_loss = state['best_loss']
