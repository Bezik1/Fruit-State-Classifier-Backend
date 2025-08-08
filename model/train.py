import torch
from model.model import FruitClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from . import parameters
        
if __name__ == "__main__":
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=parameters.patience,
        mode='min'
    )
    
    trainer = Trainer(max_epochs=parameters.num_epochs, fast_dev_run=False, callbacks=[early_stop])
    model = FruitClassifier(num_classes=1)
    trainer.fit(model)
    
    torch.save(model.state_dict(), "./models/model.pth")