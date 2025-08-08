from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import model.parameters as parameters
from const.index import BASE_PATH

class FruitClassifier(pl.LightningModule):
    def __init__(self, num_classes=1):
        super(FruitClassifier, self).__init__()
        
        self.resnet = torchvision.models.resnet34(pretrained=True)
        
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False
            
        num_fc_inputs = self.resnet.fc.in_features
        self.resnet.fc  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_fc_inputs, num_classes)
        )

        
    def forward(self, x):
        output = self.resnet(x)
        return output
        
    def val_dataloader(self):
        val_dataset = ImageFolder(BASE_PATH / "valid", transform=parameters.transform)
        val_loader = DataLoader(val_dataset, batch_size=parameters.batch_size, shuffle=False,
                                    num_workers=0, persistent_workers=False)
        return val_loader
    
    def train_dataloader(self):
        train_dataset = ImageFolder(BASE_PATH / "train", transform=parameters.transform)
        train_loader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True,
                                    num_workers=0, persistent_workers=False)
        return train_loader
    
    def training_step(self, batch, batch_index):
        images, labels = batch
        labels = labels.unsqueeze(1).float()

        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_index):
        images, labels = batch
        labels = labels.unsqueeze(1).float()

        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        
        probs = F.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        
        num_samples = labels.shape[0]
        num_correct = (predictions == labels).sum().item()
        accuracy = round(num_correct / num_samples, 4)

        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=parameters.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "accuracy",
                "interval": "epoch",
                "frequency": 1,
            }
        }