from torch import nn
from torchvision import models

class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, embedding_dim)
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc(x))
        return x
