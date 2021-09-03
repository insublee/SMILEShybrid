from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, (input_size+output_size)//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear((input_size+output_size)//2, output_size)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
