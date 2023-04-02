import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetRelu(nn.Module): # base virtual class
    def __init__(self, ):
        super().__init__()
        self.layers = nn.Sequential()
        self.name = 'NetRelu'

    def forward(self, x):
        return self.layers(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path + '/' + self.name)

    def load_model(self, path):
        self.load_state_dict(torch.load(path + '/' + self.name, map_location = torch.device(DEVICE)))
        self.eval()

class CNN_SRM(NetRelu): # CNN implementation inspired by SRM paper for emnist extension
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.Conv2d(32, 32, 3),
            # nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, 3),
            # nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.4),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128, 26),
            nn.Dropout(0.4),
            nn.Softmax(dim = 1))
        self.name = 'CNN_SRM_letter'
        print('created model: ', self.name)

class NetReluShallow(NetRelu): # shallow model that allows self define size
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, D_out)
        )
        self.mec = (D_in + 1) * H1 +\
                   min(H1, (H1 + 1) * H2) +\
                    min(H2, (H2 + 1) * D_out)
        self.name = 'NetReluShallow_MEC_' + str(self.mec) + '_DIM_' + str(D_in) + '_' + str(H1) + '_' + str(H2) + '_' + str(D_out)
        print('created model: ', self.name)


class NetReluDeep(NetRelu): # deep dark fantasy
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10))
        self.name = 'NetReluDeep'
        print('created model: ', self.name)
        

