import torch.nn as nn
import torch.nn.functional as F

class myOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(myOCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(self.bn1(x))  

        x = F.relu(self.conv2(x))
        x = self.pool(self.bn2(x)) 

        x = F.relu(self.conv3(x))
        x = self.pool(self.bn3(x))  
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x