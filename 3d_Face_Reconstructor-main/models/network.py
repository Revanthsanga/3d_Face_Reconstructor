# models/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceReconstructionNet(nn.Module):
    def __init__(self, coeff_dim: int = 299, num_triangles: int = 94464):
        super().__init__()
        self.num_triangles = num_triangles

        # CNN backbone
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Updated for your 3DMM dimensions: 199 (identity) + 100 (expression) = 299
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc_coeff = nn.Linear(256, coeff_dim)  # Now 299 instead of 228
        
        # UV prediction - matches your face count (94464 triangles)
        self.uv_fc = nn.Linear(256, 512)
        self.uv_out = nn.Linear(512, num_triangles * 3 * 2)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        coeff_out = self.fc_coeff(x)
        
        # UV prediction with tanh activation to bound outputs
        uv_features = F.relu(self.uv_fc(x))
        uv_out = self.uv_out(uv_features).view(-1, self.num_triangles, 3, 2)
        uv_out = torch.tanh(uv_out)  # Bound UVs to [-1, 1]
        
        return coeff_out, uv_out