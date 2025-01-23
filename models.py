import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml

# Load the config to get num_classes
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
num_classes = config.get('num_classes', 2)  # Default to 2 if not provided

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class SimpleCNN(nn.Module):
    # for sanity checks, a very very simple architecture.
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input channels adjusted to 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Adjust according to your input size
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # if self.num_classes==1:
        #     return torch.sigmoid(x)
        # else:
        return x

class Classifier(nn.Module):
    def __init__(self, base_model, num_classes=num_classes, dropout_prob=0.5):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout_prob)
        if hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.base_model, 'classifier'):
            if isinstance(self.base_model.classifier, nn.Sequential):
                in_features = self.base_model.classifier[-1].in_features
                self.base_model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        
        # if self.num_classes==1:
        #     return torch.sigmoid(x)
        # else:
        return x

def modify_input_layer(model, input_channels):
    if isinstance(model, models.ResNet):
        model.conv1 = nn.Conv2d(input_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, 
                                stride=model.conv1.stride, padding=model.conv1.padding, bias=(model.conv1.bias is not None))
    elif isinstance(model, models.DenseNet):
        model.features.conv0 = nn.Conv2d(input_channels, model.features.conv0.out_channels, kernel_size=model.features.conv0.kernel_size,
                                         stride=model.features.conv0.stride, padding=model.features.conv0.padding, bias=(model.features.conv0.bias is not None))
    elif isinstance(model, models.EfficientNet):
        model.features[0][0] = nn.Conv2d(input_channels, model.features[0][0].out_channels, kernel_size=model.features[0][0].kernel_size,
                                         stride=model.features[0][0].stride, padding=model.features[0][0].padding, bias=(model.features[0][0].bias is not None))
    elif isinstance(model, models.VGG):
        model.features[0] = nn.Conv2d(input_channels, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                      stride=model.features[0].stride, padding=model.features[0].padding, bias=(model.features[0].bias is not None))
    elif isinstance(model, models.AlexNet):  # Add handling for AlexNet
        model.features[0] = nn.Conv2d(input_channels, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                      stride=model.features[0].stride, padding=model.features[0].padding, bias=(model.features[0].bias is not None))
    else:
        raise ValueError("Unsupported model type for modifying input layer")
    return model


def get_model(model_name, num_classes=num_classes, pretrained=False):
    if model_name == 'resnet50':
        base_model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        base_model = models.resnet101(pretrained=pretrained)
    elif model_name == 'alexnet':
        base_model = models.alexnet(pretrained=pretrained)
    elif model_name == 'densenet121':
        base_model = models.densenet121(pretrained=pretrained)
    elif model_name == 'densenet169':
        base_model = models.densenet169(pretrained=pretrained)
    elif model_name == 'densenet201':
        base_model = models.densenet201(pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        base_model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == 'vgg16':
        base_model = models.vgg16(pretrained=pretrained)
    elif model_name == 'vgg19':
        base_model = models.vgg19(pretrained=pretrained)
    elif model_name == 'simplecnn':  # Add the new case for SimpleCNN
        base_model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    # Modify the input layer to accept 1 channel (if applicable)
    if model_name not in ["simplecnn"]:
        base_model = modify_input_layer(base_model, input_channels=1)
    
    # Apply custom weight initialization
    base_model.apply(weights_init)
    
    return Classifier(base_model, num_classes)

if __name__ == "__main__":
    model_names = ['resnet50', 'densenet121', 'densenet201', 'efficientnet_b0', 'vgg16', 'simplecnn']
    models = {name: get_model(name) for name in model_names}

    # Print model summaries to verify
    for name, model in models.items():
        print(f"Model: {name}")
        print(model)
        print("="*50)
