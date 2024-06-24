import torch
import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(Classifier, self).__init__()
        self.base_model = base_model
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
        return self.base_model(x)

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
    else:
        raise ValueError("Unsupported model type for modifying input layer")
    return model

def get_model(model_name, num_classes=2, pretrained=False):
    if model_name == 'resnet50':
        base_model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        base_model = models.resnet101(pretrained=pretrained)
    elif model_name == 'densenet121':
        base_model = models.densenet121(pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        base_model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == 'vgg16':
        base_model = models.vgg16(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    # Modify the input layer to accept 1 channel
    base_model = modify_input_layer(base_model, input_channels=1)
    
    return Classifier(base_model, num_classes)

if __name__ == "__main__":
    model_names = ['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0', 'vgg16']
    models = {name: get_model(name) for name in model_names}

    # Print model summaries to verify
    for name, model in models.items():
        print(f"Model: {name}")
        print(model)
        print("="*50)
