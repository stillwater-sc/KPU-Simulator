import torch
import torchvision.models as models
from transformers import BertModel

def load_model(model_name):
    if model_name == 'mobilenet_v2':
        return torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
    elif model_name == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        return models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        return torch.load('resnet50.pt')
    elif model_name == 'resnet101':
        return models.resnet101(pretrained=True)
    elif model_name == 'efficientnet_b0':
        return torch.hub.load('lukemelas/EfficientNet-PyTorch', 'efficientnet-b0', pretrained=True)
    elif model_name == 'bert':
        return BertModel.from_pretrained('bert-base-uncased')
    elif model_name == 'vit_b_16':
        from vit_pytorch import ViT
        return ViT('B_16_imagenet1k', pretrained=True)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

if __name__ == "__main__":
    # Example usage:
    model = load_model('resnet18')

    m = torch.compile(model)
    print(m)

