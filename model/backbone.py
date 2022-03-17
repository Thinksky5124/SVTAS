# ref:https://github.com/ruotianluo/pytorch-resnet/blob/master/resnet.py
import torch
import torch.nn as nn
import torchvision.models.resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.utils.model_zoo as model_zoo

# form neckwork
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth"
}
# from local
# model_urls = {
#     "resnet18": "./data/resnet18-f37072fd.pth",
#     "resnet34": "./data/resnet34-b627a593.pth",
#     "resnet50": "./data/resnet50-0676ba61.pth",
#     "resnet101": "./dataresnet101-63fe2227.pth",
#     "resnet152": "./data/resnet152-394f9c45.pth"
# }

class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # change
        for i in range(2, 5):
            getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        # form neckwork
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # from local
        # model.load_state_dict(torch.load(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        # form neckwork
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        # from local
        # model.load_state_dict(torch.load(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        # form neckwork
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # from local
        # model.load_state_dict(torch.load(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        # form neckwork
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        # from local
        # model.load_state_dict(torch.load(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        # form neckwork
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        # from local
        # model.load_state_dict(torch.load(model_urls['resnet152']))
    return model