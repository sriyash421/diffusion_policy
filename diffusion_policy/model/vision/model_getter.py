import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


# create a depth encoder with a custom CNN backbone
# class DepthEncoder(torch.nn.Module):
#     def __init__(self, name='resnet18', weights=None):
#         super(DepthEncoder, self).__init__()
#         self.base_model = get_resnet(name, weights=weights)
#         self.base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     def forward(self, x):
#         return self.base_model(x)

def get_depth_encoder(name='resnet18', weights=None):
    # return DepthEncoder(name=name, weights=weights)
    return get_resnet(name, weights=None)