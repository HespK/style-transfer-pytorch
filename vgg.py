import torchvision.models as models

def get_vgg_feature_network(vgg_flag):
    vgg = models.__dict__[vgg_flag](pretrained=True).features
    for param in vgg.parameters():
        param.required_grad = False
    return vgg
