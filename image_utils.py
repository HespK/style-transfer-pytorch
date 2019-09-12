import torchvision
import torchvision.transforms as transforms

from PIL import Image

# RGB mean and std of imagenet dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])

def get_transformer(imsize=None, cropsize=None):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.RandomCrop(cropsize))

    transformer.append(transforms.ToTensor())
    transformer.append(normalize)
    return transforms.Compose(transformer)


def imload(path, imsize=None, cropsize=None):
    transformer = get_transformer(imsize=imsize, cropsize=cropsize)
    image = Image.open(path).convert("RGB")
    return transformer(image).unsqueeze(0)

def imsave(image, save_path):
    image = denormalize(torchvision.utils.make_grid(image)).clamp_(0.0, 1.0)
    torchvision.utils.save_image(image, save_path)
    return None
