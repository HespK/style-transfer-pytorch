import os
import argparse
import torch

from image_utils import imload, imsave
from loss import calc_Content_Loss, calc_Gram_Loss, calc_TV_Loss, extract_features
from vgg import get_vgg_feature_network

def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda-device-no', type=int,
                    help='cpu: -1, gpu: 0 ~ n ', default=0)

    parser.add_argument('--iterations', type=int,
                    help='number of iteration for stylization image', default=100)

    parser.add_argument('--lr', type=float,
                    help='Learning rate to optimize the image', default=0.1)

    parser.add_argument('--imsize', type=int,
                    help='Size for resize image during training', default=256)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=None)

    parser.add_argument('--vgg-flag', type=str,
                    help='VGG flag for calculating losses: vgg16 or vgg19 ', default='vgg16')

    parser.add_argument('--content-layers', type=int, nargs='+', 
                    help='layer indices to extract content features, vgg16: 15, vgg19: 20', default=[15])
    
    parser.add_argument('--style-layers', type=int, nargs='+',
                    help='layer indices to extract style features, vgg16: [3,8,15,22], vgg19: [1,6,11,20,29]', default=[3, 8, 15, 22])

    parser.add_argument('--content-weight', type=float, 
                    help='content loss weight', default=1.0)
    
    parser.add_argument('--style-weight', type=float,
                    help='style loss weight', default=50.0)

    parser.add_argument('--tv-weight', type=float,
                    help='tv loss weight', default=5.0)

    parser.add_argument('--noise-content-ratio', type=float,
                    help='input image ratio, 0: noise image, 1: content image', default=0.0)

    parser.add_argument('--target-content-filename', type=str,
                    help="target content image filename", required=True)

    parser.add_argument('--target-style-filename', type=str,
                    help="target stye image filename", required=True)
    
    parser.add_argument('--save-filename', type=str,
                    help='filename for stylized image', default='stylized.png')
    
    return parser

def stylize_image(vgg, device,
                  content_image, style_image, 
                  content_weight, style_weight, tv_weight,
                  content_layers, style_layers,
                  learning_rate, iterations,
                  noise_content_ratio):
    
    # input image
    input_image = torch.randn_like(content_image).to(device)
    input_image = input_image*(1- noise_content_ratio) +  content_image.detach() * noise_content_ratio
    
    # optimizer
    optimizer = torch.optim.LBFGS ([input_image.requires_grad_()], lr=learning_rate)

        
    for i in range(iterations):
        def closure():                    
            optimizer.zero_grad()

            # extract features
            target_content_features = extract_features(vgg, content_image.detach(), content_layers)
            target_style_features = extract_features(vgg, style_image.detach(), style_layers)
            
            input_content_features = extract_features(vgg, input_image, content_layers)
            input_style_features = extract_features(vgg, input_image, style_layers)
            
            # calculate losses
            content_loss = calc_Content_Loss(input_content_features, target_content_features)
            style_loss = calc_Gram_Loss(input_style_features, target_style_features)
            tv_loss = calc_TV_Loss(input_image)
            
            total_loss = content_loss * content_weight + style_loss * style_weight + tv_loss * tv_weight
                            
            # optimization
            total_loss.backward()
            
            return total_loss
        
        # optimization
        optimizer.step(closure)
                
    return input_image


if __name__ == '__main__':
    # get arguments
    parser = build_parser()
    args= parser.parse_args()
    
    # gpu device set
    if args.cuda_device_no >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)
    device = torch.device('cuda' if args.cuda_device_no >= 0 else 'cpu')
    
    # load target images
    content_image = imload(args.target_content_filename, args.imsize, args.cropsize)
    content_image = content_image.to(device)

    style_image = imload(args.target_style_filename, args.imsize, args.cropsize)
    style_image = style_image.to(device)

    # load pre-trianed vgg
    vgg = get_vgg_feature_network(args.vgg_flag)
    vgg = vgg.to(device)

    # stylize image
    output_image = stylize_image(vgg=vgg, device=device, 
            content_image=content_image, style_image=style_image,
            content_weight=args.content_weight, style_weight=args.style_weight, tv_weight=args.tv_weight,
            content_layers=args.content_layers, style_layers=args.style_layers, 
            learning_rate=args.lr, iterations=args.iterations, noise_content_ratio=args.noise_content_ratio)

    imsave(output_image.cpu(), args.save_filename)
