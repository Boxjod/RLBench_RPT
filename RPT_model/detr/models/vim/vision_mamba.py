import numpy as np
import torch

import models_mamba # to @register_model

# 构建模型并加载模型权重
device = torch.device('cuda')
model_ckpt='/home/boxjod/RLBench/Mamba/Vim/ckpt/vim_s_midclstok_ft_81p6acc.pth'

model = models_mamba.vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=True,ckpt_path=model_ckpt, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224).to(device)

# model_without_ddp = model
# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('number of params:', n_parameters/1024//1024,'M')




# 验证
# 只输入一张图像，获取他的输出分类值
model.eval()
# image_path = '/home/boxjod/RLBench/Mamba/Vim/ImageNet-Mini/val/n01440764/ILSVRC2012_val_00046252.JPEG' # 0
# image_path = '/home/boxjod/RLBench/Mamba/Vim/ImageNet-Mini/val/n01514668/ILSVRC2012_val_00048952.JPEG' # 8
image_path = '/home/boxjod/RLBench/Mamba/Vim/ImageNet-Mini/val/n01677366/ILSVRC2012_val_00033961.JPEG' # 40 

from PIL import Image

from torchvision.transforms import transforms
transform_valid = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor()])

img = Image.open(image_path)
img_ = transform_valid(img) # .unsqueeze(0) #拓展维度
img_ = img_[np.newaxis, :]
print(f"{img_.shape}\n")

img_ = img_.to(device)
outputs = model(img_, return_features=True)
print(outputs.shape)

# _, indices = torch.max(outputs,1)
# print(int(indices.data))









# # ImageNet-Mini数据集加载与验证
# import argparse
# from pathlib import Path
# from datasets import build_dataset
# from engine import evaluate
# import utils
# def get_args_parser():
#     parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
#     # Dataset parameters
#     parser.add_argument('--input-size', default=224, type=int, help='images input size')
#     parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],type=str, help='Image Net dataset path')
#     parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
#     return parser

# parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()

# if args.output_dir:
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# # 加载数据集
# args.data_path='/home/boxjod/RLBench/Mamba/Vim/ImageNet-Mini'
# dataset_val, nb_classes = build_dataset(is_train=False, args=args) #############################
# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
# data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=int(1.5 * 64), num_workers=10, pin_memory=True, drop_last=False)

# # 数据集验证
# test_stats = evaluate(data_loader_val, model, device, torch.cuda.amp.autocast)
# print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


