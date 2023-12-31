from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model import SODModel
from src.dataloader import InfDataloader, SODLoader, pad_resize_image
import torchvision.transforms as transforms


# def parse_arguments():
def parse_arguments(img):
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    # parser.add_argument('--imgs_folder', default='./images', type=str)
    parser.add_argument('--img', default=img, type=str)
    parser.add_argument('--model_path', default='./fpn/models/best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=128, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    return parser.parse_args()


def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    # inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    inf_data = InfDataloader(img=args.img, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)

    # # print("Press 'q' to quit.")
    with torch.no_grad():
        # for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
        #     img_tor = img_tor.to(device)
        #     pred_masks, _ = model(img_tor)

        #     # Assuming batch_size = 1
        #     img_np = np.squeeze(img_np.numpy(), axis=0)
        #     img_np = img_np.astype(np.uint8)
        #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        #     pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        #     pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

        # img = cv2.imread(img)
        img = args.img
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pad images to target size
        img_np = pad_resize_image(img, None, args.img_size)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
        img_tor = normalize(img_tor)

        # img_tor = img_tor.to(device)
        img_tor = img_tor.unsqueeze(0).to(device)

        pred_masks, _ = model(img_tor)

        # Assuming batch_size = 1
        # img_np = np.squeeze(img_np.numpy(), axis=0)
        # img_np = np.squeeze(img_np, axis=0)
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))
    
    return pred_masks_raw, pred_masks_round


def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    print('MAE for the test set is :', np.mean(mae_list))


# if __name__ == '__main__':
#     results_dict = {}
#     rt_args = parse_arguments()
    
#     # Call the run_inference function and capture the results
#     pred_masks_raw_list, pred_masks_round_list = run_inference(rt_args)
    
#     # Store the results in the dictionary
#     results_dict['pred_masks_raw'] = pred_masks_raw_list
#     results_dict['pred_masks_round'] = pred_masks_round_list
