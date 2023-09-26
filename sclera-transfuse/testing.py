
from Sclera_TransUNet import UNet1
import glob
import numpy as np
import scipy.misc
import cv2
import scipy.io
import os, sys, argparse
import time
import shutil
from os.path import join, splitext, split, isfile
# import pandas as pd
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.nn as nn

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def segment(net, test_dir, save_dir, device):
    net.eval()
    imgs = [i for i in os.listdir(test_dir) if '.jpg' in i]
    nimgs = len(imgs)
    if nimgs == 0:
        print("no  images")
        return
    else:
        print("totally " + str(nimgs) + " images")
        start = time.time()
        with torch.no_grad():
            for i in range(nimgs):
                img_path = join(test_dir, imgs[i])
                image = rgb_loader(img_path)
                x,y=image.size[0:2]
                seed = np.random.randint(2147483647)  # make a seed with numpy generator
                random.seed(seed)  # apply this seed to img tranfsorms
                torch.manual_seed(seed)
                img_transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
                image = img_transform(image)
                image = image.reshape(1, 3, image.shape[1], image.shape[1])
                image = image.to(device=device, dtype=torch.float32)
                mask_pred = net(image)
                pred = torch.sigmoid(mask_pred)
                pred = np.array(pred.data.cpu().detach()[0])[0]
                pred = pred * 255
                # pred[pred >=0.5]=255
                # pred[pred < 0.5]=0
                pred = cv2.resize(pred, (x, y), interpolation=cv2.INTER_NEAREST)
                fn, ext = splitext(imgs[i])
                cv2.imwrite(join(save_dir, fn + '.png'), pred)
                print("Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..." % (i + 1, nimgs))
        end = time.time()
        avg_time = (end - start) / nimgs
        print("average time is %f seconds" % avg_time)

if __name__ == "__main__":
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    net = UNet1(128, 1)
    net = nn.DataParallel(net, device_ids=[2])
    net.to(device=device)
    net.load_state_dict(torch.load("./checkpoints/transfuse2.pth", map_location=device))

    img_paths = []
    test_path =  "./data/TestingSplit/"
    save_path = "./data/pre/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    segment(net, test_path, save_path, device)










