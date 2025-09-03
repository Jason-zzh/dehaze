import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from skimage.metrics import peak_signal_noise_ratio
import os
from piqa import SSIM
import torch
from CIEDE200 import calculate_ciede2000
import torch.nn.functional as F
def D_HAZY_DATA_EVA():
    # 数据集文件夹
    clean_folder = 'main dataset\O-HAZE\O-HAZE\# O-HAZY NTIRE 2018\GT'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'main dataset\O-HAZE\O-HAZE\# O-HAZY NTIRE 2018\clean'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian')
    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.jpg') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)
    print(ssim_loss)
    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        
        haze_path=os.path.basename(haze_image_path)
        clean_path=haze_path.replace('hazy','GT')
        clean_image_path = os.path.join(clean_folder, clean_path)  # 使用 os.path.basename 获取文件名
        print(clean_image_path)

        # 读取图片，BGR 转 RGB
        img_clean = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取原始图像
        img_haze = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)

        # 检查图像是否成功读取
        if img_clean is None or img_haze is None:
            print(f"Error reading images: {clean_image_path} or {haze_image_path}")
            continue

        #img_clean = cv2.resize(img_clean, (640, 360))
        #img_haze = cv2.resize(img_haze, (640, 360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        img_hazy_tensor = torch.tensor(img_haze, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        img_GT_tensor = torch.tensor(img_clean, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        #win_size = 3

        # 评价
        # mse_loss=F.mse_loss(img_GT_tensor,img_hazy_tensor,reduction='none').mean((1,2,3))
        # psnr_haze= 10*torch.log10(1/mse_loss).item()
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = ssim_loss(img_hazy_tensor,img_GT_tensor).item()
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

def D_HAZY_DATA_EVA_MID():
    # 数据集文件夹
    clean_folder = "main dataset/D-HAZE/NYU_GT"
    haze_folder = "main dataset/D-HAZE/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []
    cie_list=[]
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian')
    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.bmp') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)#13.5472 0.7306
    max_ssim=0
    max_path=0
    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        haze_path=os.path.basename(haze_image_path)
        clean_path=haze_path.replace('Hazy','Image_')
        clean_image_path = os.path.join(clean_folder, clean_path)  # 使用 os.path.basename 获取文件名
        print(clean_image_path)

        # 读取图片，BGR 转 RGB
        img_clean = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取原始图像
        img_haze = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)

        # 检查图像是否成功读取
        if img_clean is None or img_haze is None:
            print(f"Error reading images: {clean_image_path} or {haze_image_path}")
            continue

        #img_clean = cv2.resize(img_clean, (640, 360))
        #img_haze = cv2.resize(img_haze, (640, 360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        img_hazy_tensor = torch.tensor(img_haze, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        img_GT_tensor = torch.tensor(img_clean, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        #win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = ssim_loss(img_hazy_tensor,img_GT_tensor).item()
        cie_haze=calculate_ciede2000(clean_image_path,haze_image_path)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)
        cie_list.append(cie_haze)
        # if ssim_haze > max_ssim:
        #     max_path=haze_image_path
        #     max_ssim=ssim_haze
        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0
    cie_ave=round(sum(cie_list)/len(cie_list),4) if cie_list else 0
    #print(max_path)
    return psnr_ave, ssim_ave,cie_ave

def D_HAZY_DATA_EVA_DHID():
    # 数据集文件夹
    clean_folder = 'main dataset\I-HAZE\I-HAZE (1)\# I-HAZY NTIRE 2018\GT'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'Cycle-Dehaze/results/indoor/temp'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian')
    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.png') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)

    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        haze_path=os.path.basename(haze_image_path)
        clean_path=haze_path[0:2]+"_indoor_GT.jpg"
        clean_image_path = os.path.join(clean_folder, clean_path)  # 使用 os.path.basename 获取文件名
        print(clean_image_path)

        # 读取图片，BGR 转 RGB
        img_clean = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取原始图像
        img_haze = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)
        img_clean = cv2.resize(img_clean, (img_haze.shape[1],img_haze.shape[0]))
        # 检查图像是否成功读取
        if img_clean is None or img_haze is None:
            print(f"Error reading images: {clean_image_path} or {haze_image_path}")
            continue

        #img_clean = cv2.resize(img_clean, (640, 360))
        #img_haze = cv2.resize(img_haze, (640, 360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        img_hazy_tensor = torch.tensor(img_haze, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        img_GT_tensor = torch.tensor(img_clean, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        #win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = ssim_loss(img_hazy_tensor,img_GT_tensor).item()
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

def D_HAZY_DATA_EVA_TEST():
    # 数据集文件夹
    clean_folder = 'test_data_gt'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'test_data_clean'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian')
    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.bmp') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)

    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        haze_path=os.path.basename(haze_image_path)
        clean_path=haze_path.replace('Hazy','Image_')
        clean_image_path = os.path.join(clean_folder, clean_path)  # 使用 os.path.basename 获取文件名
        print(clean_image_path)

        # 读取图片，BGR 转 RGB
        img_haze = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)
        img_clean = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取原始图像
        

        # 检查图像是否成功读取
        if img_clean is None or img_haze is None:
            print(f"Error reading images: {clean_image_path} or {haze_image_path}")
            continue

        #img_clean = cv2.resize(img_clean, (640, 360))
        #img_haze = cv2.resize(img_haze, (640, 360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        img_hazy_tensor = torch.tensor(img_haze, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        img_GT_tensor = torch.tensor(img_clean, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        #win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = ssim_loss(img_hazy_tensor,img_GT_tensor).item()
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

if __name__ == '__main__':
    dhazy_psnr,dhazy_ssim=D_HAZY_DATA_EVA()
    #dhazy_psnr,dhazy_ssim=D_HAZY_DATA_EVA_DHID()
    #dhazy_psnr,dhazy_ssim=D_HAZY_DATA_EVA_TEST()
    
    #dhazy_psnr,dhazy_ssim,cie200=D_HAZY_DATA_EVA_MID()
    print(dhazy_psnr,dhazy_ssim)#,cie200)