import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os





def SOTS_indoor_source():

    # 数据集文件夹
    clean_folder = "/media/cbq/Warehouse/Dataset/RESIDE/SOTS/indoor/gt/"
    haze_folder = "/media/cbq/Warehouse/Dataset/RESIDE/SOTS/indoor/hazy/"
    haze_list = glob.glob(haze_folder + "*")

    psnr_list = []
    ssim_list = []

    index = 1
    num = len(haze_list)

    for haze_image_path in haze_list:
        str = haze_image_path.split("/")[-1]
        name = str[0:4] + '.png'
        clean_image_path = clean_folder + name

        # 读取图片，BGR转RGB
        img_clean = cv2.imread(clean_image_path, 4)
        img_haze = cv2.imread(haze_image_path, 4)

        height = img_clean.shape[0]
        width = img_clean.shape[1]
        img_clean = img_clean[10:(height-10),10:(width-10)]

        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze, 255)
        ssim_haze , _ = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True,gaussian_weights=True, sigma=1.5)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num,'/',index)
        index = index +1

    psnr_ave = round(sum(psnr_list)/len(psnr_list), 4)
    ssim_ave = round(sum(ssim_list)/len(ssim_list), 4)

    return psnr_ave,ssim_ave

def SOTS_outdoor_source():

    # 数据集文件夹
    clean_folder = "/media/cbq/Warehouse/Dataset/RESIDE/SOTS/outdoor/gt/"
    haze_folder = "/media/cbq/Warehouse/Dataset/RESIDE/SOTS/outdoor/hazy/"
    haze_list = glob.glob(haze_folder + "*")

    psnr_list = []
    ssim_list = []

    index = 1
    num = len(haze_list)

    for haze_image_path in haze_list:
        str = haze_image_path.split("/")[-1]
        name = str[0:4] + '.png'
        clean_image_path = clean_folder + name

        # 读取图片，BGR转RGB
        img_clean = cv2.imread(clean_image_path, 4)
        img_haze = cv2.imread(haze_image_path, 4)

        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze, 255)
        ssim_haze , _ = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num,'/',index)
        index = index +1

    psnr_ave = round(sum(psnr_list)/len(psnr_list), 4)
    ssim_ave = round(sum(ssim_list)/len(ssim_list), 4)

    return psnr_ave,ssim_ave


def SOTS_indoor_aodnet():

    # 数据集文件夹
    clean_folder = "/media/cbq/Warehouse/Dataset/RESIDE/SOTS/indoor/gt/"
    haze_folder = "/home/cbq/quan/project/dehaze/AOD-Net/results/SOTO/indoor/dehaze_AODNet/"
    haze_list = glob.glob(haze_folder + "*")

    psnr_list = []
    ssim_list = []

    index = 1
    num = len(haze_list)

    for haze_image_path in haze_list:
        str = haze_image_path.split("/")[-1]
        name = str[0:4] + '.png'
        clean_image_path = clean_folder + name

        # 读取图片，BGR转RGB
        img_clean = cv2.imread(clean_image_path, 4)
        img_haze = cv2.imread(haze_image_path, 4)

        height = img_clean.shape[0]
        width = img_clean.shape[1]
        img_clean = img_clean[10:(height-10),10:(width-10)]

        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze, 255)
        ssim_haze , _ = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num,'/',index)
        index = index +1

    psnr_ave = round(sum(psnr_list)/len(psnr_list), 4)
    ssim_ave = round(sum(ssim_list)/len(ssim_list), 4)

    return psnr_ave,ssim_ave

def SOTS_outdoor_aodnet():

    # 数据集文件夹
    clean_folder = "dataset/dataset/O-HAZE/# O-HAZY NTIRE 2018/GT"
    haze_folder = "dataset/dataset/O-HAZE/# O-HAZY NTIRE 2018/hazy"
    haze_list = glob.glob(haze_folder + "\\*.jpg")

    psnr_list = []
    ssim_list = []

    index = 1
    num = len(haze_list)
    print(num)
    for haze_image_path in haze_list:
        print(haze_image_path)
        str = haze_image_path.split("/")[-1]
        name = '/'+str[0:10]+'_GT.jpg'
        clean_image_path = clean_folder + name
        print(clean_image_path)

        # 读取图片，BGR转RGB
        img_clean = cv2.imread(clean_image_path, 4)
        img_haze = cv2.imread(haze_image_path, 4)
        img_clean = cv2.resize(img_clean,(640,360))
        img_haze = cv2.resize(img_haze,(640,360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        # cv2.imshow("clean_img",img_clean)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        win_size=3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True,win_size=win_size)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num,'/',index)
        index = index +1

    psnr_ave = round(sum(psnr_list)/len(psnr_list), 4)
    ssim_ave = round(sum(ssim_list)/len(ssim_list), 4)

    return psnr_ave,ssim_ave

import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def D_HAZY_DATA_EVA():
    # 数据集文件夹
    clean_folder = 'dataset\dataset\O-HAZE\# O-HAZY NTIRE 2018\GT'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'dataset\dataset\O-HAZE\# O-HAZY NTIRE 2018\hazy'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []

    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.jpg') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)

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

        win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True, win_size=win_size,gaussian_weights=True, sigma=1.5)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

def D_HAZY_DATA_EVA_MID():
    # 数据集文件夹
    clean_folder = 'D-HAZY_DATASET\Middlebury_GT'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'D-HAZY_DATASET\Middlebury_Clean'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []

    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.bmp') or filename.endswith('.JPG')]#modify bmp
    num = len(haze_list)
    print(num)

    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        haze_path=os.path.basename(haze_image_path)
        clean_path=haze_path.replace('Hazy.bmp','im0.png')
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

        win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True, win_size=win_size,gaussian_weights=True, sigma=1.5)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

def RICE_DATA_EVA():
    # 数据集文件夹
    clean_folder = 'RICE_DATASET\RICE2\clean'#"./D-HAZY_DATASET/NYU_Hazy"
    haze_folder = 'RICE_DATASET\RICE2\cloud'#"./D-HAZY_DATASET/NYU_HAZY_CLEAN"

    psnr_list = []
    ssim_list = []

    index = 1

    # 使用 os.listdir 遍历 haze_folder 中的所有文件
    haze_list = [os.path.join(haze_folder, filename) for filename in os.listdir(haze_folder) if filename.endswith('.png')]#modify bmp
    num = len(haze_list)
    print(num)

    for haze_image_path in haze_list:
        # 构造对应的干净图像路径
        clean_image_path = os.path.join(clean_folder, os.path.basename(haze_image_path))  # 使用 os.path.basename 获取文件名
        print(clean_image_path)

        # 读取图片，BGR 转 RGB
        img_clean = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取原始图像
        img_haze = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)

        # 检查图像是否成功读取
        if img_clean is None or img_haze is None:
            print(f"Error reading images: {clean_image_path} or {haze_image_path}")
            continue

        img_clean = cv2.resize(img_clean, (640, 360))
        img_haze = cv2.resize(img_haze, (640, 360))
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)

        win_size = 3

        # 评价
        psnr_haze = peak_signal_noise_ratio(img_clean, img_haze)
        ssim_haze = structural_similarity(img_clean, img_haze, data_range=255, multichannel=True, win_size=win_size)
        psnr_list.append(psnr_haze)
        ssim_list.append(ssim_haze)

        print(num, '/', index)
        index += 1

    psnr_ave = round(sum(psnr_list) / len(psnr_list), 4) if psnr_list else 0
    ssim_ave = round(sum(ssim_list) / len(ssim_list), 4) if ssim_list else 0

    return psnr_ave, ssim_ave

if __name__ == '__main__':

    # SOTS_indoor_source_psnr,SOTS_indoor_source_ssim=SOTS_indoor_source()
    # SOTS_outdoor_source_psnr,SOTS_outdoor_source_ssim=SOTS_outdoor_source()
    # SOTS_all_source_psnr = round((SOTS_indoor_source_psnr+SOTS_outdoor_source_psnr)/2, 4)
    # SOTS_all_source_ssim = round((SOTS_indoor_source_ssim+SOTS_outdoor_source_ssim)/2, 4)
    #
    # print('SOTS_indoor_source_psnr =', SOTS_indoor_source_psnr)
    # print('SOTS_indoor_source_ssim =', SOTS_indoor_source_ssim)
    # print('SOTS_outdoor_source_psnr =', SOTS_outdoor_source_psnr)
    # print('SOTS_outdoor_source_ssim =', SOTS_outdoor_source_ssim)
    # print('SOTS_all_source_psnr =', SOTS_all_source_psnr)
    # print('SOTS_all_source_ssim =', SOTS_all_source_ssim)


    # SOTS_indoor_aodnet_psnr,SOTS_indoor_aodnet_ssim=SOTS_indoor_aodnet()
    # SOTS_outdoor_aodnet_psnr,SOTS_outdoor_aodnet_ssim=SOTS_outdoor_aodnet()
    # SOTS_all_aodnet_psnr = round((SOTS_indoor_aodnet_psnr+SOTS_outdoor_aodnet_psnr)/2, 4)
    # SOTS_all_aodnet_ssim = round((SOTS_indoor_aodnet_ssim+SOTS_outdoor_aodnet_ssim)/2, 4)
    dhazy_psnr,dhazy_ssim=D_HAZY_DATA_EVA()
    # dhazy_psnr,dhazy_ssim=RICE_DATA_EVA()
    # print('SOTS_indoor_aodnet_psnr =', SOTS_indoor_aodnet_psnr)
    # print('SOTS_indoor_aodnet_ssim =', SOTS_indoor_aodnet_ssim)
    # print('SOTS_outdoor_aodnet_psnr =', SOTS_outdoor_aodnet_psnr)
    # print('SOTS_outdoor_aodnet_ssim =', SOTS_outdoor_aodnet_ssim)
    # print('SOTS_all_aodnet_psnr =', SOTS_all_aodnet_psnr)
    # print('SOTS_all_aodnet_ssim =', SOTS_all_aodnet_ssim)
    print(dhazy_psnr,dhazy_ssim)


