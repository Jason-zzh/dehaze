import torch
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dehaze_single_noev import dehaze_func
import cv2

def dehaze_fft(img_in, errd_in, e2_in,device):

    # 傅里叶变换
    img_f = torch.fft.fft2(img_in).to(device)

    # 参数
    eer = 2
    errd = errd_in  # 影响度
    e2 = e2_in  # 平滑度
    pr = 5
    l = -0.35

    # 图片宽、长
    xmax, ymax = img_in.shape

    # 生成坐标
    xc = torch.ones((xmax, 1), dtype=torch.float32).to(device)
    yc = torch.ones((ymax, 1), dtype=torch.float32).to(device)
    xr = torch.arange(0, xmax, dtype=torch.float32).reshape(1, xmax).to(device)
    yr = torch.arange(0, ymax, dtype=torch.float32).reshape(1, ymax).to(device)
    mx = ((torch.matmul(yc, xr)).T * 2 * math.pi / (xmax - 1)).to(device)
    my = (torch.matmul(xc, yr) * 2 * math.pi / (ymax - 1)).to(device)

    # 主要公式
    dx = (mx ** eer) / ((1 + errd * (mx ** 2)) ** e2)
    dy = (my ** eer) / ((1 + errd * (my ** 2)) ** e2)
    
    # 倍数叠加
    mm = pr * (dx + dy)
    mm[0, 0] = l  # 保证中心频率处的值为l

    # 频域相乘
    img_fd = img_f * (mm + 1)

    # 傅里叶逆变换
    img_if = torch.fft.ifft2(img_fd).to(device)

    # 取实部并向下取整
    img_out = torch.real(img_if).to(device)

    # 数值限制在0-255
    img_out = torch.clamp(img_out, 0, 255).to(device)

    # 转换为uint8类型
    # img_out = img_out.to(torch.uint8).to(device)
    
    # 转换为float32
    # img_out = img_out.to(torch.float32).to(device)

    return img_out 

def dehaze_func_torch(img_haze, errd, e2,device):

    # 分离RGB通道
    img_r = img_haze[:, :, 0]# R
    img_g = img_haze[:, :, 1]# G
    img_b = img_haze[:, :, 2]# B

    # 对每个通道进行去雾处理
    img_r_dehazed = dehaze_fft(img_r, errd, e2, device)
    img_g_dehazed = dehaze_fft(img_g, errd, e2, device)
    img_b_dehazed = dehaze_fft(img_b, errd, e2, device)

    # 合并RGB通道
    img_dehaze = torch.stack([img_r_dehazed, img_g_dehazed, img_b_dehazed], dim=-1).to(device)

    return img_dehaze

if __name__=='__main__':
    img1_p="dataset/dataset/O-HAZE/# O-HAZY NTIRE 2018/hazy/01_outdoor_hazy.jpg"
    img2_p="dataset/dataset/O-HAZE/# O-HAZY NTIRE 2018/GT/01_outdoor_GT.jpg"
    img1=cv2.imread(img1_p,4)
    img2=cv2.imread(img2_p,4)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1_dehaze=dehaze_func(img1,0.42,4.47)
    print(structural_similarity(img1_dehaze,img2,data_range=255, multichannel=True, win_size=3))
    # img1_tensor = torch.tensor(img1, dtype=torch.float32)
    # img2_tensor = torch.tensor(img2, dtype=torch.float32)
    # img1_dehaze_tensor=dehaze_func_torch(img1_tensor,1,1.5)
    # img1_dehaze_new=img1_dehaze_tensor.numpy()
    # print(structural_similarity(img1_dehaze_new,img2,data_range=255, multichannel=True, win_size=3))
    
