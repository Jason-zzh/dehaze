import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img1_p="dataset/dataset/O-HAZE/# O-HAZY NTIRE 2018/hazy/01_outdoor_hazy.jpg"
img_hazy=cv2.imread(img1_p,4)
img_hazy=cv2.cvtColor(img_hazy, cv2.COLOR_BGR2RGB)
cv2.imshow('BMP Image', img_hazy)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_hazy_tensor=torch.tensor(img_hazy, dtype=torch.float32).to(device)
img_hazy_new=img_hazy_tensor.cpu().numpy()
print(img_hazy_tensor.shape)
cv2.imshow('BMP Image', img_hazy_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
