import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms#,models
from dehaze_single_noev import dehaze_func
from piqa import SSIM
epochs=50
class adj_para(nn.Module):
    def __init__(self):
        super(adj_para,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        output1 = 1 + self.sigmoid(output1)
        output2 = 1 + self.sigmoid(output2)
        return torch.cat((output1, output2), dim=1)
       
def loss_func(output,target):#input as tensor
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian').cuda()
    return 1-ssim_loss(output, target)

class train_set(Dataset):
    def __init__(self,root_dir):
        self.img_pathlist_GT=[]
        self.img_pathlist_hazy=[]
        GT_path="GT"
        Hazy_path="hazy"
        hazy_path=os.path.join(root_dir,Hazy_path)
        gt_path=os.path.join(root_dir,GT_path)
        image_path=os.listdir(hazy_path)
        image_group=[img for img in image_path if os.path.isfile(os.path.join(hazy_path,img))]
        for img in image_group:
            self.img_pathlist_hazy.append(os.path.join(hazy_path,img))
            gt_img = img.replace('hazy', 'GT')
            self.img_pathlist_GT.append(os.path.join(gt_path,gt_img))
              
    def __len__(self):
        return len(self.img_pathlist_hazy)
    
    def __getitem__(self, idx):
        img_path_hazy=self.img_pathlist_hazy[idx]
        img_path_GT=self.img_pathlist_GT[idx]
        img_hazy=cv2.imread(img_path_hazy,4)
        #img_hazy=cv2.resize(img_hazy,(640,360))
        img_GT=cv2.imread(img_path_GT,4)
        #img_GT=cv2.resize(img_GT,(640,360))
        img_hazy = cv2.cvtColor(img_hazy, cv2.COLOR_BGR2RGB)
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
        group={'GT':img_GT,'Hazy':img_hazy}
        return group
   
def collate(batch):
    res=[]
    for dir in batch:
        res_dir={}
        GT_img=dir['GT']
        Hazy_img=dir['Hazy']
        res_dir['GT']=GT_img
        res_dir['Hazy']=Hazy_img
        res.append(res_dir)
    return res
        
class DehazeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, imglist, errd, e2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = transforms.ToTensor()
        img_dehaze_tensor = torch.tensor([]).to(device)
        for img in imglist:
            img_dehaze = dehaze_func(img, errd.item(), e2.item())
            img_dehaze_tensor_single=transform(img_dehaze).to(device).unsqueeze(0).requires_grad_()
            img_dehaze_tensor=torch.cat((img_dehaze_tensor, img_dehaze_tensor_single), dim=1)
        ctx.save_for_backward(img_dehaze_tensor[0], errd, e2)
        return img_dehaze_tensor

    @staticmethod
    def backward(ctx, grad_output):
        img, errd, e2 = ctx.saved_tensors
        # 获取图像尺寸
        xmax, ymax = img.shape[-2:]
        # 计算 mx 和 my
        mx = torch.arange(0, xmax, dtype=torch.float32).reshape(1, xmax).to(img.device) * 2 * np.pi / (xmax - 1)
        my = torch.arange(0, ymax, dtype=torch.float32).reshape(1, ymax).to(img.device) * 2 * np.pi / (ymax - 1)
        # 计算 dx 和 dy
        dx = (mx ** 2) / (1 + errd * mx ** 2) ** e2
        dy = (my ** 2) / (1 + errd * my ** 2) ** e2
        # 计算 x 方向的梯度
        grad_errd = (-2 * mx ** 4 * e2 / (1 + errd * mx ** 2) ** (e2 + 1)).sum() * grad_output.sum()
        grad_e2 = (-dx * torch.log(1 + errd * mx ** 2)).sum() * grad_output.sum()
        # 计算 y 方向的梯度
        grad_errd += (-2 * my ** 4 * e2 / (1 + errd * my ** 2) ** (e2 + 1)).sum() * grad_output.sum()
        grad_e2 += (-dy * torch.log(1 + errd * my ** 2)).sum() * grad_output.sum()
        return None, grad_errd, grad_e2
 
def train_single(batch,model,optimizer1,optimizer2,loss_fn=loss_func):#single graph training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    img_GT_tensor = torch.tensor([]).to(device)
    img_hazy_list=[]
    for item in batch:
        img_GT=item['GT']
        img_hazy=item['Hazy']
        img_GT_tensor_single = transform(img_GT).to(device).unsqueeze(0).requires_grad_()
        img_GT_tensor=torch.cat((img_GT_tensor, img_GT_tensor_single), dim=1)
        img_hazy_list.append(img_hazy)
    input_tensor = torch.tensor([1.0,1.5], dtype=torch.float32).to(device).unsqueeze(0)
    theta = model(input_tensor)
    errd, e2 = theta[0]
    print(errd.item(),e2.item())
    img_dehazed_tensor=DehazeFunction.apply(img_hazy_list, errd, e2)
    loss = loss_fn(img_dehazed_tensor, img_GT_tensor)
    print(loss)
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    print("== Gradients ==")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm()}")  # 打印梯度范数
        else:
            print(f"{name}: No gradient")
    optimizer1.step()
    optimizer2.step()
    
def train(root_dir):
    set=train_set(root_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(set, batch_size=8, shuffle=True, collate_fn=collate)
    model=adj_para()
    model.train()
    model.to(device)
    optimizer1 = optim.Adam(model.fc1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model.fc2.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in dataloader:
            train_single(batch,model,optimizer1,optimizer2)
    torch.save(model.state_dict(),'model.pth')
       
def evaluate(ckpt_path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=adj_para()
    model.to(device)
    checkpoint=torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.eval()
    init_params = torch.tensor([1.0, 1.5], requires_grad=True).float().unsqueeze(0).to(device)
    outputs = model(init_params)
    outputs=outputs.squeeze(0).cpu().detach().numpy()
    errd, e2 = outputs[0], outputs[1]
    print(errd,e2)

if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    train("dataset\dataset\O-HAZE\# O-HAZY NTIRE 2018")
    evaluate("model.pth")