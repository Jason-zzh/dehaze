import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import math
from torch.utils.data import Dataset,DataLoader
from torch_dehaze import dehaze_func_torch
from piqa import SSIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DehazeOptimizer(nn.Module):
    def __init__(self):
        super(DehazeOptimizer, self).__init__()
        # 初始化参数为可学习变量
        self.errd = nn.Parameter(torch.tensor(1.0))  # 初始值
        self.e2 = nn.Parameter(torch.tensor(2.0))    # 初始值

    def forward(self, img):
        return dehaze_func_torch(img, self.errd, self.e2, device)

def loss_func(output,target):#input as tensor
    ssim_loss = SSIM(n_channels=3, window_size=3,reduction='gaussian').cuda()
    output=output.permute(0, 3, 1, 2)/255.0
    target=target.permute(0, 3, 1, 2)/255.0
    return 1-ssim_loss(output, target).mean()

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
        img_GT=cv2.imread(img_path_GT,4)
        img_hazy = cv2.cvtColor(img_hazy, cv2.COLOR_BGR2RGB)
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
        img_hazy =cv2.resize(img_hazy,(256,256))
        img_GT =cv2.resize(img_GT,(256,256))
        img_hazy_tensor = torch.tensor(img_hazy, dtype=torch.float32).to(device)
        img_GT_tensor = torch.tensor(img_GT, dtype=torch.float32).to(device)
        group={'GT':img_GT_tensor,'Hazy':img_hazy_tensor}
        return group
       
def train(model, hazy_image_batch, gt_image_batch, optimizer):
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    dehazed_image_batch_list=[]
    for hazy_image in hazy_image_batch:
        # 进行去雾计算
        dehazed_image = model(hazy_image)
        dehazed_image_batch_list.append(dehazed_image.unsqueeze(0))
    # 计算损失
    dehazed_image_batch=torch.cat(dehazed_image_batch_list, dim=0)
    loss = loss_func(dehazed_image_batch, gt_image_batch)
    loss.backward()
        
    # 更新参数
    optimizer.step()

    # 打印和记录损失
    return model,optimizer,loss

if __name__=='__main__':
    root_path="main dataset\I-HAZE\I-HAZE (1)\# I-HAZY NTIRE 2018"
    set=train_set(root_path)
    dataloader = DataLoader(set, batch_size=8, shuffle=True)
    model = DehazeOptimizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    epochs=200
    loss_history=[]
    for epoch in range(epochs):
        loss=0
        for batch in dataloader:
            model,optimizer,loss= train(model, batch['Hazy'], batch['GT'],optimizer)
            loss_history.append(loss.item())
            print("batch complete")
        print(f"epoch  {epoch + 1} complete")
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    print("Optimized errd:", model.errd.item())
    print("Optimized e2:", model.e2.item())
    torch.save(model.state_dict(),'model.pth')
    # 可视化损失变化
    plt.plot(loss_history)
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()