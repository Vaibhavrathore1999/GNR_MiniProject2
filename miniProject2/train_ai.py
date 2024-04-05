import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import ToTensor
from PIL import Image
from temp import DeblurNet
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class DeblurDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blur_path = self.data.iloc[idx, 1]
        sharp_path = self.data.iloc[idx, 2]
        
        blur_img = Image.open(blur_path)
        sharp_img = Image.open(sharp_path)
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img

# Define the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeblurNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the dataset and dataloader
transform = ToTensor()
dataset = DeblurDataset('/home/ninad/vaibhav_r/siddhant/SRN-Deblur/train_info.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# TensorBoard writer
writer = SummaryWriter()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)
    for i, (blur_imgs, sharp_imgs) in enumerate(dataloader):
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)
        print("Inside Training Loop")
        optimizer.zero_grad()
        
        outputs = model(blur_imgs)
        loss = criterion(outputs, sharp_imgs)
        print("Loss Calcluated")
        loss.backward()
        print("Backprop Done")
        optimizer.step()
        print("Parameters Updated")
        running_loss += loss.item()
        print("Loss Added to be displayed")
        # Update TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        
        # Update batch progress bar
        batch_progress_bar = tqdm(total=blur_imgs.size(0), desc=f"Batch {i+1}/{len(dataloader)}", position=1, leave=False)
        batch_progress_bar.update(blur_imgs.size(0))
        batch_progress_bar.set_postfix(loss=loss.item())
        batch_progress_bar.close()
        print("Epoch done")
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    epoch_progress_bar.update(len(dataloader))
    epoch_progress_bar.close()

# Save the model
torch.save(model.state_dict(), 'deblur_model.pth')
# Close TensorBoard writer
writer.close()