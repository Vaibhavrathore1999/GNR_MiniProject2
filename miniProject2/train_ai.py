import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
from model import DeblurModel
# from torch.utils.tensorboard import SummaryWriter

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

# device = "cpu"
model = DeblurModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the dataset and dataloader
data_transform = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = DeblurDataset('/home/ninad/vaibhav_r/shubh/image_pairs.csv', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 25
save_interval = 2  # Save weights every 2 epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, (blur_imgs, sharp_imgs) in enumerate(dataloader):
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(blur_imgs)
        loss = criterion(outputs, sharp_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Displaying batch progress bar
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss:.4f}')

    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

    # Save weights every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f'weights_epoch_{epoch+1}.pth')
        print(f'Saved weights at epoch {epoch+1}')

