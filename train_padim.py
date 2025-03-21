import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet50, resnet34

import lightly.models as models
import lightly.loss as loss
import lightly.data as data
from lightly.transforms import SimCLRTransform

from tqdm import tqdm
import pandas as pd

from utils import ImageDatasetTrain

#  train padim with unsupervised contrastive loss
# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),  # Standardize image size
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for stable training
#     ]
# )
transform = SimCLRTransform(
    input_size=224,
    min_scale=1,
    vf_prob=0,
    rr_prob=0,
    cj_prob=0.1,
    hf_prob=0,
)  # Adjust size if needed
# Load the dataset
data_path = "../input_train_new"
csv_path = "../Y_train_new.csv"
csv_true = pd.read_csv(csv_path)
classes_dict = {
    "GOOD": 0,
    "Boucle plate": 1,
    "Lift-off blanc": 2,
    "Lift-off noir": 3,
    "Missing": 4,
    "Short circuit MOS": 5,
    "Drift": 6,
}
csv_true["Label"] = csv_true["Label"].replace(classes_dict)

dataset = data.LightlyDataset(data_path, transform=transform)
# dataset = ImageDatasetTrain(data_path, csv_true=csv_true, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Use a model with resnet backbone
model = models.SimCLR(backbone=resnet50(pretrained=True), num_ftrs=512)
model.backbone.fc = torch.nn.Linear(in_features=2048, out_features=512, bias=False)
print(model)
criterion = loss.NTXentLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(30):
    total_loss = 0
    for imgs in tqdm(dataloader):
        img, _ = imgs[0], imgs[1]
        img1 = img[0].to(device)
        img2 = img[1].to(device)

        features1 = model(img1)
        features2 = model(img2)
        loss = criterion(features1, features2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Save the model
torch.save(model.backbone.state_dict(), "self_supervised_resnet34.pth")
