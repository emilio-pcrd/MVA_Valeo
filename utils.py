import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []

        for img_name in os.listdir(self.data_path):
            img_path = os.path.join(self.data_path, img_name)
            self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(img_path)

        return filename, image


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


class ImageDatasetTrain(Dataset):
    def __init__(self, image_folder, csv_true, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        self.transform = transform
        self.csvtrue = csv_true

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        id_label = self.csvtrue[
            self.csvtrue["filename"] == self.image_files[idx]
        ].index[0]
        label = self.csvtrue["Label"][id_label]
        if self.transform:
            image = self.transform(image)

        return image, label


def rotate_and_crop_image(path, angle, crop_box, new_folder, plot=False):
    """
    Applique une rotation d'un angle donné et un crop sur une image, puis enregistre le résultat en .png.
    Optionnellement, affiche l'image si plot=True.

    Arguments :
    - input_image_path : Chemin de l'image d'entrée.
    - output_image_path : Chemin pour enregistrer l'image modifiée.
    - angle : L'angle de rotation en degrés (positif dans le sens antihoraire).
    - crop_box : Un tuple (left, upper, right, lower) définissant les points du crop en pixels.
    - plot : Booléen, si True affiche l'image modifiée.
    """
    # Ouvrir l'image
    image = Image.open(path)

    # Appliquer la rotation (expand=True permet d'agrandir l'image pour qu'elle s'ajuste au cadre après rotation)
    rotated_image = image.rotate(angle, expand=True)

    # Appliquer le crop avec les points fournis
    cropped_image = rotated_image.crop(crop_box)

    # Enregistrer l'image au format PNG
    filename = path.split("/")[-1]
    new_path = new_folder + filename

    cropped_image.save(new_path, format="PNG")
    # print(f"Image enregistrée avec succès sous {new_path}")

    # Si plot=True, afficher l'image modifiée
    if plot:
        plt.imshow(cropped_image)
        plt.axis("off")  # Masquer les axes
        plt.title(f"Image après rotation de {angle}° et crop")
        plt.show()


class ImageDatasetBinaryTrain(Dataset):
    def __init__(self, image_folder, csv_true, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform
        self.csvtrue = csv_true

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        id_label = self.csvtrue[
            self.csvtrue["filename"] == self.image_files[idx]
        ].index[0]
        label = self.csvtrue["Label"][id_label]
        if self.transform:
            image = self.transform(image)
        if label != 0:
            label = 1
        return image, label
