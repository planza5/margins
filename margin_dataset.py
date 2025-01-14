import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MarginsDataset(Dataset):
    """
    Dataset para manejar imágenes originales y sus versiones rotadas 180 grados.
    """
    def __init__(self,
                 dir_real="./Images_512x1024",
                 dir_margin="./Images_512x1024_Margin",
                 do_normalize=True,
                 mode="train",
                 split_ratio=0.8,
                 seed=42,
                 augment_rotation=False):
        """
        :param augment_rotation: Si es True, incluye imágenes rotadas 180 grados.
        """
        super().__init__()
        self.dir_real = dir_real
        self.dir_margin = dir_margin
        self.do_normalize = do_normalize
        self.mode = mode.lower().strip()
        self.split_ratio = split_ratio
        self.augment_rotation = augment_rotation

        # Preprocesamiento (normalización y conversión a tensor)
        self.to_tensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # Emparejar imágenes
        data_pairs_all = self._get_data_pairs()

        # Dividir en train/test
        random.seed(seed)
        random.shuffle(data_pairs_all)
        split_index = int(len(data_pairs_all) * split_ratio)
        self.train_pairs = data_pairs_all[:split_index]
        self.test_pairs = data_pairs_all[split_index:]

    def _get_data_pairs(self):
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        real_files = sorted(
            f for f in os.listdir(self.dir_real)
            if os.path.splitext(f)[1].lower() in valid_exts
        )
        no_margin_files = sorted(
            f for f in os.listdir(self.dir_margin)
            if os.path.splitext(f)[1].lower() in valid_exts
        )
        data_pairs = []
        for fname in real_files:
            if fname in no_margin_files:
                path_real = os.path.join(self.dir_real, fname)
                path_margin = os.path.join(self.dir_margin, fname)
                data_pairs.append((path_margin, path_real))
        return data_pairs

    def __len__(self):
        return len(self.train_pairs) if self.mode == "train" else len(self.test_pairs)

    def __getitem__(self, index):
        pairs = self.train_pairs if self.mode == "train" else self.test_pairs
        path_margin, path_real = pairs[index]

        # Cargar imágenes
        img_margin = Image.open(path_margin).convert('RGB')
        img_real = Image.open(path_real).convert('RGB')

        # Aplicar rotación aleatoria (si augment_rotation es True)
        if self.augment_rotation and random.random() > 0.5:
            img_margin = img_margin.rotate(180)
            img_real = img_real.rotate(180)

        # Convertir a tensores
        img_margin = self.to_tensor(img_margin)
        img_real = self.to_tensor(img_real)

        # Normalizar (opcional)
        if self.do_normalize:
            img_margin = self.normalizer(img_margin)
            img_real = self.normalizer(img_real)

        return img_margin, img_real
