import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MarginsDataset(Dataset):
    """
    Dataset que internamente hace el split de 80% - 20% para entrenamiento y test.
    Se agrega código defensivo para verificar que las imágenes se carguen correctamente
    y tengan la resolución esperada (512 x 1024).
    """

    def __init__(self,
                 dir_real="C:/Users/planz/Pictures/Images_512x1024",
                 dir_no_margin="./Images_No_Margin",
                 do_normalize=True,
                 mode="train",
                 split_ratio=0.8,
                 seed=42):
        """
        :param dir_real: Directorio con las imágenes con márgenes.
        :param dir_no_margin: Directorio con las imágenes sin márgenes.
        :param do_normalize: Indica si queremos normalizar las imágenes.
        :param mode: "train" o "test" (indica qué subconjunto usar).
        :param split_ratio: Proporción para entrenamiento (ej. 0.8 -> 80%).
        :param seed: Semilla para reproducir la separación aleatoria.
        """
        super().__init__()
        self.dir_real = dir_real
        self.dir_no_margin = dir_no_margin
        self.do_normalize = do_normalize
        self.mode = mode.lower().strip()
        self.split_ratio = split_ratio

        # Conversión a tensor y normalización opcionales
        self.to_tensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # Carga y empareja las imágenes (con validación)
        data_pairs_all = self._get_data_pairs()

        # Mezclamos para que la división sea aleatoria
        random.seed(seed)
        random.shuffle(data_pairs_all)

        # Calculamos el índice de corte para el split
        split_index = int(len(data_pairs_all) * split_ratio)
        self.train_pairs = data_pairs_all[:split_index]
        self.test_pairs = data_pairs_all[split_index:]

    def _get_data_pairs(self):
        """
        Crea una lista de tuplas (img_sin_margen, img_con_margen) en formato PIL.
        Verifica que cada imagen se cargue sin problemas y tenga la resolución esperada.
        """
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        real_files = sorted(
            f for f in os.listdir(self.dir_real)
            if os.path.splitext(f)[1].lower() in valid_exts
        )
        no_margin_files = sorted(
            f for f in os.listdir(self.dir_no_margin)
            if os.path.splitext(f)[1].lower() in valid_exts
        )

        data_pairs = []
        expected_resolution = (512, 1024)  # PIL devuelve (ancho, alto)

        for fname in real_files:
            if fname in no_margin_files:
                path_real = os.path.join(self.dir_real, fname)
                path_no_margin = os.path.join(self.dir_no_margin, fname)

                try:
                    # Se intenta cargar la imagen
                    img_real = Image.open(path_real)
                    img_no_margin = Image.open(path_no_margin)

                    # Verificar la resolución
                    if img_real.size != expected_resolution:
                        raise ValueError(
                            f"La imagen real '{fname}' tiene resolución {img_real.size}, se esperaba {expected_resolution}.")
                    if img_no_margin.size != expected_resolution:
                        raise ValueError(
                            f"La imagen sin márgenes '{fname}' tiene resolución {img_no_margin.size}, se esperaba {expected_resolution}.")

                    # Convertir a modo RGB
                    img_real = img_real.convert('RGB')
                    img_no_margin = img_no_margin.convert('RGB')

                except Exception as e:
                    print(f"Error al cargar la imagen '{fname}': {e}")
                    # Para detener el programa en caso de error:
                    raise

                data_pairs.append((img_no_margin, img_real))

        if not data_pairs:
            raise RuntimeError("No se encontraron pares de imágenes válidos. Verifica los directorios y las imágenes.")

        return data_pairs

    def __len__(self):
        # Depende del modo: train o test
        return len(self.train_pairs) if self.mode == "train" else len(self.test_pairs)

    def __getitem__(self, index):
        # Selecciona la lista de acuerdo al modo (train o test)
        if self.mode == "train":
            img_no_margin_pil, img_real_pil = self.train_pairs[index]
        else:
            img_no_margin_pil, img_real_pil = self.test_pairs[index]

        # Convertir la imagen PIL a Tensor
        img_no_margin = self.to_tensor(img_no_margin_pil)
        img_real = self.to_tensor(img_real_pil)

        # Normalización opcional
        if self.do_normalize:
            img_no_margin = self.normalizer(img_no_margin)
            img_real = self.normalizer(img_real)

        return img_no_margin, img_real