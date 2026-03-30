import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class CelebACustom(Dataset):
    """
    CelebA dataset que lee directamente del disco (sin torchvision download).
    Filtra conceptos y target por nombre.

    Estructura esperada en root/:
        img_align_celeba/    <- imágenes
        list_attr_celeba.txt
        list_eval_partition.txt
    """

    def __init__(self, root, concept_names, target_name, split='train', transform=None):
        self.root = root
        self.transform = transform

        partition_path = os.path.join(root, 'list_eval_partition.txt')
        attr_path = os.path.join(root, 'list_attr_celeba.txt')

        # Cargar particiones (0=train, 1=valid, 2=test)
        split_map = {'train': 0, 'valid': 1, 'test': 2}
        split_df = pd.read_csv(
            partition_path, sep=r'\s+', header=None,
            names=['img', 'split'], engine='python'
        )
        self.img_names = split_df[split_df['split'] == split_map[split]]['img'].values

        # Cargar atributos: saltar primera línea (count), usar nombre imagen como índice
        attr_df = pd.read_csv(
            attr_path, sep=r'\s+', skiprows=1,
            index_col=0, engine='python'
        )
        # CelebA usa -1/+1, convertir a 0/1
        attr_df = (attr_df + 1) // 2

        # Filtrar solo las filas del split actual
        attr_df = attr_df.loc[self.img_names]

        # Pre-cargar conceptos y target como tensores
        self.concepts = torch.tensor(attr_df[concept_names].values, dtype=torch.float32)
        self.targets = torch.tensor(attr_df[target_name].values, dtype=torch.float32)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'img_align_celeba', self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.concepts[idx], self.targets[idx]
