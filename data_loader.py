import os
import torch
import nibabel as nib
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))

class CamusSequenceDataset(Dataset):
    def __init__(self, data_dir, target_shape=(256, 256)):
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")

        all_files = sorted([f for f in os.listdir(data_dir) if "half_sequence.nii" in f and "_gt" not in f])

        for f in all_files:
            img_path = os.path.join(data_dir, f)
            mask_path = os.path.join(data_dir, f.replace(".nii", "_gt.nii"))
            visao = "2CH" if "2CH" in f else "4CH"

            if os.path.exists(mask_path):
                img_obj = nib.load(img_path)
                num_frames = img_obj.shape[-1] 
                for i in range(num_frames):
                    self.samples.append({
                        "img_path": img_path, 
                        "mask_path": mask_path, 
                        "frame_idx": i, 
                        "visao": visao
                    })
        
        print(f"Dataset inicializado com {len(self.samples)} frames totais.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carregando os dados do frame específico
        img_vol = nib.load(sample["img_path"]).get_fdata()
        mask_vol = nib.load(sample["mask_path"]).get_fdata()
        
        img_frame = img_vol[:, :, sample["frame_idx"]]
        mask_frame = mask_vol[:, :, sample["frame_idx"]]

        # Redimensionamento
        img_res = cv2.resize(img_frame, self.target_shape, interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_frame, self.target_shape, interpolation=cv2.INTER_NEAREST)

        # Lógica de 2 canais (LV_2CH e LV_4CH) para o Biplano de Simpson
        target = np.zeros((2, self.target_shape[0], self.target_shape[1]), dtype=np.float32)
        lv_mask = (mask_res == 1).astype(np.float32) # Classe 1 é o LV

        if sample["visao"] == "2CH":
            target[0, :, :] = lv_mask
        else:
            target[1, :, :] = lv_mask

        # Normalização 0-1
        img_res = (img_res - np.min(img_res)) / (np.max(img_res) - np.min(img_res) + 1e-8)

        # Retorno dos Tensores (C, H, W)
        img_tensor = torch.from_numpy(img_res).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target).float()

        return img_tensor, target_tensor

if __name__ == "__main__":
    dataset = CamusSequenceDataset(data_dir=DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, targets = next(iter(dataloader))
    print(f"Sucesso! Imagens: {images.shape} | Targets: {targets.shape}")