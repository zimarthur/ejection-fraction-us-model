import os
import torch
from torch.utils.data import Dataset

class CamusSequenceDataset(Dataset):
    def __init__(self, data_dir, allowed_patients=None):
        self.data_dir = data_dir
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")

        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

        for f in all_files:
            patient_id = f.split('_')[0]
            if allowed_patients is not None and patient_id not in allowed_patients:
                continue
            self.samples.append(os.path.join(data_dir, f))
        
        print(f"Dataset inicializado com {len(self.samples)} tensores.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        data = torch.load(file_path, weights_only=True)
        
        # Lógica de Classificação: Extrai a visão do nome do arquivo
        filename = os.path.basename(file_path)
        if "2CH" in filename:
            label = 0.0  # A2C
        else:
            label = 1.0  # A4C
            
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return data['img'], data['mask'], label_tensor