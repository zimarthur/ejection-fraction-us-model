import os
import torch
from torch.utils.data import Dataset, DataLoader


class CamusSequenceDataset(Dataset):
    def __init__(self, data_dir, allowed_patients=None):
        self.data_dir = data_dir
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")

        # Busca apenas os nossos novos tensores .pt
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

        for f in all_files:
            # O ID do paciente continua sendo o primeiro bloco antes do '_'
            patient_id = f.split('_')[0]

            if allowed_patients is not None and patient_id not in allowed_patients:
                continue

            self.samples.append(os.path.join(data_dir, f))
        
        print(f"Dataset inicializado com {len(self.samples)} tensores otimizados.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Leitura direta do disco para a memória. Muito mais rápido!
        # weights_only=True é uma boa prática de segurança nas versões mais novas do PyTorch
        data = torch.load(self.samples[idx], weights_only=True)
        
        # O arquivo .pt já contém a imagem e a máscara no shape e normalização corretos
        return data['img'], data['mask']
