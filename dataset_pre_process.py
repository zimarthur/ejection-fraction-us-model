import os
import torch
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

def prepare_fast_dataset(orig_dir, fast_dir, target_shape=(256, 256)):
    """Extrai frames dos NIfTIs, rotaciona para o padrão clínico e salva como .pt"""
    os.makedirs(fast_dir, exist_ok=True)
    all_files = [f for f in os.listdir(orig_dir) if "half_sequence.nii" in f and "_gt" not in f]
    
    print("Pré-processando, rotacionando e salvando frames no formato .pt...")
    for f in tqdm(all_files):
        patient_id = f.split('_')[0]
        visao = "2CH" if "2CH" in f else "4CH"
        
        img_path = os.path.join(orig_dir, f)
        mask_path = os.path.join(orig_dir, f.replace(".nii", "_gt.nii"))
        
        if not os.path.exists(mask_path):
            continue
            
        img_vol = nib.load(img_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()
        num_frames = img_vol.shape[-1]
        
        for i in range(num_frames):
            img_frame = img_vol[:, :, i]
            mask_frame = mask_vol[:, :, i]
            
            # 1. Resize
            img_res = cv2.resize(img_frame, target_shape, interpolation=cv2.INTER_LINEAR)
            mask_res = cv2.resize(mask_frame, target_shape, interpolation=cv2.INTER_NEAREST)
            
            # 2. ROTAÇÃO PARA O PADRÃO CLÍNICO (Cone para cima)
            # Como a ponta estava na esquerda, giramos 90 graus no sentido horário
            img_res = cv2.rotate(img_res, cv2.ROTATE_90_CLOCKWISE)
            mask_res = cv2.rotate(mask_res, cv2.ROTATE_90_CLOCKWISE)
            
            # 3. Normalização (0-1)
            img_res = (img_res - np.min(img_res)) / (np.max(img_res) - np.min(img_res) + 1e-8)
            target = (mask_res == 1).astype(np.float32)
            
            # 4. Converte para Tensor (C, H, W)
            img_tensor = torch.from_numpy(img_res).float().unsqueeze(0)
            target_tensor = torch.from_numpy(target).float().unsqueeze(0)
            
            # Salva no disco
            save_name = f"{patient_id}_{visao}_frame{i}.pt"
            torch.save({'img': img_tensor, 'mask': target_tensor}, os.path.join(fast_dir, save_name))


# Execute isso logo após copiar os dados do Drive
DATASET_ORIGINAL = "C:/Users/Usuario/Documents/Mestrado/dataset/treino_validacao"
DATASET_FAST = "C:/Users/Usuario/Documents/Mestrado/dataset/treino_validacao_fast"
prepare_fast_dataset(DATASET_ORIGINAL, DATASET_FAST)