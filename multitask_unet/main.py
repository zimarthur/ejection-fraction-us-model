import os
import random

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from data_loader import CamusSequenceDataset # Certifique-se que o nome do arquivo seja este

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    MODEL_SAVE_PATH = "results/unet.pth"
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --- NOVA LÓGICA DE SPLIT POR PACIENTE ---
    # 1. Identifica todos os arquivos e extrai os IDs únicos dos pacientes
    all_files = [f for f in os.listdir(DATASET_PATH) if "half_sequence.nii" in f and "_gt" not in f]
    unique_patients = list(set([f.split('_')[0] for f in all_files]))
    unique_patients.sort() # Ordena para garantir consistência antes de embaralhar

    # 2. Embaralha a lista de pacientes (usamos uma seed para reprodutibilidade, se quiser)
    random.seed(42) 
    random.shuffle(unique_patients)

    # 3. Divide a lista (ex: 80% treino, 20% validação)
    train_ratio = 0.8
    split_idx = int(len(unique_patients) * train_ratio)
    
    train_patients = unique_patients[:split_idx]
    val_patients = unique_patients[split_idx:]

    print(f"Total de pacientes na pasta: {len(unique_patients)}")
    print(f"Pacientes para Treino: {len(train_patients)} | Validação: {len(val_patients)}")

    # 4. Instancia os datasets passando as listas autorizadas
    train_dataset = CamusSequenceDataset(data_dir=DATASET_PATH, allowed_patients=train_patients)
    val_dataset = CamusSequenceDataset(data_dir=DATASET_PATH, allowed_patients=val_patients)

    # 5. Cria os dataloaders normalmente
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # NOVAS FUNÇÕES DE PERDA
    criterion_seg = nn.BCEWithLogitsLoss() # Para a máscara do ventrículo (1 canal)
    criterion_cls = nn.CrossEntropyLoss()  # Para a classificação das câmaras (2 classes)

    os.makedirs("results", exist_ok=True)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Desempacotando img, mask e label
        for idx, (img, mask, label) in enumerate(pbar):
            img = img.to(device)
            mask = mask.to(device)
            label = label.to(device)

            # Forward (retorna predição espacial e predição de classe)
            pred_mask, pred_label = model(img)
            
            # Calculando as perdas individuais
            loss_seg = criterion_seg(pred_mask, mask)
            loss_cls = criterion_cls(pred_label, label)
            
            # Somando as perdas (você pode adicionar pesos aqui no futuro, ex: loss_seg + 0.5 * loss_cls)
            loss = loss_seg + loss_cls

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            pbar.set_postfix({"loss_total": loss.item(), "loss_seg": loss_seg.item(), "loss_cls": loss_cls.item()})

        train_loss = train_running_loss / len(train_dataloader)

        # Validação
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for img, mask, label in val_dataloader:
                img = img.to(device)
                mask = mask.to(device)
                label = label.to(device)
                
                pred_mask, pred_label = model(img)
                
                loss_seg = criterion_seg(pred_mask, mask)
                loss_cls = criterion_cls(pred_label, label)
                loss = loss_seg + loss_cls
                
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_dataloader)

        print(f"\nSummary Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("-" * 30)

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Treinamento concluído!")
