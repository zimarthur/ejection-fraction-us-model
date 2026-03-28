import os

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

    full_dataset = CamusSequenceDataset(data_dir=DATASET_PATH)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs("results", exist_ok=True)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for idx, (img, target) in enumerate(pbar):
            img = img.to(device)
            target = target.to(device)

            # Forward
            y_pred = model(img)
            loss = criterion(y_pred, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        train_loss = train_running_loss / len(train_dataloader)

        # 6. Validação
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for img, target in val_dataloader:
                img = img.to(device)
                target = target.to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, target)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_dataloader)

        print(f"\nSummary Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("-" * 30)

        # Salvar melhor modelo (opcional: adicionar lógica de best_val_loss)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Treinamento concluído!")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
