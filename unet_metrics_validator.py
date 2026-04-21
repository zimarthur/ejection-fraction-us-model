import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import dc, hd95

# =========================
# CONFIGURAÇÕES
# =========================
TEST_DIR = r"C:\Users\Usuario\Documents\Mestrado\dataset\teste"
MODEL_PATH = r"unet_01_04.pth"
OUTPUT_CSV = "metricas_unet_01_04.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5

# =========================
# CARREGA O MODELO
# =========================
from unet import UNet

model = UNet(in_channels=1, num_classes=1, base_filters= 8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# LISTA DOS ARQUIVOS
# =========================
all_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.endswith(".pt")
])

print(f"Encontrados {len(all_files)} arquivos de teste")

# Guarda métricas por frame
rows = []

# =========================
# LOOP DE AVALIAÇÃO
# =========================
with torch.no_grad():
    for file_name in tqdm(all_files):
        data = torch.load(os.path.join(TEST_DIR, file_name))

        img = data["img"].unsqueeze(0).to(DEVICE)   # [1, 1, H, W]
        gt = data["mask"].squeeze().cpu().numpy().astype(bool)

        # Nome do paciente = parte antes do primeiro "_"
        patient_id = file_name.split("_")[0]

        # Inferência
        pred = model(img)

        # Se a rede retorna logits
        pred = torch.sigmoid(pred)

        # Threshold
        pred = (pred > THRESHOLD).float()

        pred = pred.squeeze().cpu().numpy().astype(bool)

        # Dice
        dice_value = dc(pred, gt)

        # HD95
        if pred.sum() == 0 and gt.sum() == 0:
            hd95_value = 0.0
        elif pred.sum() == 0 or gt.sum() == 0:
            hd95_value = np.nan
        else:
            hd95_value = hd95(pred, gt)

        rows.append({
            "patient_id": patient_id,
            "file": file_name,
            "dice": dice_value,
            "hd95": hd95_value
        })

# =========================
# DATAFRAME POR FRAME
# =========================
df_frames = pd.DataFrame(rows)

# =========================
# MÉDIA POR PACIENTE
# =========================
df_patients = (
    df_frames
    .groupby("patient_id")
    .agg(
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        hd95_mean=("hd95", "mean"),
        hd95_std=("hd95", "std"),
        num_frames=("file", "count")
    )
    .reset_index()
)

# Se paciente tiver apenas 1 frame, std vira NaN; opcionalmente substitui por 0
df_patients = df_patients.fillna(0)

# =========================
# MÉTRICAS GLOBAIS
# =========================
final_dice_mean = df_frames["dice"].mean()
final_dice_std = df_frames["dice"].std()

final_hd95_mean = df_frames["hd95"].mean(skipna=True)
final_hd95_std = df_frames["hd95"].std(skipna=True)

# Adiciona uma linha final ao dataframe
final_row = pd.DataFrame([{
    "patient_id": "FINAL",
    "dice_mean": final_dice_mean,
    "dice_std": final_dice_std,
    "hd95_mean": final_hd95_mean,
    "hd95_std": final_hd95_std,
    "num_frames": len(df_frames)
}])

df_output = pd.concat([df_patients, final_row], ignore_index=True)

# =========================
# SALVA CSV
# =========================
df_output.to_csv(OUTPUT_CSV, index=False)

print("\n===== RESULTADOS POR PACIENTE =====")
print(df_output)

print(f"\nCSV salvo em: {OUTPUT_CSV}")