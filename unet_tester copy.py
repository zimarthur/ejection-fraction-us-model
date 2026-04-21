import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from medpy.metric.binary import dc, hd95

from unet import UNet

# =========================
# CONFIGURAÇÕES
# =========================

MODEL_CONFIGS = [
    {
        "name": "UNet 64",
        "path": "models/unet_30_03.pth",
        "base_filters": 64,
    },
    {
        "name": "UNet 32",
        "path": "models/unet_29_03.pth",
        "base_filters": 32,
    },
    {
        "name": "UNet 16",
        "path": "models/unet_31_03.pth",
        "base_filters": 16,
    },
    {
        "name": "UNet 8",
        "path": "models/unet_01_04.pth",
        "base_filters": 8,
    },
]

TEST_FOLDER = r"C:/Users/Usuario/Documents/Mestrado/dataset/teste/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5

# =========================
# CARREGA TODOS OS MODELOS
# =========================

models = []

for config in MODEL_CONFIGS:
    model = UNet(
        in_channels=1,
        num_classes=1,
        base_filters=config["base_filters"]
    ).to(DEVICE)

    state_dict = torch.load(config["path"], map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    models.append({
        "name": config["name"],
        "model": model
    })

print(f"{len(models)} modelos carregados")

# =========================
# LISTA TODOS OS ARQUIVOS
# =========================

all_files = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.endswith(".pt")
])

print(f"Encontrados {len(all_files)} arquivos para testar")

# =========================
# LOOP DE TESTE
# =========================

for file_name in all_files:
    file_path = os.path.join(TEST_FOLDER, file_name)

    sample = torch.load(file_path, map_location="cpu")

    img = sample["img"]      # [1, H, W]
    gt_mask = sample["mask"] # [1, H, W]

    img_np = img.squeeze().numpy()
    gt_mask_np = gt_mask.squeeze().numpy()

    gt_bool = gt_mask_np.astype(bool)

    input_tensor = img.unsqueeze(0).to(DEVICE)  # [1,1,H,W]

    predictions = []

    # =========================
    # INFERÊNCIA DE TODOS MODELOS
    # =========================

    for model_info in models:
        model = model_info["model"]
        model_name = model_info["name"]

        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

        inference_time = (time.perf_counter() - start_time) * 1000

        pred_mask = output.squeeze().cpu().numpy()
        pred_mask_binary = (pred_mask > THRESHOLD).astype(np.uint8)

        pred_bool = pred_mask_binary.astype(bool)

        # =========================
        # MÉTRICAS
        # =========================

        dice_value = dc(pred_bool, gt_bool)

        if pred_bool.sum() == 0 and gt_bool.sum() == 0:
            hd95_value = 0.0
        elif pred_bool.sum() == 0 or gt_bool.sum() == 0:
            hd95_value = np.nan
        else:
            hd95_value = hd95(pred_bool, gt_bool)

        predictions.append({
            "name": model_name,
            "mask": pred_mask_binary,
            "time": inference_time,
            "dice": dice_value,
            "hd95": hd95_value
        })

        print(
            f"{file_name} | {model_name} -> "
            f"{inference_time:.2f} ms | "
            f"Dice: {dice_value:.4f} | "
            f"HD95: {hd95_value:.4f}"
        )

    # =========================
    # PLOT
    # =========================

    total_plots = 1 + len(predictions)

    fig, axes = plt.subplots(
        1,
        total_plots,
        figsize=(6 * total_plots, 6)
    )

    if total_plots == 1:
        axes = [axes]

    # Ground Truth
    axes[0].set_title("Ground Truth")
    axes[0].imshow(img_np, cmap="gray")
    axes[0].imshow(gt_mask_np, cmap="jet", alpha=0.35)
    axes[0].axis("off")

    # Predições dos modelos
    for idx, pred in enumerate(predictions, start=1):
        hd95_text = (
            f"{pred['hd95']:.2f}"
            if not np.isnan(pred["hd95"])
            else "NaN"
        )

        title = (
            f"{pred['name']}\n"
            f"{pred['time']:.1f} ms\n"
            f"Dice: {pred['dice']:.3f}\n"
            f"HD95: {hd95_text}"
        )

        axes[idx].set_title(title, fontsize=10)

        axes[idx].imshow(img_np, cmap="gray")
        axes[idx].imshow(pred["mask"], cmap="jet", alpha=0.35)
        axes[idx].axis("off")

    plt.suptitle(file_name, fontsize=16)
    plt.tight_layout()

    # Espera fechar antes de continuar
    plt.show(block=True)