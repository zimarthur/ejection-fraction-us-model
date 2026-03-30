
import os
import time
import torch
import matplotlib.pyplot as plt

from unet import UNet

# =========================
# CONFIGURAÇÕES
# =========================
MODEL_PATH = "unet.pth"
TEST_FOLDER = "C:/Users/Usuario/Documents/Mestrado/dataset/teste/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5

# =========================
# CARREGA O MODELO
# =========================
model = UNet(in_channels=1, num_classes=1).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# Lista todos os arquivos .pt
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

    img = sample['img']          # [1, H, W]
    mask = sample['mask']        # [1, H, W]

    img_np = img.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    input_tensor = img.unsqueeze(0).to(DEVICE)  # [1,1,H,W]

    # Mede tempo de inferência
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

        # Necessário para medir corretamente na GPU
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    inference_time = (time.perf_counter() - start_time) * 1000

    pred_mask = output.squeeze().cpu().numpy()
    pred_mask_binary = (pred_mask > THRESHOLD).astype(float)

    print(f"{file_name} -> {inference_time:.2f} ms")

    # =========================
    # OVERLAYS
    # =========================
    plt.figure(figsize=(18, 8))

    # Primeira linha
    plt.subplot(2, 3, 1)
    plt.title("Imagem Original")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Máscara Anotada")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title(f"Predição {inference_time:.1f} ms")
    plt.imshow(pred_mask_binary, cmap="gray")
    plt.axis("off")

    # Segunda linha
    plt.subplot(2, 3, 4)
    plt.title("Overlay Máscara Anotada")
    plt.imshow(img_np, cmap="gray")
    plt.imshow(mask_np, cmap="jet", alpha=0.35)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Overlay Predição")
    plt.imshow(img_np, cmap="gray")
    plt.imshow(pred_mask_binary, cmap="jet", alpha=0.35)
    plt.axis("off")

    # Sexto espaço vazio para manter alinhamento
    plt.subplot(2, 3, 6)
    plt.axis("off")

    plt.suptitle(file_name)
    plt.tight_layout()
    # Espera a janela ser fechada antes de continuar para a próxima imagem
    plt.show(block=True)

