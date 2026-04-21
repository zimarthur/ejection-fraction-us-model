
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet

# =====================================================
# CONFIGURAÇÕES
# =====================================================

MODEL_PATH = "models/unet_30_03.pth"
BASE_FILTERS = 64

TEST_FILE = r"C:/Users/Usuario/Documents/Mestrado/dataset/teste/patient0451_2CH_frame10.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
N_DISKS = 20
PIXEL_SPACING_MM = 1.0   # ajustar se souber o espaçamento real

# =====================================================
# CARREGA MODELO
# =====================================================

model = UNet(
    in_channels=1,
    num_classes=1,
    base_filters=BASE_FILTERS
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# =====================================================
# CARREGA AMOSTRA
# =====================================================

sample = torch.load(TEST_FILE, map_location="cpu")

img = sample["img"]          # [1, H, W]
gt_mask = sample["mask"]     # opcional, apenas para comparação

img_np = img.squeeze().numpy()

input_tensor = img.unsqueeze(0).to(DEVICE)

# =====================================================
# INFERÊNCIA
# =====================================================

with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)

pred_mask = output.squeeze().cpu().numpy()
pred_mask = (pred_mask > THRESHOLD).astype(np.uint8)

# =====================================================
# PEGA O MAIOR CONTORNO
# =====================================================

contours, _ = cv2.findContours(
    pred_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    raise ValueError("Nenhuma máscara encontrada")

contour = max(contours, key=cv2.contourArea)

# =====================================================
# ENCONTRA O EIXO MAIOR
# =====================================================

# Ajusta uma elipse para aproximar o eixo principal
if len(contour) < 5:
    raise ValueError("Contorno muito pequeno para ajuste de elipse")

ellipse = cv2.fitEllipse(contour)
(center_x, center_y), (axis1, axis2), angle_deg = ellipse

major_axis = max(axis1, axis2)
minor_axis = min(axis1, axis2)

angle_rad = np.deg2rad(angle_deg)

# Vetor do eixo principal
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)

# Corrige se a elipse retornou o eixo menor como principal
if axis2 > axis1:
    vx = np.cos(angle_rad + np.pi / 2)
    vy = np.sin(angle_rad + np.pi / 2)

# Pontos das extremidades do eixo maior
x1 = center_x - vx * major_axis / 2
y1 = center_y - vy * major_axis / 2

x2 = center_x + vx * major_axis / 2
y2 = center_y + vy * major_axis / 2

# =====================================================
# DIVIDE EM DISCOS
# =====================================================

disk_centers = []
disk_diameters = []

# Vetor perpendicular ao eixo principal
px = -vy
py = vx

for i in range(N_DISKS):
    t = (i + 0.5) / N_DISKS

    cx = x1 + (x2 - x1) * t
    cy = y1 + (y2 - y1) * t

    disk_centers.append((cx, cy))

    # Amostra ao longo da direção perpendicular
    distances = []

    for s in np.linspace(-major_axis, major_axis, 500):
        xx = int(round(cx + px * s))
        yy = int(round(cy + py * s))

        if (
            0 <= xx < pred_mask.shape[1]
            and 0 <= yy < pred_mask.shape[0]
        ):
            if pred_mask[yy, xx] > 0:
                distances.append(s)

    if len(distances) == 0:
        diameter = 0
    else:
        diameter = max(distances) - min(distances)

    disk_diameters.append(diameter)

# =====================================================
# CALCULA VOLUME PELO MÉTODO DE SIMPSON
# =====================================================

# Altura de cada disco
h = major_axis / N_DISKS

volume_pixels = 0.0

for diameter in disk_diameters:
    radius = diameter / 2
    disk_volume = np.pi * radius**2 * h
    volume_pixels += disk_volume

# Converte para mm³ caso pixel spacing seja conhecido
volume_mm3 = volume_pixels * (PIXEL_SPACING_MM ** 3)
volume_ml = volume_mm3 / 1000.0

print(f"Volume estimado: {volume_ml:.2f} mL")

# =====================================================
# VISUALIZAÇÃO
# =====================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(img_np, cmap="gray")
ax.imshow(pred_mask, cmap="jet", alpha=0.35)

# Desenha eixo principal
ax.plot([x1, x2], [y1, y2], color="yellow", linewidth=2)

# Desenha discos
for (cx, cy), diameter in zip(disk_centers, disk_diameters):
    r = diameter / 2

    # direção perpendicular
    dx = px * r
    dy = py * r

    ax.plot(
        [cx - dx, cx + dx],
        [cy - dy, cy + dy],
        color="cyan",
        linewidth=2
    )

    ax.scatter(cx, cy, color="red", s=10)

ax.set_title(
    f"Simpson Biplane - {N_DISKS} discos\n"
    f"Volume estimado: {volume_ml:.2f} mL"
)
ax.axis("off")

plt.tight_layout()
plt.show()
