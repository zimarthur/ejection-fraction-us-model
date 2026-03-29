import torch
import matplotlib.pyplot as plt
import os

FAST_DIR = "C:/Users/Usuario/Documents/Mestrado/dataset/treino_validacao_fast" 

# Ordena para garantir que vamos ver os frames na sequência do batimento cardíaco
all_pt_files = sorted(os.listdir(FAST_DIR))

# 1. Ativa o modo interativo do Matplotlib
plt.ion() 

# 2. Cria a figura e os eixos uma única vez fora do loop
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Inicializa imagens vazias apenas para criar os objetos no plot
im_img = axes[0].imshow(torch.zeros(256, 256), cmap='gray', vmin=0, vmax=1)
im_mask = axes[1].imshow(torch.zeros(256, 256), cmap='gray', vmin=0, vmax=1)

axes[0].axis('off')
axes[1].axis('off')
axes[1].set_title("Máscara do Ventrículo Esquerdo")
plt.tight_layout()

print("Iniciando visualização em loop...")

# Itera sobre os arquivos (limitei a 100 para teste, remova o limite se quiser ver tudo)
for file_name in all_pt_files[:100]:
    pt_path = os.path.join(FAST_DIR, file_name)
    
    # Carrega os tensores
    data = torch.load(pt_path, weights_only=True)
    img = data['img'].squeeze().numpy()
    mask = data['mask'].squeeze().numpy()
    
    # 3. Atualiza os pixels das imagens que já estão na tela
    im_img.set_data(img)
    im_mask.set_data(mask)
    
    # Atualiza o título para o frame atual
    axes[0].set_title(f"Ultrassom (Rotacionado)\nArquivo: {file_name}")
    
    # 4. Desenha a nova imagem e pausa a execução por 0.5 segundos
    plt.pause(0.5)

# Desativa o modo interativo no final e mantém a última imagem aberta
plt.ioff()
print("Visualização concluída!")
plt.show()