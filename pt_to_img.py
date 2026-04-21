import torch
import cv2
import numpy as np
import os
import glob

def pt_to_images(pt_filepath, output_dir):
    """Lê um arquivo .pt e salva a imagem e a máscara como .png"""
    # Extrai o nome do arquivo sem a extensão para usar ao salvar
    base_name = os.path.basename(pt_filepath).replace('.pt', '')
    
    # 1. Carrega o arquivo .pt
    data = torch.load(pt_filepath)
    img_tensor = data['img']
    mask_tensor = data['mask']
    
    # 2. Converte de Tensor (1, 256, 256) para Numpy Array (256, 256)
    # O .squeeze(0) remove aquela dimensão extra que foi adicionada com o .unsqueeze(0)
    img_np = img_tensor.squeeze(0).numpy()
    mask_np = mask_tensor.squeeze(0).numpy()
    
    # 3. Desnormalização (de 0.0-1.0 para 0-255)
    img_np = (img_np * 255).astype(np.uint8)
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # OPCIONAL: Se quiser desfazer a rotação clínica para a orientação original do NIfTI
    # img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # mask_np = cv2.rotate(mask_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 4. Salva no disco
    img_save_path = os.path.join(output_dir, f"{base_name}_img.png")
    mask_save_path = os.path.join(output_dir, f"{base_name}_mask.png")
    
    cv2.imwrite(img_save_path, img_np)
    cv2.imwrite(mask_save_path, mask_np)
    
    print(f"Salvo: {base_name}_img.png e {base_name}_mask.png")

def converter_pasta(input_dir, output_dir):
    """Lê todos os arquivos .pt de uma pasta e converte para .png"""
    # Cria a pasta de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Busca todos os arquivos que terminam com .pt na pasta de entrada
    caminho_busca = os.path.join(input_dir, "*.pt")
    arquivos_pt = glob.glob(caminho_busca)
    
    if not arquivos_pt:
        print(f"Nenhum arquivo .pt encontrado na pasta: {input_dir}")
        return
        
    print(f"Encontrados {len(arquivos_pt)} arquivos .pt. Iniciando conversão...")
    
    # Itera sobre a lista de arquivos e converte um por um
    for filepath in arquivos_pt:
        pt_to_images(filepath, output_dir)
        
    print("-" * 30)
    print("Conversão de todos os arquivos concluída com sucesso!")

# --- Exemplo de Uso ---
INPUT_DIR = "C:/Users/Usuario/Documents/Mestrado/dataset/teste"
OUTPUT_DIR = "C:/Users/Usuario/Documents/Mestrado/dataset/teste_png"

# Chama a nova função passando a pasta de entrada e a de saída
converter_pasta(INPUT_DIR, OUTPUT_DIR)