from PIL import Image
import glob
import os

# Caminho para suas imagens originais
image_paths = glob.glob("C:/Users/Usuario/Documents/Mestrado/dataset/teste_png/*.png")

for path in image_paths:
    # Abre a imagem, converte para 1 canal (Grayscale) e redimensiona para 256x256
    img = Image.open(path).convert("L").resize((256, 256))
    
    # Salva substituindo ou em nova pasta
    filename = os.path.basename(path)
    img.save(f"C:\\Users\\Usuario\\Documents\\Mestrado\\dataset\\teste_png_1d\\{filename}")