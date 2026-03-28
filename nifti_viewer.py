import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def animar_sequencia(paciente_id, visao="2CH"):
    base_path = os.path.join('..', 'dataset')
    
    # Caminhos dos arquivos de sequência e labels
    path_img = os.path.join(base_path, f"{paciente_id}_{visao}_half_sequence.nii")
    path_gt = os.path.join(base_path, f"{paciente_id}_{visao}_half_sequence_gt.nii")

    if not os.path.exists(path_img) or not os.path.exists(path_gt):
        print("Arquivos de sequência não encontrados.")
        return

    # Carregar dados (H, W, Frames)
    img_seq = nib.load(path_img).get_fdata()
    gt_seq = nib.load(path_gt).get_fdata()
    
    num_frames = img_seq.shape[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Inicialização dos plots
    im1 = ax1.imshow(img_seq[:, :, 0], cmap='gray')
    ax1.set_title("Ultrassom (Sequência)")
    
    # Exibindo apenas o Ventrículo Esquerdo (Classe 1) na máscara
    im2 = ax2.imshow(np.where(gt_seq[:, :, 0] == 1, 1, 0), cmap='Blues')
    ax2.set_title("Máscara Ground Truth (Ventrículo)")

    def update(frame):
        # Atualiza a imagem de ultrassom
        im1.set_data(img_seq[:, :, frame])
        
        # Atualiza a máscara (focando na classe 1)
        mask_frame = np.where(gt_seq[:, :, frame] == 1, 1, 0)
        im2.set_data(mask_frame)
        
        fig.suptitle(f"Frame {frame+1}/{num_frames} - {paciente_id} {visao}")
        return im1, im2

    # Criar a animação
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    plt.tight_layout()
    plt.show()

# Executar a animação
animar_sequencia('patient0001', visao='2CH')