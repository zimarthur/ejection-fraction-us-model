import os
import torch
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

from unet import UNet


def preprocess_frame(frame, target_shape=(256,256)):

    frame = cv2.resize(frame, target_shape)

    frame = (frame - frame.min())/(frame.max()-frame.min()+1e-8)

    tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)

    return tensor


def run_inference_with_gt(nii_path, model, device):

    gt_path = nii_path.replace(".nii","_gt.nii")

    volume = nib.load(nii_path).get_fdata()
    gt_volume = nib.load(gt_path).get_fdata()

    num_frames = volume.shape[-1]

    is_2ch = "2CH" in nii_path

    model.eval()

    for i in range(num_frames):

        frame = volume[:,:,i]
        gt_frame = gt_volume[:,:,i]

        img_tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():

            pred_logits = model(img_tensor)
            pred_probs = torch.sigmoid(pred_logits)
            pred_mask = (pred_probs > 0.5).float()

        pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

        if is_2ch:
            pred_lv = pred_mask_np[0]
        else:
            pred_lv = pred_mask_np[1]

        gt_lv = (gt_frame == 1).astype(np.float32)

        frame_res = cv2.resize(frame,(256,256))
        gt_lv = cv2.resize(gt_lv,(256,256),interpolation=cv2.INTER_NEAREST)

        fig,ax = plt.subplots(1,4,figsize=(16,4))

        ax[0].imshow(frame_res,cmap="gray")
        ax[0].set_title("Frame")

        ax[1].imshow(gt_lv,cmap="gray")
        ax[1].set_title("GT")

        ax[2].imshow(pred_lv,cmap="gray")
        ax[2].set_title("Prediction")

        overlay = np.stack([frame_res]*3,-1)

        overlay[:,:,1] += gt_lv*0.5
        overlay[:,:,2] += pred_lv*0.5

        ax[3].imshow(overlay)
        ax[3].set_title("GT (green) vs Pred (blue)")

        for a in ax:
            a.axis("off")

        plt.show()