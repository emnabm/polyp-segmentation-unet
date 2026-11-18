"""Démo Gradio : segmentation de polypes (U-Net + encodeur ResNet34)."""
import numpy as np
import cv2
import torch
import gradio as gr
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = "weights/resnet34_bcedice_best.pth"   # chemin vers tes poids (voir README)

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE).eval()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict(image, threshold):
    h, w = image.shape[:2]
    x = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    red = np.array([255, 0, 0])
    overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * red).astype(np.uint8)
    return overlay, mask * 255


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Image d'endoscopie"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Seuil de décision"),
    ],
    outputs=[gr.Image(label="Superposition (polype en rouge)"), gr.Image(label="Masque prédit")],
    title="Segmentation de polypes colorectaux (U-Net)",
    description="Charge une image d'endoscopie : le modèle colorie la zone du polype.",
)

if __name__ == "__main__":
    demo.launch()
