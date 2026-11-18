# Segmentation de polypes colorectaux avec U-Net

Projet de Vision par ordinateur avancée, 3e année cycle ingénieur en informatique, ISSAT Sousse.

## Problème
Entrée : image RGB d'endoscopie (256 × 256 × 3). Sortie : masque binaire (256 × 256 × 1) où 1 = polype.

## Dataset
Kvasir-SEG ([lien de téléchargement]) · licence : [à compléter] · ≈ 1000 images · découpage 80/10/10 (graine 42).

## Modèles
| Modèle | Encodeur | Perte | IoU test | Dice test |
|---|---|---|---|---|
| Baseline | U-Net (32 canaux), poids aléatoires | BCE | [à compléter] | [à compléter] |
| Amélioré | ResNet34 pré-entraîné (ImageNet) | BCE + Dice | [à compléter] | [à compléter] |

## Structure du dépôt
- `notebooks/` : préparation des données, entraînement, comparaison
- `src/unet.py` : modèle U-Net écrit pour ce projet
- `app.py` : démo Gradio
- `results/` : courbes, prédictions, schéma d'architecture

## Reproduire
1. Télécharger Kvasir-SEG et le placer dans `data/`.
2. `pip install -r requirements.txt`
3. Exécuter les notebooks dans l'ordre (GPU recommandé, testé sur Google Colab, Tesla T4, PyTorch 2.14.1).
4. Poids entraînés : [lien Google Drive ou Hugging Face]. Les placer dans `weights/`.
5. Lancer la démo : `python app.py`

## Environnement
Python [version] · PyTorch 2.14.1+cu130 · Albumentations [version] · OpenCV [version] · segmentation-models-pytorch [version]

## Références
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015.
- Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, 2020.
- [autres références à ajouter]

## Code externe et code personnel
- Écrit par moi : `src/unet.py`, préparation des données, boucle d'entraînement, métriques, démo.
- Externe : `segmentation-models-pytorch` (encodeur ResNet34), Albumentations, Gradio.
