import os
import cv2
import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load LLaVA-Pythia tokenizer & model (change path or model name as needed)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModel.from_pretrained("EleutherAI/pythia-410m")
model.eval()

# Function to embed text using Pythia
def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        # Use CLS token's hidden state from last layer
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

# Paths
output_dir = "hucenrotia_TM5_700_screwing_psu_dataset"
os.makedirs(output_dir, exist_ok=True)
h5_path = os.path.join(output_dir, 'episode_0.h5')

# Image setup
H, W = 240, 320
num_frames = 100
image_path = "image.jpg"
wrist_image_path = "wrist_image.jpg"
img = cv2.resize(cv2.imread(image_path), (W, H))
wrist_img = cv2.resize(cv2.imread(wrist_image_path), (W, H))

# Instruction
instruction = "tighten the PSU screw"

# Init lists
list_image = []
list_wrist_image = []
list_action = []
list_is_edited = []
list_qpos = []
list_qvel = []
list_language_embedding = []

# Fill with data
for _ in range(num_frames):
    list_image.append(img)
    list_wrist_image.append(wrist_img)
    list_action.append(np.random.randn(7).astype(np.float32))
    list_is_edited.append(0)
    list_qpos.append(np.random.randn(7).astype(np.float32))
    list_qvel.append(np.random.randn(7).astype(np.float32))
    list_language_embedding.append(embed_text(instruction))  

# Stack arrays
image_sequence = np.stack(list_image, axis=0)
wrist_image_sequence = np.stack(list_wrist_image, axis=0)
list_action = np.stack(list_action, axis=0)
list_is_edited = np.stack(list_is_edited, axis=0)
list_qpos = np.stack(list_qpos, axis=0)
list_qvel = np.stack(list_qvel, axis=0)
list_language_embedding = np.stack(list_language_embedding, axis=0)

# Save to HDF5
with h5py.File(h5_path, 'w') as hf:
    observations_group = hf.create_group('observations')
    images_group = observations_group.create_group('images')

    images_group.create_dataset('image', data=image_sequence)
    images_group.create_dataset('wrist_image', data=wrist_image_sequence)
    hf.create_dataset('action', data=list_action)
    hf.create_dataset('is_edited', data=list_is_edited)
    hf.create_dataset('language_embedding', data=list_language_embedding)
    hf.create_dataset('language_raw', data=np.string_(instruction))
    observations_group.create_dataset('qpos', data=list_qpos)
    observations_group.create_dataset('qvel', data=list_qvel)

print(f"Saved dataset to '{h5_path}' with language embeddings.")
