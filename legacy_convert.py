# %% legacy convert
from legacy import load_network_pkl
import lzma
import torch
from PIL import Image
import pickle

# %%
with open('pretrained/legacy/wikiart-uncond.pkl', 'rb') as f:
    temp = load_network_pkl(f)
with open('pretrained/wikiart-uncond.pkl', 'wb') as f:
    pickle.dump(temp, f)
# %% load pretrained model
with open('pretrained/wikiart-uncond.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
                         # NCHW, float32, dynamic range [-1, +1]
# %% sample an image
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
noise_mode = ['const', 'random', 'none'][1]
img = G(z, c,truncation_psi=0.9,noise_mode=noise_mode)  
#convert and render it
img_dat = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
Image.fromarray(img_dat.cpu().numpy())
# %%
