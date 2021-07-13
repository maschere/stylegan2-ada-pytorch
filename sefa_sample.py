# %% imports
from typing import List
import numpy as np
from PIL import Image
import torch
import pickle
import lzma
import re
import itertools
import textwrap, os
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets
from IPython.display import display
import io

from legacy import load_network_pkl

def display_images(
    images: List[Image.Image], 
    columns:int=None):
    width=30
    height=21
    label_wrap_length=50
    label_font_size=16
    if columns is None:
        columns = int(np.sqrt(len(images)))+1

    if not images:
        print("No images to display.")
        return 

    #height = max(height, int(len(images)/columns) * height)
    fig = plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        p = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        axes = plt.imshow(image)
        p.get_xaxis().set_visible(False)
        p.get_yaxis().set_visible(False)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 
    #fig.tight_layout()

# %%
# # %% load pretrained model
# with open('pretrained/maps.pkl', 'rb') as f:
#     G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
#                          # NCHW, float32, dynamic range [-1, +1]
# # %% sample an image
# z = torch.randn([1, G.z_dim]).cuda()    # latent codes
# c = None                                # class labels (not used in this example)
# noise_mode = ['const', 'random', 'none'][1]
# img = G(z, c,truncation_psi=0.9,noise_mode=noise_mode)  
# #convert and render it
# img_dat = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
# Image.fromarray(img_dat.cpu().numpy())
# # %% legacy convert
# with lzma.open('pretrained/legacy/anime.pkl.xz', 'rb') as f:
#     temp = load_network_pkl(f)
# # %% 
# with open('pretrained/anime.pkl', 'wb') as f:
#     pickle.dump(temp, f)
# # %%

# import os
# import re
# # load images
# p = "data/mtga/"
# for img_f in map(lambda x: re.match("^.*\d_AIF\.png$", x), os.listdir(p)):
#     if img_f is None:
#         continue
#     img_a:np.ndarray = np.asfarray(Image.open(p + img_f.group(0)))
#     if img_a.shape[0] != img_a.shape[1] or img_a.shape[2]<3:
#         continue
#     img = Image.fromarray(img_a[:,:,0:3].astype("uint8"))
#     if img_a.shape[0] == 1024:
#         img = img.resize((512,512))
#     img.save(p.replace("mtga/","mtga/processed/") + img_f.group(0))

# # %%
# # load images
# import glob
# p = "data/pixelchar/"
# for img_f in glob.iglob(p + '**/*.png', recursive=True):
#     img_a:np.ndarray = np.asfarray(Image.open(img_f))
#     if img_a.shape[0] != img_a.shape[1] or img_a.shape[2]<3:
#         continue
#     img = Image.fromarray(img_a[:,:,0:3].astype("uint8"))
#     img = img.resize((256,256),Image.NEAREST)
#     img.save(img_f)
# # %%
# #python train.py --outdir=training-runs --data=data/pixelchar.zip --cfg=paper256 --mirror=1 --resume=pretrained/ffhq-res256-mirror-paper256-noaug.pkl --snap=10
#python dataset_tool2.py --source data/pixelchar/orig --dest pixelchar.zip --width=64 --height=64
#python train.py --outdir=training-runs --data=pixelchar.zip --cfg=auto --mirror=1 --snap=10 --nobench=True --workers=2 --cond=1 --kimg=10000
# import pickle
# with open('pretrained/maps.pkl', 'rb') as f:
#     data = pickle.load(f)
# # %%
# with open('pretrained/maps.pkl', 'rb') as f:
#     gen = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
#                         # NCHW, float32, dynamic range [-1, +1]
# def generate(G, seed:int = None, trunc = 0.9, noise_mode = ['const', 'random', 'none'][1],c = None):
#     # %% sample an image
#     if seed is None:
#         z = torch.randn([1, G.z_dim]).cuda()    # latent codes
#     else:
#         z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).cuda()
#     # class labels (not used in this example)
#     img = G(z, c,truncation_psi=trunc,noise_mode=noise_mode)  
#     #convert and render it
#     img_dat = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
#     return Image.fromarray(img_dat.cpu().numpy())

# generate(gen,1)
# %%
def sefa_sample(generator,num_sam = 5,num_sem = 5, distances = np.linspace(-3.0,3.0, 10), sel_layers = list(range(18)), rand_seed=124):
    #get all style weights
    weights = []
    layers = []
    i = 0
    for w in generator.synthesis.named_parameters():
        if re.search("conv\d\.affine\.weight$", w[0]) is not None:
            print(w[0])
            #print(w[1].T.shape)
            weights.append(w[1].T.cpu().detach().numpy())
            layers.append(i)
            i += 1
    weights.append(list(generator.synthesis.named_modules())[-1][1].weight.T.cpu().detach().numpy())
    i += 1
    weights = [weights[i] for i in sel_layers]
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    #normalize and decompose into eigenvalues
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    boundaries = eigen_vectors.T
    #sample space
    # Set random seed.
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # Prepare codes.
    codes = torch.randn(num_sam, generator.z_dim).cuda()
    codes = generator.mapping(codes,None,truncation_psi=0.7,truncation_cutoff=8)
    codes = codes.detach().cpu().numpy()

    #iterate samples X semantics X distances
    p = itertools.product(range(num_sem),range(num_sam), range(len(distances)))
    imgs=[]
    for sem_id,sam_id,d_id in p:
        code = codes[sam_id:sam_id + 1]
        boundary = boundaries[sem_id:sem_id + 1]
        d = distances[d_id]
        temp_code = code.copy()
        temp_code[:,sel_layers,:] += boundary * d
        temp_code = torch.from_numpy(temp_code).type(torch.FloatTensor).cuda()
        image = generator.synthesis(temp_code)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
        image = Image.fromarray(image.cpu().numpy())
        fname = f"sefa_results/sam{sam_id}_sem{sem_id}_d{d_id}.jpg"
        setattr(image,"filename", fname)
        imgs.append(image)
    return imgs,boundaries,codes
        
# %% tokens004276
#with open('pretrained/dnd_faces1024-snapshot-001028.pkl', 'rb') as f:
with open('pretrained/tokens004276.pkl', 'rb') as f:
    gen = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

# %%
# %%

# %%
def generate(G, seed:int = None, z = None, zhat = None, trunc = 0.9, noise_mode = ['const', 'random', 'none'][1],c = None, boundary = None, boundary_layers = list(range(18))):
    # %% sample an image
    if seed is None and z is None:
        z = torch.randn([1, G.z_dim]).cuda()    # latent codes
    elif seed is not None and z is None:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).cuda()
    # class labels (not used in this example)
    if zhat is None:
        zhat = G.mapping(z,None,truncation_psi=trunc,truncation_cutoff=8)
    else:
        zhat = torch.from_numpy(zhat).type(torch.FloatTensor).cuda()
    #add boundary semantics
    if boundary is not None:
        zhat[:,boundary_layers,:] += boundary
    #img = G(z, c,truncation_psi=trunc,noise_mode=noise_mode)  
    img = G.synthesis(zhat,noise_mode=noise_mode)
    #convert and render it
    img_dat = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
    return Image.fromarray(img_dat.cpu().numpy())
# %%
generate(gen,4)
# %%
np.random.seed(124)
torch.manual_seed(124)
z2 = torch.randn(5, gen.z_dim).cuda()[2:3]

generate(gen, z=z2)
# %%
imgs,boundaries,zhats = sefa_sample(gen, rand_seed=125, num_sam=5, num_sem=5,sel_layers=list(range(16)))
# %%
@interact(semantic=range(5))
def update(semantic=0):
    # img_byte_arr = io.BytesIO()
    # imgs[a].save(img_byte_arr, format='PNG')
    # display(ipywidgets.Image(value=img_byte_arr.getvalue(),format="png",width=imgs[a].width, height=imgs[a].height))
    display_images(imgs[50*semantic:50*semantic+50],10)

# %%
generate(gen, zhat=zhats[2:3],boundary=boundaries[0:1]*1+boundaries[1:2]*-1+boundaries[2:3]*-3,boundary_layers=list(range(16))).save("test.png")
# %%
