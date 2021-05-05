# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pickle
import sys
import tarfile
import zipfile

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import click
from click.core import Option
import numpy as np
import PIL.Image
from tqdm import tqdm
import pandas as pd



def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

def open_image_folder(source_dir, *, max_images: Optional[int]) -> Tuple[List[str],Optional[List[int]]]:
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    labels = [os.path.dirname(os.path.relpath(f, source_dir)) for f in input_images]
    labels = pd.Categorical(labels).codes #convert to integers
    assert(len(labels)==len(input_images))
    dat = pd.DataFrame({"img": input_images, "labels": labels})
    

    p = np.random.permutation(len(input_images))
    u_labels = np.unique(labels)
    if len(u_labels) > 1 and max_images is not None:
        #sample per class
        dat = dat.groupby("labels").apply(lambda x: x.sample(n=min(max_images, len(x)))).reset_index(drop=True)
    elif max_images is not None:
        #sample over all
        dat = dat.loc[p]
    
    

    return dat["img"].to_numpy().tolist(), dat["labels"].to_numpy().tolist() if len(u_labels) > 1 else None


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is empty.
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images (per class for conditional)', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
@click.option('--channels', help='Output height', type=int, default=3)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    width: Optional[int],
    height: Optional[int],
    channels: int
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                - Load LSUN dataset
    --source cifar-10-python.tar.gz - Load CIFAR-10 dataset
    --source path/                  - Recursively load all images from path/
    --source dataset.zip            - Recursively load all images from dataset.zip

    The output dataset format can be either an image folder or a zip archive.  Specifying
    the output format and path:

    \b
    --dest /path/to/dir             - Save output files under /path/to/dir
    --dest /path/to/dataset.zip     - Save output files into /path/to/dataset.zip archive

    Images within the dataset archive will be stored as uncompressed PNG.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    images,labels_raw = open_image_folder(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    labels = []

    for idx in tqdm(range(len(images))):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = PIL.Image.open(images[idx])
        #resize
        if width is not None and height is not None:
            img = img.resize((width,height),PIL.Image.LANCZOS)
        if channels == 3:
            img = PIL.Image.fromarray(np.asarray(img)[:,:,:3],"RGB")
        elif channels == 4:
            img = PIL.Image.fromarray(np.asarray(img)[:,:,:4],"RGBA")

        # Save the image as an uncompressed PNG.
        # img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        if labels_raw is not None:
            labels.append([archive_fname, labels_raw[idx]])
        #print(image['label'])

    metadata = {
        'labels': labels if len(labels)>0 else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
