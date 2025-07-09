# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Configuration parameters
openslidelevel = 0            # level from openslide to read
tilesize = 10000              # size of the tile to load from openslide
patchsize = 32                # patch size needed by our DL model

minhits = 100                 # minimum objects per tile to process
batchsize = 1024              # number of patches per batch for inference
nclasses = 2                  # number of output classes for the model
classnames = ["negative", "positive"]  # class names for QuPath visualization
colors = [-377282, -9408287]  # corresponding RGB colors for those classes

mask_patches = False          # blackout background around objects if trained that way

# File paths
json_fname = r'MTA38GPC.json'                  # input GeoJSON file
json_annotated_fname = r'1L1_nuclei_reg_anno.json'  # output annotated GeoJSON file
model_fname = "lymph_model.pth"              # path to DL model
wsi_fname = "MTA3-8_GPC3.svs"                 # whole-slide image file

# --- Imports ---
import os
import gzip
import numpy as np
import cv2
import openslide
from math import ceil
import geojson
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
import torch

# --- Setup ---
print("Looking for file:", os.path.abspath(wsi_fname))
print("Exists?", os.path.exists(wsi_fname))
print("Current directory:", os.getcwd())

os.environ['PATH'] = os.path.expanduser('~/Documents/GitHub/QuPath') + os.pathsep + os.environ.get('PATH', '')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def divide_batch(arr, n):
    for i in range(0, arr.shape[0], n):
        yield arr[i:i + n]

# --- Load Model (stub) ---
# model = MyModel().to(device)
# checkpoint = torch.load(model_fname, map_location=device)
# model.load_state_dict(checkpoint['model_dict'])
# model.eval()

# --- Load GeoJSON ---
if json_fname.endswith('.gz'):
    with gzip.GzipFile(json_fname, 'r') as f:
        allobjects = geojson.loads(f.read().decode('ascii'))
else:
    with open(json_fname) as f:
        allobjects = geojson.load(f)
print(f"Loaded {len(allobjects)} objects from {json_fname}")

# Build shapes and centroids
allshapes = [shape(obj.get('nucleusGeometry', obj.get('geometry'))) for obj in allobjects]
allcenters = [s.centroid for s in allshapes]
searchtree = STRtree(allcenters)

# Open WSI
osh = openslide.OpenSlide(wsi_fname)
level_dims = osh.level_dimensions[openslidelevel]
scalefactor = int(osh.level_downsamples[openslidelevel])
paddingsize = (patchsize // 2) * scalefactor
int_coords = lambda coords: np.array(coords).round().astype(np.int32)

# --- Tile iteration (no progress bars) ---
for y in range(0, level_dims[1], round(tilesize * scalefactor)):
    for x in range(0, level_dims[0], round(tilesize * scalefactor)):
        tilepoly = Polygon([
            (x, y),
            (x + tilesize * scalefactor, y),
            (x + tilesize * scalefactor, y + tilesize * scalefactor),
            (x, y + tilesize * scalefactor)
        ])
        hits_idx = [int(h) for h in searchtree.query(tilepoly)]
        if len(hits_idx) < minhits:
            continue

        tile = np.asarray(
            osh.read_region(
                (x - paddingsize, y - paddingsize),
                openslidelevel,
                (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize)
            )
        )[:, :, :3]

        if mask_patches:
            mask = np.zeros(tile.shape[:2], dtype=tile.dtype)
            exteriors = [int_coords(allshapes[i].boundary.coords) for i in hits_idx]
            exteriors_shifted = [
                (ext - np.array([x - paddingsize, y - paddingsize])) // scalefactor
                for ext in exteriors
            ]
            cv2.fillPoly(mask, exteriors_shifted, 1)

        arr_out = np.zeros((len(hits_idx), patchsize, patchsize, 3), dtype=tile.dtype)
        id_out = np.zeros((len(hits_idx), 1), dtype=int)

        for i, idx in enumerate(hits_idx):
            cx, cy = allcenters[idx].x, allcenters[idx].y
            col = int((cx - x + paddingsize) // scalefactor)
            row = int((cy - y + paddingsize) // scalefactor)
            patch = tile[
                row - patchsize//2:row + patchsize//2,
                col - patchsize//2:col + patchsize//2,
                :
            ]
            if mask_patches:
                mp = mask[row - patchsize//2:row + patchsize//2,
                          col - patchsize//2:col + patchsize//2]
                patch = patch * mp[:, :, None]
            arr_out[i] = patch
            id_out[i, 0] = idx

        # Classify patches in batches (no outer progress bars)
        batches = list(divide_batch(arr_out, batchsize))
        classids = []
        if batches:
            for batch in batches:
                inp = torch.from_numpy(batch.transpose(0,3,1,2)).float().to(device) / 255
                # preds = model(inp)
                # classids.append(torch.argmax(preds,1).cpu().numpy())
                classids.append(np.random.choice([0,1], inp.shape[0]))
            classids = np.concatenate(classids)
        else:
            classids = np.array([], dtype=int)

        # Assign classifications
        for (idx_arr,), cid in zip(id_out, classids):
            allobjects[idx_arr]['properties']['classification'] = {
                'name': classnames[cid],
                'colorRGB': colors[cid]
            }

# Save output
if json_annotated_fname.endswith('.gz'):
    with gzip.open(json_annotated_fname, 'wt', encoding='ascii') as out:
        geojson.dump(allobjects, out)
else:
    with open(json_annotated_fname, 'w') as out:
        geojson.dump(allobjects, out)
