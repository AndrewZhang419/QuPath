openslidelevel = 0  # level from openslide to read
tilesize = 10000  # size of the tile to load from openslide
patchsize = 32  # patch size needed by our DL model

minhits = 100  # the minimum number of objects needed to be present within a tile for the tile to be computed on
batchsize = 1024  # how many patches we want to send to the GPU at a single time
nclasses = 2  # number of output classes our model is providing
classnames = ["Other", "Lymphocyte"]  # the names of those classes which will appear in QuPath later on
colors = [-377282, -9408287]  # their associated color

mask_patches = False  # blackout surrounding region (if model trained with masks)

json_fname = r'MTA38GPC.json'  # input geojson file
json_annotated_fname = r'1L1_nuclei_reg_anno.json'  # output geojson file
model_fname = "lymph_model.pth"  # DL model path
wsi_fname = "1L1_-_2019-09-10_16.44.58.ndpi"  # WSI file

import os
import openslide

print("Current directory:", os.getcwd())

os.environ['PATH'] = '~/Documents/GitHub/QuPath' + ';' + os.environ['PATH']

from tqdm.autonotebook import tqdm
from math import ceil
import matplotlib.pyplot as plt

import geojson
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon

import torch
from torch import nn
from torchsummary import summary
import numpy as np
import cv2
import gzip

device = torch.device('cuda')

def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]

# --- Load your model here
# model = LoadYourModelHere().to(device)
# checkpoint = torch.load(model_fname, map_location=lambda storage, loc: storage)
# model.load_state_dict(checkpoint["model_dict"])
# model.eval()
# summary(model, (3, 32, 32))

if json_fname.endswith(".gz"):
    with gzip.GzipFile(json_fname, 'r') as f:
        allobjects = geojson.loads(f.read(), encoding='ascii')
else:
    with open(json_fname) as f:
        allobjects = geojson.load(f)

print("done loading")

allshapes = [shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj else obj["geometry"]) for obj in allobjects]
allcenters = [s.centroid for s in allshapes]
point_to_index = {pt: i for i, pt in enumerate(allcenters)}
searchtree = STRtree(allcenters)
print("done building tree")

osh = openslide.OpenSlide(wsi_fname)
nrow, ncol = osh.level_dimensions[0]
nrow = ceil(nrow / tilesize)
ncol = ceil(ncol / tilesize)

scalefactor = int(osh.level_downsamples[openslidelevel])
paddingsize = patchsize // 2 * scalefactor
int_coords = lambda x: np.array(x).round().astype(np.int32)

for y in tqdm(range(0, osh.level_dimensions[0][1], round(tilesize * scalefactor)), desc="outer", leave=False):
    for x in tqdm(range(0, osh.level_dimensions[0][0], round(tilesize * scalefactor)), desc=f"inner {y}", leave=False):

        tilepoly = Polygon([[x, y], [x + tilesize * scalefactor, y],
                            [x + tilesize * scalefactor, y + tilesize * scalefactor],
                            [x, y + tilesize * scalefactor]])
        hits = searchtree.query(tilepoly)

        if len(hits) < minhits:
            continue

        tile = np.asarray(osh.read_region((x - paddingsize, y - paddingsize), openslidelevel,
                                          (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize)))[:, :, :3]

        if mask_patches:
            mask = np.zeros(tile.shape[:2], dtype=tile.dtype)
            exteriors = [int_coords(allshapes[point_to_index[hit]].boundary.coords) for hit in hits]
            exteriors_shifted = [(ext - np.array([(x - paddingsize), (y - paddingsize)])) // scalefactor for ext in exteriors]
            cv2.fillPoly(mask, exteriors_shifted, 1)

        arr_out = np.zeros((len(hits), patchsize, patchsize, 3))
        id_out = np.zeros((len(hits), 1))

        for i, (hit, arr, id) in enumerate(zip(hits, arr_out, id_out)):
            px, py = hit.coords[0]
            c = int((px - x + paddingsize) // scalefactor)
            r = int((py - y + paddingsize) // scalefactor)
            patch = tile[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2, :]

            if mask_patches:
                maskpatch = mask[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2]
                patch = np.multiply(patch, maskpatch[:, :, None])

            arr[:] = patch
            id[:] = point_to_index[hit]

        classids = []
        for batch_arr in tqdm(divide_batch(arr_out, batchsize), leave=False):
            batch_arr_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2)).type(torch.FloatTensor).to(device) / 255
            classids.append(np.random.choice([0, 1], batch_arr_gpu.shape[0]))
        classids = np.hstack(classids)

        for id, classid in zip(id_out, classids):
            allobjects[int(id)]["properties"]['classification'] = {
                'name': classnames[classid],
                'colorRGB': colors[classid]
            }

if json_annotated_fname.endswith(".gz"):
    with gzip.open(json_annotated_fname, 'wt', encoding="ascii") as zipfile:
        geojson.dump(allobjects, zipfile)
else:
    with open(json_annotated_fname, 'w') as outfile:
        geojson.dump(allobjects, outfile)