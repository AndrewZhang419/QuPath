import os
import openslide
from tqdm.autonotebook import tqdm
from math import ceil
import matplotlib.pyplot as plt
import geojson
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
import torch
import numpy as np
import cv2
import gzip

# --- Configurations ---
openslidelevel = 0
tilesize = 10000
patchsize = 32
minhits = 100
batchsize = 1024
nclasses = 2
classnames = ["Other", "Lymphocyte"]
colors = [-377282, -9408287]
mask_patches = False

json_fname = r'MTA38GPC.json'
json_annotated_fname = r'1L1_nuclei_reg_anno.json'
model_fname = "lymph_model.pth"
wsi_fname = "MTA3-8_GPC3.svs"

device = torch.device('cuda')

def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]

# --- Load JSON ---
if json_fname.endswith(".gz"):
    with gzip.GzipFile(json_fname, 'r') as f:
        allobjects = geojson.loads(f.read(), encoding='ascii')
else:
    with open(json_fname) as f:
        allobjects = geojson.load(f)

print("done loading")

# --- Geometry Processing ---
allshapes = [shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj else obj["geometry"]) for obj in allobjects]
allcenters = [s.centroid for s in allshapes]
print("done converting")

# --- Build STRtree + Coordinate Mapping ---
searchtree = STRtree(allcenters)
center_to_index = {(pt.x, pt.y): idx for idx, pt in enumerate(allcenters)}  # Robust coordinate-based mapping
print("done building tree")

# --- Open Slide ---
osh = openslide.OpenSlide(wsi_fname)
nrow, ncol = osh.level_dimensions[0]
nrow = ceil(nrow / tilesize)
ncol = ceil(ncol / tilesize)

scalefactor = int(osh.level_downsamples[openslidelevel])
paddingsize = patchsize // 2 * scalefactor

int_coords = lambda x: np.array(x).round().astype(np.int32)

# --- Processing Loop ---
for y in tqdm(range(0, osh.level_dimensions[0][1], round(tilesize * scalefactor)), desc="outer", leave=False):
    for x in tqdm(range(0, osh.level_dimensions[0][0], round(tilesize * scalefactor)), desc=f"inner {y}", leave=False):
        tilepoly = Polygon([[x, y], [x + tilesize * scalefactor, y],
                            [x + tilesize * scalefactor, y + tilesize * scalefactor],
                            [x, y + tilesize * scalefactor]])
        hits = searchtree.query(tilepoly)

        if len(hits) < minhits:
            continue

        tile = np.asarray(osh.read_region((x - paddingsize, y - paddingsize), openslidelevel,
                                          (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize)))[:, :, 0:3]

        if mask_patches:
            mask = np.zeros((tile.shape[0:2]), dtype=tile.dtype)
            exteriors = [int_coords(allshapes[center_to_index[(hit.x, hit.y)]].boundary.coords) for hit in hits]
            exteriors_shifted = [(ext - np.array([(x - paddingsize), (y - paddingsize)])) // scalefactor for ext in exteriors]
            cv2.fillPoly(mask, exteriors_shifted, 1)

        arr_out = np.zeros((len(hits), patchsize, patchsize, 3))
        id_out = np.zeros((len(hits), 1))

        for hit, arr, id in zip(hits, arr_out, id_out):
            px, py = hit.coords[:][0]
            idx = center_to_index[(px, py)]  # Lookup using coordinates
            c = int((px - x + paddingsize) // scalefactor)
            r = int((py - y + paddingsize) // scalefactor)
            patch = tile[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2, :]

            if mask_patches:
                maskpatch = mask[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2]
                patch = np.multiply(patch, maskpatch[:, :, None])

            arr[:] = patch
            id[:] = idx  # Save index safely

        # Dummy classifier (replace with real model later)
        classids = []
        for batch_arr in tqdm(divide_batch(arr_out, batchsize), leave=False):
            batch_arr_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2)).float().to(device) / 255
            classids.append(np.random.choice([0, 1], batch_arr_gpu.shape[0]))
        classids = np.hstack(classids)

        for id, classid in zip(id_out, classids):
            allobjects[int(id)]["properties"]['classification'] = {
                'name': classnames[classid],
                'colorRGB': colors[classid]
            }

# --- Save Output JSON ---
if json_annotated_fname.endswith(".gz"):
    with gzip.open(json_annotated_fname, 'wt', encoding="ascii") as zipfile:
        geojson.dump(allobjects, zipfile)
else:
    with open(json_annotated_fname, 'w') as outfile:
        geojson.dump(allobjects, outfile)

print("Processing completed successfully.")