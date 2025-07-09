#!/usr/bin/env python3
import os
import gzip
import geojson
import numpy as np
import pandas as pd
import openslide
from shapely.geometry import shape, Polygon
import seaborn as sns
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────────
openslidelevel       = 0
tilesize             = 10000        # must match your analysis script
gsi_fname            = '1L1_nuclei_reg_anno.json'
wsi_fname            = 'MTA3-8_GPC3.svs'

# ─── Load annotated GeoJSON ────────────────────────────────────────────────────
if gsi_fname.endswith('.gz'):
    with gzip.open(gsi_fname, 'rt', encoding='ascii') as f:
        allobjects = geojson.load(f)
else:
    with open(gsi_fname) as f:
        allobjects = geojson.load(f)

# ─── Extract centroids & labels ────────────────────────────────────────────────
geoms     = [shape(obj.get('nucleusGeometry', obj.get('geometry'))) for obj in allobjects]
centroids = [g.centroid for g in geoms]
labels    = [obj['properties']['classification']['name'] for obj in allobjects]

# ─── Open the whole-slide to get dimensions ────────────────────────────────────
osh         = openslide.OpenSlide(wsi_fname)
level_dims  = osh.level_dimensions[openslidelevel]
scalefactor = int(osh.level_downsamples[openslidelevel])

# ─── Tile the slide & count per class ─────────────────────────────────────────
records = []
for y in range(0, level_dims[1], int(tilesize * scalefactor)):
    for x in range(0, level_dims[0], int(tilesize * scalefactor)):
        tile_poly = Polygon([
            (x, y),
            (x + tilesize * scalefactor, y),
            (x + tilesize * scalefactor, y + tilesize * scalefactor),
            (x, y + tilesize * scalefactor),
        ])
        # correct list comprehension: filter centroids inside
        hits = [i for i, c in enumerate(centroids) if tile_poly.contains(c)]
        if not hits:
            continue
        for cls in set(labels):
            count = sum(1 for i in hits if labels[i] == cls)
            records.append({'Class': cls, 'Count': count})

# ─── Build DataFrame & plot ─────────────────────────────────────────────────
df = pd.DataFrame(records)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.violinplot(x="Class", y="Count", hue="Class", data=df, inner="quartile", palette="Set2", legend=False)
plt.title("Per-Tile Cell Counts by Class")
plt.ylabel("Cells per Tile")
plt.tight_layout()
plt.show()
