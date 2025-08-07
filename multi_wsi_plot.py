#!/usr/bin/env python3
import argparse, glob, os, gzip, geojson
import numpy as np
import pandas as pd
import openslide
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import shape

def load_qupath_objects(json_path):
    if json_path.endswith('.gz'):
        with gzip.open(json_path, 'rt') as f:
            return geojson.load(f)
    with open(json_path, 'r') as f:
        return geojson.load(f)

def analyze_json(json_path, wsi_path, level=0, tilesize=10000):
    objs = load_qupath_objects(json_path)
    geoms = [shape(o.get('nucleusGeometry', o.get('geometry'))) for o in objs]
    xs = np.array([g.centroid.x for g in geoms])
    ys = np.array([g.centroid.y for g in geoms])
    labels = [o['properties']['classification']['name'] for o in objs]

    slide = openslide.OpenSlide(wsi_path)
    scale_factor = int(slide.level_downsamples[level])
    step = tilesize * scale_factor
    tx = (xs // step).astype(int)
    ty = (ys // step).astype(int)

    df = pd.DataFrame({'tx': tx, 'ty': ty, 'Class': labels})
    dfc = (
        df.groupby(['tx','ty','Class'], sort=False)
          .size().reset_index(name='Count')
    )

    basename = os.path.splitext(os.path.basename(json_path))[0]
    dfc['Annotation'] = basename
    return dfc

def main():
    p = argparse.ArgumentParser(
        description="Plot log₂(cells/tile+1) for Positive vs Negative"
    )
    # JSON input: single file or directory
    gj = p.add_mutually_exclusive_group(required=True)
    gj.add_argument(
        '-j','--json', dest='json_path',
        help='One QuPath JSON file (.json or .json.gz)'
    )
    gj.add_argument(
        '-J','--json-dir', dest='json_dir',
        help='Folder of QuPath JSONs (*.json or *.json.gz)'
    )
    # WSI input: single file or directory
    gw = p.add_mutually_exclusive_group(required=True)
    gw.add_argument(
        '-w','--wsi', dest='wsi_path', required=True,
        help='One WSI file (.svs) OR folder of WSIs'
    )
    p.add_argument(
        '--level', type=int, default=0,
        help='OpenSlide level to use'
    )
    p.add_argument(
        '--tilesize', type=int, default=10000,
        help='Tile side length in px at that level'
    )
    args = p.parse_args()

    # 1) Gather JSON files
    if args.json_dir:
        json_files = sorted(glob.glob(os.path.join(args.json_dir, '*.json*')))
    else:
        json_files = [args.json_path]

    # 2) Gather WSI files
    if os.path.isdir(args.wsi_path):
        wsi_list = sorted(glob.glob(os.path.join(args.wsi_path, '*.svs')))
        wsi_map  = {os.path.splitext(os.path.basename(w))[0]: w for w in wsi_list}
        single_mode = False
    else:
        wsi_map = {None: args.wsi_path}
        single_mode = True

    # 3) Pair each JSON with its matching WSI
    pairs = []
    for json_path in json_files:
        base = os.path.splitext(os.path.basename(json_path))[0]
        if single_mode:
            pairs.append((json_path, wsi_map[None]))
        else:
            if base not in wsi_map:
                raise RuntimeError(f"No matching WSI for JSON '{base}'")
            pairs.append((json_path, wsi_map[base]))

    # 4) Run analysis on each (json, wsi) pair
    dfs = [
        analyze_json(json_path, wsi_path, args.level, args.tilesize)
        for json_path, wsi_path in pairs
    ]

    # 5) Combine and compute log2 counts
    df = pd.concat(dfs, ignore_index=True)
    df['Log2Count'] = np.log2(df['Count'] + 1)

    # 6) Plot
    plt.figure(figsize=(max(12, len(dfs)*0.6), 6))
    sns.set(style="whitegrid")
    ax = sns.stripplot(
        x='Annotation', y='Log2Count', hue='Class', data=df,
        dodge=True, jitter=0.3, size=1, color='gray', alpha=0.3,
        hue_order=['Negative','Positive'], legend=False
    )
    sns.boxplot(
        x='Annotation', y='Log2Count', hue='Class', data=df,
        width=0.3, dodge=True, saturation=1.0,
        palette={'Negative':'steelblue','Positive':'firebrick'},
        boxprops={'edgecolor':'black','linewidth':1},
        whiskerprops={'color':'black','linewidth':1},
        capprops={'color':'black','linewidth':1},
        medianprops={'color':'black','linewidth':1},
        flierprops={'marker':' '},
        hue_order=['Negative','Positive'], ax=ax
    )
    ymin, ymax = ax.get_ylim()
    for i in range(len(dfs)):
        ax.vlines(i, ymin, ymax, color='black', linewidth=0.5, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    label_map = {'Positive':'GPC3 Positive','Negative':'GPC3 Negative'}
    ax.legend(handles, [label_map.get(l,l) for l in labels],
              title='Class', loc='upper right')

    ax.set_ylabel('log2(Cells per Tile + 1)')
    ax.set_xlabel('Sample number')
    plt.title('MTA3-8 Log of GPC3+ Positive vs Negative Cells')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == '__main__':
    main()
