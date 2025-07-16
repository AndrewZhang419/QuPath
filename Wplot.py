import argparse, glob, os, gzip, geojson, re
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
    # 1) Get centroids & classes
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
        df.groupby(['tx', 'ty', 'Class'], sort=False)
          .size()
          .reset_index(name='Count')
    )

    # Extract only the number after "rect"
    basename = os.path.splitext(os.path.basename(json_path))[0]
    match = re.search(r'rect(\d+)', basename)
    annotation_label = match.group(1) if match else basename
    dfc['Annotation'] = annotation_label

    return dfc

def main():
    p = argparse.ArgumentParser(
        description="Plot log₂(cells/tile+1) for Positive vs Negative"
    )
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument(
        '-j', '--json', dest='json_path',
        help='One QuPath JSON file'
    )
    gp.add_argument(
        '-J', '--json-dir', dest='json_dir',
        help='Folder of QuPath JSONs (*.json)'
    )
    p.add_argument(
        '-w', '--wsi', dest='wsi_path', required=True,
        help='Whole‐slide image file (.svs)'
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

    if args.json_dir:
        files = sorted(glob.glob(os.path.join(args.json_dir, '*.json')))
    else:
        files = [args.json_path]

    dfs = [analyze_json(f, args.wsi_path, args.level, args.tilesize)
           for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['Log2Count'] = np.log2(df['Count'] + 1)

    plt.figure(figsize=(max(12, len(files) * 0.6), 6))
    sns.set(style="whitegrid")

  ###  ax = sns.stripplot(
  ###      x='Annotation', y='Log2Count', hue='Class', data=df,
  ###      dodge=True, jitter=0.3, size=1, color='gray', alpha=0.3,
  ###      hue_order=['Negative', 'Positive'],
  ###      legend=False
  ###  )

  ###  sns.boxplot(
  ###      x='Annotation', y='Log2Count', hue='Class', data=df,
  ###      width=0.3, dodge=True, saturation=1.0,
  ###      palette={'Negative': 'steelblue', 'Positive': 'firebrick'},
  ###      boxprops={'edgecolor': 'black', 'linewidth': 1},
  ###      whiskerprops={'color': 'black', 'linewidth': 1},
  ###      capprops={'color': 'black', 'linewidth': 1},
  ###      medianprops={'color': 'black', 'linewidth': 1},
  ###      flierprops={'marker': ' '},
  ###      hue_order=['Negative', 'Positive'],
  ###      ax=ax
  ###  )

###if function doesn't work or says line error, make sure to just add 4 spaces

    ax = sns.violinplot(
        x='Annotation', y='Log2Count', hue='Class', data=df,
        split=True, inner='quartile',
        palette={'Negative': 'steelblue', 'Positive': 'firebrick'},
        hue_order=['Negative', 'Positive']
    )

    ymin, ymax = ax.get_ylim()
    for i in range(len(files)):
        ax.vlines(i, ymin, ymax, color='black', linewidth=0.5, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    label_map = {'Positive': 'GPC3 Positive', 'Negative': 'GPC3 Negative'}
    new_labels = [label_map.get(label, label) for label in labels]
    ax.legend(handles, new_labels, title='Class', loc='upper right')

    ax.set_ylabel('log2(Cells\u2009per\u2009Tile\u200A+\u200A1)')
    ax.set_xlabel('Sample number')  # Set x-axis label
    plt.title('MTA3-8 Log of GPC3+ Positive vs Negative Cells')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Give room for the xlabel
    plt.show()

if __name__ == '__main__':
    main()