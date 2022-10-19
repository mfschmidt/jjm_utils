#!/usr/bin/env python3

# plot_mtl_copes.py

import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np
import pandas as pd
import re
from scipy.stats import ttest_1samp


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Read in contrast copes from Feat analyses and plot "
                    "them voxel-wise in hippocampus and amygdala.",
    )
    parser.add_argument(
        "input",
        help="The path to the csv data file created by extract_copes_per_mask.py",
    )
    parser.add_argument(
        "-o", "--output",
        help="The path to which the plot will be written.",
    )
    parser.add_argument(
        "--skip-voxels", action="store_true",
        help="set to skip plotting of individual voxels",
    )
    parser.add_argument(
        "--plot-boxes", action="store_true",
        help="set to plot a box plot over the scatter",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Check conditions
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Path '{args.input}' does not exist.")
        sys.exit(1)

    if args.output is None:
        setattr(
            args, "output", 
            str(input_path.parent / input_path.name.replace("csv", "png"))
        )

    # Store some calculated directories in their own variables
    subject_dir = input_path.parent.parent.parent

    # Determine subject and project names from them
    if subject_dir.name.startswith("sub-"):
        setattr(args, "subject", subject_dir.name)
        setattr(args, "project", subject_dir.parent.name)
        if args.verbose:
            print(f"Subject appears to be {args.subject}, "
                  f"from the {args.project} project.")
    else:
        print("Could not determine subject or project "
              f"from data file {args.input}.")

    return args


def get_voxels(path):
    """
    """

    # Load the data requested
    voxels = pd.read_csv(path)
    
    # Determine laterality from the end of the mask name
    voxels['hemi'] = voxels['mask'].apply(lambda x: x[-2:])

    # Determine atlas name from mask name
    voxels['atlas'] = voxels['mask'].apply(lambda x: x[9:12])

    # Strip the atlas characteristics from the beginning and end each mask.
    # Typical mask is like:
    # "res-bold_fs7_Name_mask.T1.lh"  # before next line
    # "-------------Name-----------"  # after next line
    voxels['mask'] = voxels['mask'].apply(lambda x: x[13: -11])

    return voxels.loc[voxels['atlas'].isin(['amy', 'fs7', ]), :]


def get_color_map(voxels, alpha=0.7):
    """
    """

    # Extract all regions recognized by FreeSurfer
    lut_file = Path(os.environ["FREESURFER_HOME"]) / "FreeSurferColorLUT.txt"
    line_re = re.compile(r"([0-9]+)\s+([\w-]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+).*")
    regions = []
    for line in open(lut_file, "r"):
        match = line_re.match(line)
        if match:
            regions.append({
                "id": int(match.group(1)),
                "desc": match.group(2).replace('-', ''),
                "r": int(match.group(3)),
                "g": int(match.group(4)),
                "b": int(match.group(5)),
                "a": int(match.group(6)),
            })
    lut = pd.DataFrame(regions)

    # Save the ones we need in a seaborn-friendly color map
    cmap = {}
    for desc in voxels['mask'].unique():
        color = lut[lut['desc'] == desc].iloc[0]
        # all fs7 colors are alpha==0, which makes them transparent.
        # We override this with our own alpha
        c = (
            color.get("r", 0) / 255,
            color.get("g", 0) / 255,
            color.get("b", 0) / 255,
            alpha,  # color.get("a", 0) / 255,  
        )
        cmap[desc] = c

    return cmap


def get_marker_map(voxels):
    """
    """

    # Hard-coded marker map, by atlas
    markers = {
        "fs7": "o",
        "amy": "^",
    }

    # Seaborn needs the markers mapped to regions
    mmap = {}
    for desc in voxels['mask'].unique():
        atlases = voxels.loc[voxels['mask'] == desc, 'atlas']
        if len(atlases) > 0:
            atlas = atlases.iloc[0]
        else:
            atlas = "fs7"  # default
            print(f"Region {desc} found no atlas!?")
        mmap[desc] = markers[atlas]

    return mmap


def plot(
        voxels,
        title="Voxels",
        plot_voxels=True,
        plot_boxes=False,
        cmap=None,
        mmap=None
):
    """
    """

    # Add a mean y for each region, based on precise y values
    y_means = voxels[['hemi', 'mask', 'y']].groupby(['hemi', 'mask']).mean()
    voxels['mean_y'] = voxels.apply(
        lambda row: y_means.loc[(row['hemi'], row['mask']), 'y'],
        axis=1
    )

    # Jitter the y (in data, x in plot) values to spread points out.
    # Multiply a normal distribution with mean 0 and sd 1 by a fake sd
    # We want to spread them out a bit, but not overlap other y-values
    voxels['jitter_y'] = voxels['y'] + np.random.randn(len(voxels)) * 0.20

    # Box plots cannot categorize by floats, and NaNs cannot be made into ints
    voxels.loc[voxels['mean_y'].isna(), 'mean_y'] = 0.0
    voxels['mean_y'] = voxels['mean_y'].astype(int)
    # Finally, after modifications are complete, filter only valid voxels for plotting.
    voxels = voxels.dropna()

    min_x, max_x = voxels['y'].min(), voxels['y'].max()
    min_value, max_value = voxels['value'].min(), voxels['value'].max()

    fig, (left_ax, right_ax) = plt.subplots(ncols=2, sharey='all', figsize=(18, 6))

    # Plot the COPE value at each voxel, colored by region
    if plot_voxels:
        sns.scatterplot(
            data=voxels[voxels['hemi'] == 'lh'], x='jitter_y', y='value',
            hue="mask", palette=cmap, style='mask', markers=mmap, s=60,
            zorder=3, ax=left_ax,
        )
        sns.scatterplot(
            data=voxels[voxels['hemi'] == 'rh'], x='jitter_y', y='value',
            hue="mask", palette=cmap, style='mask', markers=mmap, s=60,
            zorder=3, ax=right_ax,
        )

    # Plot the COPE values box at each region, colored by region
    if plot_boxes:
        for hemisphere in [
            {"hemi": 'lh', "ax": left_ax, },
            {"hemi": 'rh', "ax": right_ax, },
        ]:
            for region in voxels['mask'].unique():
                region_idx = voxels.loc[voxels['mask'] == region].index
                hemi_idx = voxels.loc[voxels['hemi'] == hemisphere['hemi']].index
                idx = region_idx.intersection(hemi_idx)
                beta_values = voxels.loc[idx, 'value']
                y_values = voxels.loc[idx, 'y']
                mean_beta = np.mean(beta_values)
                sd_beta = np.std(beta_values)
                min_beta = np.min(beta_values)
                max_beta = np.max(beta_values)
                mean_y = np.mean(y_values)
                edge_color = (
                    cmap[region][0], cmap[region][1], cmap[region][2], 1.0,
                )
                face_color = (
                    cmap[region][0], cmap[region][1], cmap[region][2], 0.2,
                )
                hemisphere['ax'].add_patch(patches.Rectangle(
                    (mean_y - 0.02, min_beta),
                    0.04, max_beta - min_beta,
                    ec=edge_color,
                    zorder=1, fill=False
                ))
                hemisphere['ax'].add_patch(patches.Rectangle(
                    (mean_y - 0.5, mean_beta - sd_beta),
                    1.0, 2.0 * sd_beta,
                    ec=edge_color, fc=face_color,
                    zorder=2, fill=True
                ))
                ttest_result = ttest_1samp(beta_values, popmean=0.0)
                if ttest_result.pvalue < 0.01 and ttest_result.statistic > 0:
                    hemisphere['ax'].text(
                        mean_y, max_value, '**',
                        color=edge_color, ha='center', va='bottom', weight='bold',
                        transform=hemisphere['ax'].transData
                    )
                elif ttest_result.pvalue < 0.05 and ttest_result.statistic > 0:
                    hemisphere['ax'].text(
                        mean_y, max_value, '*',
                        color=edge_color, ha='center', va='top', weight='bold',
                        transform=hemisphere['ax'].transData
                    )
                if ttest_result.pvalue < 0.01 and ttest_result.statistic < 0:
                    hemisphere['ax'].text(
                        mean_y, min_value, '**',
                        color=edge_color, ha='center', va='top', weight='bold',
                        transform=hemisphere['ax'].transData
                    )
                elif ttest_result.pvalue < 0.05 and ttest_result.statistic < 0:
                    hemisphere['ax'].text(
                        mean_y, min_value, '*',
                        color=edge_color, ha='center', va='bottom', weight='bold',
                        transform=hemisphere['ax'].transData
                    )

    # Use only one legend, and put it outside of the plotting area.
    left_ax.legend().remove()
    right_ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    for ax in [left_ax, right_ax, ]:
        # With A and P on the plot for voxels, we use hlines rather than
        # axhline to cut this zero line short of those A/P annotations.
        # The axes leaves a margin between far left and min_x, same on R
        ax.hlines(y=0, xmin=min_x, xmax=max_x, linestyle=":", color="gray")
        # ax.set_xticks([])
        ax.set_ylabel("GLM contrast of parameter estimates")

    # Expand the x axis limits just a bit to make way for head/tail annotations
    left_ax.set_xlim((min_x - 1, max_x + 1))
    right_ax.set_xlim((min_x - 1, max_x + 1))  # will be inverted below
    right_ax.invert_xaxis()

    # Throw in some head and tail annotations to ensure clarity
    left_ax.set_xlabel("posterior  -  anterior")
    left_ax.annotate("A", xy=(max_x + 0.25, 0.0), xycoords="data",
                     horizontalalignment="left", verticalalignment="center")
    left_ax.annotate("P", xy=(min_x - 0.25, 0.0), xycoords="data",
                     horizontalalignment="right", verticalalignment="center")

    right_ax.set_xlabel("anterior  -  posterior")
    right_ax.annotate("A", xy=(max_x + 0.25, 0.0), xycoords="data",
                      horizontalalignment="right", verticalalignment="center")
    right_ax.annotate("P", xy=(min_x - 0.25, 0.0), xycoords="data",
                      horizontalalignment="left", verticalalignment="center")

    left_ax.set_title("Left hippocampus and amygdala")
    right_ax.set_title("Right hippocampus and amygdala")

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def get_title(args):
    """
    """

    pattern = re.compile(r".*cope-(\S+)_voxels.*")
    match = pattern.search(Path(args.input).name)
    if match:
        plot_title = (
            f"{args.project.upper()} Subject {args.subject}: {match.group(1)}"
        )
    else:
        plot_title = (
            f"{args.project.upper()} Subject {args.subject}: unknown score"
        )
    return plot_title


def main(args):
    """ Entry point """

    voxels = get_voxels(args.input)

    # Plot the data and save the plot
    figure = plot(
        voxels,
        title=get_title(args),
        plot_voxels=(not args.skip_voxels),
        plot_boxes=args.plot_boxes,
        cmap=get_color_map(voxels, alpha=0.5),
        mmap=get_marker_map(voxels),
    )
    if args.verbose:
        print(f"Writing figure to {args.output}")
    figure.savefig(args.output)


if __name__ == "__main__":
    main(get_arguments())
