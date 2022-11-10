#!/usr/bin/env python3

# plot_mtl_copes.py

import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches, path
import seaborn as sns
import numpy as np
import pandas as pd
import re
from scipy.stats import ttest_1samp, ttest_ind


sns.set_style(style='white')


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Read in contrast copes from Feat analyses and plot "
                    "them as beta values per ROI.",
    )
    parser.add_argument(
        "input", type=str,
        help="This app has two modes, if given a csv file as input, "
             "This is the path to the csv data file created by "
             "extract_copes_per_mask.py. The voxels in that csv file"
             "will be plotted. But if given a directory, "
             "this directory contains subjects, each of which has "
             "prepared csv files. Those voxels will be summarized "
             "per subject and subject means will be plotted.",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="The path to which the plot will be written.",
    )
    parser.add_argument(
        "--group-file", type=str,
        help="The path to find the group membership of each subject.",
    )
    parser.add_argument(
        "--mask-csv-subdir", type=str,
        help="The subdirectory between 'masks' and csv files.",
    )
    parser.add_argument(
        "--cope", type=int, default=0,
        help="Plot only this cope. Only necessary for group plots.",
    )
    parser.add_argument(
        "--atlas", type=str, default="",
        help="Regions from this atlas will be used to group voxels.",
    )
    parser.add_argument(
        "--winsorize", action="store_true",
        help="if true, change the value of outliers > 3 global SD to 3 SD.",
    )
    parser.add_argument(
        "--skip-voxels", action="store_true",
        help="set to skip plotting of individual voxels",
    )
    parser.add_argument(
        "--draw-boxes", action="store_true",
        help="set to plot a box plot over the scatter",
    )
    parser.add_argument(
        "--plot-at-mean-ap", action="store_true",
        help="set to plot data at true mean anterior-posterior coordinates",
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

    if input_path.is_file():
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
    elif input_path.is_dir():
        setattr(args, "subject", "group")
        setattr(args, "project", input_path.name)

    return args


def parse_masks(dataframe):
    """
    Parse the 'mask' column, saving parts as their own columns
    """

    # Strip the atlas characteristics from the beginning and end each mask.
    # Typical mask is like:
    # "res-bold_fs7_Name_mask.T1.lh"  # before parsing

    dataframe['hemi'] = dataframe['mask'].apply(
        lambda mask: mask.split(".")[-1]
    )
    dataframe['short_mask'] = dataframe['mask'].apply(
        lambda mask: mask.split(".")[0].replace("res-bold_", "")
    )
    dataframe['atlas'] = dataframe['short_mask'].apply(
        lambda mask: mask.split("_")[0]
    )
    dataframe['mask'] = dataframe['short_mask'].apply(
        lambda mask: "_".join(mask.split("_")[1: -1])
    )
    dataframe['old_mask'] = dataframe['short_mask'].apply(
        lambda mask: mask.split("_")[1]
    )
    dataframe['weighted_value'] = dataframe['value'] * dataframe['label']

    return dataframe


def get_subject_means(project_dir, feat_threshold_string, group_map, verbose=False):
    """

        :param Path project_dir: the directory containing subject directories
        :param str feat_threshold_string: the name of the subdirectory
                                          containing csv files, something like
                                          'lev1_2_m6_5mm_T1_t-0.50'
        :param dict group_map: subject to group mapping dict
        :param bool verbose: set to True for additional print info

        :return dict: dict with subjects as keys and diagnoses as values
    """

    cope_files = []
    re_pattern = re.compile(r"(sub-\S+)_cope-cope([0-9]+)_voxels_by_masks.csv")
    glob_pattern = f"sub-*/masks/{feat_threshold_string}/sub-*_cope-cope*_voxels_by_masks.csv"
    i = 0
    for i, file in enumerate(project_dir.glob(glob_pattern)):
        match = re_pattern.match(file.name)
        if match:
            df = pd.read_csv(file)
            subject_id = match.group(1)
            df['dx'] = group_map.get(subject_id, "NA")
            df['cope'] = int(match.group(2))
            cope_files.append(df)
            if verbose:
                print(f"found {str(file)} with {len(df)} voxels")
        else:
            print(f"{file} did not match expected BIDS naming.")
    if verbose:
        print(f"Read {i + 1} files")

    voxel_level_data = parse_masks(pd.concat(cope_files, axis=0))

    groupers = ['subject', 'hemi', 'mask', 'cope', 'atlas', 'dx', ]
    columns = ['value', 'weighted_value', 'y']
    region_level_data = voxel_level_data[groupers + columns].groupby(groupers).mean().reset_index()

    region_midpoints = region_level_data[['hemi', 'mask', 'y', ]].groupby(['hemi', 'mask']).mean()
    region_level_data['group_y'] = region_level_data.apply(
        lambda row: region_midpoints.loc[(row['hemi'], row['mask']), 'y'],
        axis=1
    )

    return region_level_data


def get_group_map(csv_file, verbose=False):
    """ Read csv file containing diagnoses per subject, return as a dict

        :param Path csv_file: the csv file containing diagnoses
        :param bool verbose: set to True for additional print info

        :return dict: dict with subjects as keys and diagnoses as values
    """

    groups = pd.read_csv(csv_file, index_col=0)
    groups = groups[["MDD_status (MDD=1, Hv=2)"]]
    groups['dx'] = groups['MDD_status (MDD=1, Hv=2)'].apply(
        lambda status: {1: "MDD", 2: "HV"}[status] if status in [1, 2] else "NA"
    )
    if verbose:
        print(f"Group found for {len(groups)} subjects.")
        print(groups.groupby('dx').count())

    dx_map = {}
    for idx, row in groups.iterrows():
        dx_map[idx] = row['dx']

    return dx_map


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
        usable_desc = desc if desc in lut['desc'].values else f"HP_{desc}"
        color = lut[lut['desc'] == usable_desc].iloc[0]
        # all fs7 colors are alpha==0, which makes them transparent.
        # We override this with our own alpha
        c = (
            color.get("r", 0) / 255,
            color.get("g", 0) / 255,
            color.get("b", 0) / 255,
            alpha,  # color.get("a", 0) / 255,  
        )
        cmap[desc] = c

    # Tack on some colors for grouping
    cmap["mdd"] = (0.6980, 0.1333, 0.1333, 0.5000, )
    cmap["hv"] = (0.1333, 0.5451, 0.1333, 0.5000, )

    return cmap


def get_marker_map(voxels):
    """ Define plot markers to differentiate atlases
    """

    # Hard-coded marker map, by atlas
    markers = {
        "fs7": "o",
        "amy": "^",
        "hbt": "s",
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


def get_safe_stats(data):
    """

    """

    if len(data) > 0:
        return np.mean(data), np.std(data), np.min(data), np.max(data)
    else:
        return 0.0, 0.0, 0.0, 0.0


def draw_significance(p, t, x, top, bottom, c, ax):
    """
    """

    sig_y = top * 1.1 if t > 0 else bottom * 1.1
    if p < 0.01:
        sig_string = "***"
        sig_va = 'top' if t < 0 else 'bottom'
    elif p < 0.05:
        sig_string = "**"
        sig_va = 'center'
    elif p < 0.1:
        sig_string = "*"
        sig_va = 'bottom' if t < 0 else 'top'
    else:
        sig_string, sig_va = "N/A", "N/A"
    if p < 0.1:
        ax.text(
            x, sig_y, sig_string,
            color=c, ha='center', va=sig_va, weight='bold',
            transform=ax.transData
        )


def regions_in_ap_order(voxels):
    return voxels.sort_values('ap_order')['mask'].unique()


def plot_comparisons(voxels, y, hemi, cmap, top, bottom, ax):
    """ """

    labeled = set()
    for region in regions_in_ap_order(voxels):
        region_idx = voxels.loc[voxels['mask'] == region].index
        hemi_idx = voxels.loc[voxels['hemi'] == hemi].index
        rgn_mean_y = np.mean(voxels.loc[region_idx.intersection(hemi_idx), y])
        comp_values = {}
        for dx in voxels['dx'].unique():
            dx_idx = voxels.loc[voxels['dx'] == dx].index
            idx = region_idx.intersection(hemi_idx).intersection(dx_idx)
            comp_values[dx] = voxels.loc[idx, 'value'].values
            mean_beta, sd_beta, min_beta, max_beta = get_safe_stats(comp_values[dx])
            mean_y, sd_y, min_y, max_y = get_safe_stats(voxels.loc[idx, y].values)
            c = cmap.get(dx.lower(), (0.5, 0.5, 0.5, 0.5))
            edge_color = (c[0], c[1], c[2], 0.8)
            face_color = (c[0], c[1], c[2], 0.2)

            # Draw the vertical range line
            range_path = path.Path(
                np.array([
                    (mean_y - 0.01 if dx == 'HV' else mean_y + 0.01, min_beta),
                    (mean_y - 0.01 if dx == 'HV' else mean_y + 0.01, max_beta),
                ]),
                [path.Path.MOVETO, path.Path.LINETO, ]
            )
            ax.add_patch(patches.PathPatch(
                range_path, ec=edge_color, zorder=1, fill=False
            ))

            # Only create one label per legend,
            # so remember if we've labeled this dx yet.
            label = dx if dx not in labeled else None
            labeled.add(dx)
            ax.add_patch(patches.Rectangle(
                (mean_y - 0.3 if dx == 'HV' else mean_y + 0.01, mean_beta - sd_beta),
                0.29, 2.0 * sd_beta, label=label,
                ec=edge_color, fc=face_color,
                zorder=2, fill=True
            ))

        min_value, max_value = voxels['value'].min(), voxels['value'].max()
        top = np.max((np.abs(min_value), np.abs(max_value)))
        bottom = -1.0 * top
        if len(comp_values['MDD']) > 1 and len(comp_values['HV']) > 1:
            ttest_result = ttest_ind(comp_values['HV'], comp_values['MDD'])
            p, t = ttest_result.pvalue, ttest_result.statistic
            draw_significance(p, t, rgn_mean_y, top, bottom,
                              cmap[region], ax)


def plot_boxes(voxels, y, hemi, cmap, top, bottom, ax):
    """

    """

    for region in regions_in_ap_order(voxels):
        region_idx = voxels.loc[voxels['mask'] == region].index
        hemi_idx = voxels.loc[voxels['hemi'] == hemi].index
        idx = region_idx.intersection(hemi_idx)
        if len(idx) < 1:
            continue  # can't do anything with an empty list, skip this region.
        beta_values = voxels.loc[idx, 'value'].values
        mean_beta, sd_beta, min_beta, max_beta = get_safe_stats(beta_values)
        mean_y, sd_y, min_y, max_y = get_safe_stats(voxels.loc[idx, y].values)
        edge_color = (
            cmap[region][0], cmap[region][1], cmap[region][2], 1.0,
        )
        face_color = (
            cmap[region][0], cmap[region][1], cmap[region][2], 0.2,
        )

        # Draw the vertical range line
        range_path = path.Path(
            np.array([(mean_y, min_beta), (mean_y, max_beta), ]),
            [path.Path.MOVETO, path.Path.LINETO, ]
        )
        ax.add_patch(patches.PathPatch(
            range_path, ec=edge_color, zorder=1, fill=False
        ))

        # Draw the box
        ax.add_patch(patches.Rectangle(
            (mean_y - 0.4, mean_beta - sd_beta),
            0.8, 2.0 * sd_beta,
            ec=edge_color, fc=face_color,
            zorder=2, fill=True
        ))

        # Annotate significance asterisks
        min_value, max_value = voxels['value'].min(), voxels['value'].max()
        top = np.max((np.abs(min_value), np.abs(max_value)))
        bottom = -1.0 * top
        if len(beta_values) > 1:
            ttest_result = ttest_1samp(beta_values, popmean=0.0)
            p, t = ttest_result.pvalue, ttest_result.statistic
            draw_significance(p, t, mean_y, top, bottom, edge_color, ax)


def add_ap_atlas_ordinal(dataframe):
    """ Based on anterior-posterior location, assign a rank to regions
    """

    # Create an ordinal value to order regions anterior-posterior
    # with a gap between atlases
    atlases = sorted(dataframe['atlas'].unique(), reverse=True)
    atlas_adds = {}
    for i, atlas in enumerate(atlases):
        atlas_adds[atlas] = i + 1

    # Create a dataframe of atlases, in order of their mean y
    y_means = dataframe[['atlas', 'mask', 'y']].groupby(['atlas', 'mask']).mean()
    y_means = y_means.sort_values(
        'y', ascending=True
    ).sort_values(
        'atlas', ascending=False
    )

    # Then give each atlas a rank, first in ordered list, then in real data
    y_means['ap_order'] = list(range(len(y_means)))
    dataframe['ap_order'] = dataframe.apply(
        lambda row: (
            y_means.loc[(row['atlas'], row['mask']), 'ap_order']
            + atlas_adds[row['atlas']]
        ),
        axis=1
    )
    return dataframe


def annotate_horizontal_zero(hemi, ax):
    """ """

    # Anterior to the right, for left hemisphere, default
    a_x, a_ha, a_va = 0.99, 'right', 'bottom'
    p_x, p_ha, p_va = 0.01, 'left', 'bottom'
    if hemi == 'rh':
        a_x, a_ha, a_va = 0.01, 'left', 'bottom'
        p_x, p_ha, p_va = 0.99, 'right', 'bottom'

    ax.hlines(y=0.0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
              linestyle=":", color="gray")
    ax.annotate("A", xy=(a_x, 0.5), xycoords="axes fraction", ha=a_ha, va=a_va)
    ax.annotate("P", xy=(p_x, 0.5), xycoords="axes fraction", ha=p_ha, va=p_va)


def annotate_multiple_atlases(voxels, top, ax):
    """ """

    # Label atlas sections
    # Draw the vertical range line
    for atlas in voxels.sort_values('ap_order')['atlas'].unique():
        atlas_min = voxels[voxels['atlas'] == atlas]['ap_order'].min()
        atlas_max = voxels[voxels['atlas'] == atlas]['ap_order'].max()
        atlas_mid = atlas_min + ((atlas_max - atlas_min) / 2.0)
        loci = np.array([
            (atlas_min - 0.50, top * 1.1),
            (atlas_min - 0.50, top * 1.2),
            (atlas_max + 0.50, top * 1.2),
            (atlas_max + 0.50, top * 1.1),
        ])
        actions = [
            path.Path.MOVETO, path.Path.LINETO,
            path.Path.LINETO, path.Path.LINETO,
        ]
        atlas_label_path = path.Path(loci, actions)
        ax.add_patch(patches.PathPatch(
            atlas_label_path, ec='black', zorder=1, fill=False
        ))
        ax.text(
            atlas_mid, top * 1.2, atlas,
            color='black', ha='center', va='bottom',
            transform=ax.transData
        )


def plot_voxels(data, mask, ax, **kwargs):
    sns.scatterplot(
        data=data[mask], x='jitter_y', y='value', ax=ax,
        s=60, zorder=3, **kwargs
    )


def prepare_data(data, winsorize=False):
    """ """
    # Box plots cannot categorize by floats, and NaNs cannot be made into ints,
    # so find unplottable values and remove them.
    # voxels.loc[voxels['mean_y'].isna(), 'mean_y'] = 0.0
    # voxels['mean_y'] = voxels['mean_y'].astype(int)
    # Finally, after modifications are complete, filter only valid voxels for plotting.
    data = data.dropna()

    # Pull back severe outliers, if requested.
    if winsorize:
        pos_val_threshold = np.mean(data['value']) + 3 * np.std(data['value'])
        neg_val_threshold = np.mean(data['value']) - 3 * np.std(data['value'])
        data.loc[data['value'] > pos_val_threshold, 'value'] = pos_val_threshold
        data.loc[data['value'] < neg_val_threshold, 'value'] = neg_val_threshold

    # Two atlases include "HP_tail", but they are plotted separately.
    # For the legend and layout to work, regions must have unique labels.
    # Remove the 'HP_' prefix from three hbt regions to discriminate.
    hbt_filter = data['atlas'] == 'hbt'
    data.loc[hbt_filter, 'mask'] = data.loc[hbt_filter, 'mask'].apply(
        lambda roi: roi.split("_")[-1]
    )

    # Exclude any regions that are not represented bilaterally
    bilateral_regions = [r for r in data['mask'].unique() if (
                         (r in data[data['hemi'] == 'lh']['mask'].unique()) and
                         (r in data[data['hemi'] == 'rh']['mask'].unique()))]
    data = data[data['mask'].isin(bilateral_regions)]

    # Add the anterior-posterior order of regions for evenly spread box plots
    add_ap_atlas_ordinal(data)

    # Add a mean y for each region, based on precise y values
    # y_means = voxels[['hemi', 'mask', 'group_y']].groupby(['hemi', 'mask']).mean()
    # voxels['mean_y'] = voxels.apply(
    #     lambda row: y_means.loc[(row['hemi'], row['mask']), 'group_y'],
    #     axis=1
    # )

    return data


def order_legend_labels(ax, ordered_labels):
    """ """

    orig_handles, orig_labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    assert(len(orig_handles) == len(ordered_labels))
    assert(len(orig_labels) == len(ordered_labels))
    for label in ordered_labels:
        idx = orig_labels.index(label)
        new_handles.append(orig_handles[idx])
        new_labels.append(orig_labels[idx])

    return new_handles, new_labels


def get_title(args, cope=None, atlas=None):
    """ Generate a title for the plot
    """

    plot_title = "Title generation failed."
    if Path(args.input).is_file():
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
    elif Path(args.input).is_dir():
        cope_string = "" if cope is None else f"cope {cope}"
        atlas_string = "" if atlas is None else f" in {atlas} atlas"
        plot_title = f"{args.project.upper()} Subject averages for {cope_string}{atlas_string}"

    return plot_title


def plot_subject(
        voxels,
        title="Voxels",
        draw_voxels=True,
        draw_boxes=False,
        plot_at_mean_ap=False,
        cmap=None,
        mmap=None
):
    """
    """

    # Do we plot at the actual y value, or the evenly spaced group ordinal?
    y_to_plot = 'y' if plot_at_mean_ap else 'ap_order'

    # Get global max and min so both axes have matching ranges
    post_most, ant_most = voxels[y_to_plot].min(), voxels[y_to_plot].max()
    min_value, max_value = voxels['value'].min(), voxels['value'].max()
    top = np.max((np.abs(min_value), np.abs(max_value)))
    bottom = -1.0 * top

    # Jitter the y (in data, x in plot) values to spread points out.
    # Multiply a normal distribution with mean 0 and sd 1 by a fake sd
    # We want to spread them out a bit, but not overlap other y-values
    voxels['jitter_y'] = voxels[y_to_plot] + np.random.randn(len(voxels)) * 0.20

    fig, (left_ax, right_ax) = plt.subplots(ncols=2, sharey='all', figsize=(18, 6))

    # Plot the COPE value at each voxel, colored by region
    if draw_voxels:
        plot_voxels(voxels, voxels['hemi'] == 'lh', left_ax,
                    hue="mask", style='mask', palette=cmap, markers=mmap)
        plot_voxels(voxels, voxels['hemi'] == 'rh', right_ax,
                    hue="mask", style='mask', palette=cmap, markers=mmap)

    # Plot the COPE values box at each region, colored by region
    if draw_boxes:
        plot_boxes(voxels, y_to_plot, 'lh', cmap, top, bottom, left_ax)
        plot_boxes(voxels, y_to_plot, 'rh', cmap, top, bottom, right_ax)

    # Use only one legend, and put it outside the plotting area.
    left_ax.legend().remove()
    legend_handles, legend_labels = order_legend_labels(
        right_ax,
        voxels.sort_values('ap_order', ascending=False)['mask'].unique()
    )
    right_ax.legend(handles=legend_handles, labels=legend_labels,
                    bbox_to_anchor=(1.04, 0.5),
                    loc="center left", borderaxespad=0)
    annotate_horizontal_zero('lh', left_ax)
    annotate_horizontal_zero('rh', right_ax)
    left_ax.set_ylabel("GLM contrast of parameter estimates")
    right_ax.text(0.99, 0.01, '* p<0.1, ** p<0.05, *** p<0.01',
                  color='black', ha='right', va='bottom',
                  transform=right_ax.transAxes)

    for ax in [left_ax, right_ax, ]:
        annotate_multiple_atlases(voxels, top, ax)
        ax.set_xlim((post_most - 1, ant_most + 1))
        ax.set_ylim((1.3 * bottom, 1.3 * top))
        if not plot_at_mean_ap:
            ax.set_xticks([])

    # Make the right side a mirror to the left
    right_ax.invert_xaxis()

    # Throw in some head and tail annotations to ensure clarity
    left_ax.set_xlabel("posterior  -  anterior")
    right_ax.set_xlabel("anterior  -  posterior")
    left_ax.set_title("Left hippocampus and amygdala")
    right_ax.set_title("Right hippocampus and amygdala")

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_group(
        voxels,
        title="Voxels",
        draw_voxels=True,
        draw_boxes=False,
        plot_at_mean_ap=False,
        cmap=None,
        mmap=None
):
    """
    """

    # Do we plot at the actual y value, or the evenly spaced group ordinal?
    y_to_plot = 'group_y' if plot_at_mean_ap else 'ap_order'

    # Jitter the y (in data, x in plot) values to spread points out.
    # Multiply a normal distribution with mean 0 and sd 1 by a fake sd
    # We want to spread them out a bit, but not overlap other y-values
    jitter_randomness = np.abs(np.random.randn(len(voxels))) * 0.20
    group_sign = [1 if dx == 'MDD' else -1 for dx in voxels['dx']]
    voxels['jitter_y'] = voxels[y_to_plot] + (group_sign * jitter_randomness)

    post_most, ant_most = voxels[y_to_plot].min(), voxels[y_to_plot].max()
    min_value, max_value = voxels['value'].min(), voxels['value'].max()
    top = np.max((np.abs(min_value), np.abs(max_value)))
    bottom = -1.0 * top

    # Axes limits are manually set below, do not use sharex or sharey
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18, 12))
    left_ax, right_ax = axes[0, 0], axes[0, 1]
    left_delta_ax, right_delta_ax = axes[1, 0], axes[1, 1]

    # Plot the COPE value at each voxel, colored by region
    if draw_voxels:
        plot_voxels(voxels, (voxels['hemi'] == 'lh'),
                    left_ax, hue='mask', style='mask',
                    palette=cmap, markers=mmap)
        plot_voxels(voxels, (voxels['hemi'] == 'rh'),
                    right_ax, hue='mask', style='mask',
                    palette=cmap, markers=mmap)
        plot_voxels(voxels, (voxels['hemi'] == 'lh') & (voxels['dx'] == 'MDD'),
                    left_delta_ax, color=cmap.get('mdd'), markers=mmap)
        plot_voxels(voxels, (voxels['hemi'] == 'rh') & (voxels['dx'] == 'MDD'),
                    right_delta_ax, color=cmap.get('mdd'), markers=mmap)
        plot_voxels(voxels, (voxels['hemi'] == 'lh') & (voxels['dx'] == 'HV'),
                    left_delta_ax, color=cmap.get('hv'), markers=mmap)
        plot_voxels(voxels, (voxels['hemi'] == 'rh') & (voxels['dx'] == 'HV'),
                    right_delta_ax, color=cmap.get('hv'), markers=mmap)

    # Plot the COPE values box at each region, colored by region
    if draw_boxes:
        plot_boxes(voxels, y_to_plot, 'lh', cmap, top, bottom, left_ax)
        plot_boxes(voxels, y_to_plot, 'rh', cmap, top, bottom, right_ax)
        plot_comparisons(voxels, y_to_plot, 'lh', cmap, top, bottom, left_delta_ax)
        plot_comparisons(voxels, y_to_plot, 'rh', cmap, top, bottom, right_delta_ax)

    # Use only one legend, and put it outside the plotting area.
    left_ax.legend().remove()
    left_delta_ax.legend().remove()
    legend_handles, legend_labels = order_legend_labels(
        right_ax,
        voxels.sort_values('ap_order', ascending=False)['mask'].unique()
    )
    right_ax.legend(handles=legend_handles, labels=legend_labels,
                    bbox_to_anchor=(1.10, 0.5),
                    loc="center left", borderaxespad=0)
    right_delta_ax.legend(bbox_to_anchor=(1.10, 0.5),
                          loc="center left", borderaxespad=0)
    right_delta_ax.text(0.99, 0.01, '* p<0.1, ** p<0.05, *** p<0.01',
                        color='black', ha='right', va='bottom',
                        transform=right_delta_ax.transAxes)

    for ax in [left_ax, right_ax, left_delta_ax, right_delta_ax]:
        if len(voxels['atlas'].unique()) > 1:
            annotate_multiple_atlases(voxels, top, ax)
        ax.set_xlim((post_most - 1, ant_most + 1))
        ax.set_ylim((1.3 * bottom, 1.3 * top))
        if not plot_at_mean_ap:
            ax.set_xticks([])

    for ax in [left_ax, left_delta_ax, ]:
        # ax.yaxis.tick_left()
        ax.set_ylabel("GLM contrast of parameter estimates")
        annotate_horizontal_zero('lh', ax)

    # Expand the x-axis limits just a bit to make way for head/tail annotations
    for ax in [right_ax, right_delta_ax, ]:
        ax.invert_xaxis()
        # ax.yaxis.tick_right()
        ax.set_ylabel("")
        annotate_horizontal_zero('rh', ax)

    # Throw in some head and tail annotations to ensure clarity
    left_ax.set_xlabel("")
    left_delta_ax.set_xlabel("posterior  -  anterior")
    right_ax.set_xlabel("")
    right_delta_ax.set_xlabel("anterior  -  posterior")

    num_participants = len(voxels['subject'].unique())
    left_ax.set_title(f"Left hippocampus and amygdala (combined, n={num_participants})")
    right_ax.set_title(f"Right hippocampus and amygdala (combined, n={num_participants})")
    left_delta_ax.set_title("Left hippocampus and amygdala (MDD vs HV)")
    right_delta_ax.set_title("Right hippocampus and amygdala (MDD vs HV)")

    fig.suptitle(title)
    fig.tight_layout()

    # print(f"Data limits {(min_x, max_x)} {(min_value, max_value)}")
    # print(f"One axes shapes {left_ax.get_xlim()} {left_ax.get_ylim()}")

    return fig


def main(args):
    """ Entry point """

    input_path = Path(args.input)
    if input_path.is_file():
        voxels = parse_masks(pd.read_csv(input_path))
        voxels = prepare_data(voxels, winsorize=args.winsorize)
        figure = plot_subject(
            voxels,
            title=get_title(args),
            draw_voxels=(not args.skip_voxels),
            draw_boxes=args.draw_boxes,
            plot_at_mean_ap=args.plot_at_mean_ap,
            cmap=get_color_map(voxels, alpha=0.5),
            mmap=get_marker_map(voxels),
        )
        # Plot the data and save the plot
        if args.verbose:
            print(f"Writing figure to {args.output}")
        figure.savefig(args.output)
    elif input_path.is_dir():
        group_map = get_group_map(args.group_file)
        voxels = get_subject_means(input_path, args.mask_csv_subdir, group_map, verbose=args.verbose)
        if args.cope == 0:
            copes = sorted(voxels['cope'].unique())
        else:
            copes = [args.cope, ]
        if args.atlas == "":
            atlases = sorted(voxels['atlas'].unique())
        else:
            atlases = [args.atlas, ]
        for cope in copes:
            cope_idx = voxels[voxels['cope'] == cope].index
            voxels_to_plot = prepare_data(
                voxels.loc[cope_idx, :], winsorize=args.winsorize
            )
            figure = plot_group(
                voxels_to_plot,
                title=get_title(args, cope=cope),
                draw_voxels=(not args.skip_voxels),
                draw_boxes=args.draw_boxes,
                plot_at_mean_ap=args.plot_at_mean_ap,
                cmap=get_color_map(voxels_to_plot, alpha=0.5),
                mmap=get_marker_map(voxels_to_plot),
            )
            # Plot the data and save the plot
            winsor_str = '_w' if args.winsorize else ''
            filename = f"group_cope-{str(cope)}{winsor_str}.png"
            if args.verbose:
                print(f"Writing figure {filename} to {args.output}")
            figure.savefig(Path(args.output) / filename)
            for atlas in atlases:
                atlas_idx = voxels[voxels['atlas'] == atlas].index
                combined_idx = cope_idx.intersection(atlas_idx)
                voxels_to_plot = prepare_data(
                    voxels.loc[combined_idx, :], winsorize=args.winsorize
                )
                figure = plot_group(
                    voxels_to_plot,
                    title=get_title(args, cope=cope, atlas=atlas),
                    draw_voxels=(not args.skip_voxels),
                    draw_boxes=args.draw_boxes,
                    plot_at_mean_ap=args.plot_at_mean_ap,
                    cmap=get_color_map(voxels_to_plot, alpha=0.5),
                    mmap=get_marker_map(voxels_to_plot),
                )
                # Plot the data and save the plot
                filename = f"group_cope-{str(cope)}_atlas-{atlas}{winsor_str}.png"
                if args.verbose:
                    print(f"Writing figure {filename} to {args.output}")
                figure.savefig(Path(args.output) / filename)

    else:
        print("The specified input is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main(get_arguments())
