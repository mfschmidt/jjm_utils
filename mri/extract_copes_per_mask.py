#!/usr/bin/env python3

# extract_copes_per_mask.py

import sys
import re
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="For a given feat results path, find masks in a "
                    "./masks/ subdirectory, and extract beta values "
                    "from each ./stats/cope*.nii.gz image from within "
                    "each mask. Results will be stored in an excel "
                    "spreadsheet for each cope image.",
    )
    parser.add_argument(
        "featpath",
        help="The path to a feat results directory",
    )
    parser.add_argument(
        "--cope",
        help="only run on the specified cope file",
    )
    parser.add_argument(
        "--mask",
        help="only run on the specified mask file",
    )
    parser.add_argument(
        "--mask-path",
        help="where to store the newly created masks",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.9,
        help="thresholds for inclusion in probabilistic masks",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Ensure the feat directory exists
    args.featpath = Path(args.featpath).resolve()
    if args.featpath.exists():
        if args.verbose:
            print(f"Path is '{args.featpath}'")
    else:
        print(f"Path '{args.featpath}' does not exist.")
        sys.exit(1)

    if hasattr(args, "mask_path"):
        setattr(args, "mask_path",
                Path(args.mask_path).resolve())
    else:
        setattr(args, "mask_path",
                (args.featpath / ".." / "masks").resolve())
    
    setattr(args, "cope_path",
            (args.featpath / "stats").resolve())

    # Figure out the subject ID from the path
    pattern = re.compile(r"sub-([A-Z][0-9]+)")
    match = pattern.search(str(args.featpath))
    if match:
        setattr(args, "subject_id", match.group(1))
    else:
        if ".feat" in str(args.featpath):
            setattr(args, "subject_id",
                    args.featpath.parent.name.replace("sub-", ""))
        else:
            setattr(args, "subject_id",
                    args.featpath.parent.parent.parent.parent.name.replace("sub-", ""))

    if args.verbose:
        print(f"Subject is '{args.subject_id}'.")
        print(f"Masks are at '{args.mask_path}'.")
        print(f"Copes are at '{args.cope_path}'.")

    # Clean up cope and mask arguments by accepting multiple specifications.
    cope_attempts = []
    # if args.cope is None, we'll just find and run them all.
    if args.cope is not None:
        cope = Path(args.cope)  # first try
        cope_attempts.append(cope)
        if not cope.exists():
            cope = args.cope_path / args.cope
            cope_attempts.append(cope)
        if not cope.exists():
            cope = args.cope_path / f"{args.cope}.nii.gz"
            cope_attempts.append(cope)
        if not cope.exists():
            cope = args.cope_path / f"cope{args.cope}.nii.gz"
            cope_attempts.append(cope)
        if not cope.exists():
            print(f"Cope '{args.cope}' cannot be found.")
            if args.verbose:
                for attempt in cope_attempts:
                    print(f"  could not find '{attempt}'")
            sys.exit(1)
        args.cope = cope

    mask_attempts = []
    # if args.mask is None, we'll just find and run them all.
    if args.mask is not None:
        mask = Path(args.mask)  # first try
        mask_attempts.append(mask)
        if not mask.exists():
            mask = args.mask_path / args.mask
            mask_attempts.append(mask)
        if not mask.exists():
            mask = args.mask_path / f"{args.mask}.nii.gz"
            mask_attempts.append(mask)
        if not mask.exists():
            candidates = list(args.mask_path.glob(f"*{args.mask}*.nii.gz"))
            if len(candidates) == 1:
                mask = candidates[0]
                mask_attempts.append(mask)
            elif len(candidates) > 1:
                print(f"{len(candidates)} masks match '{args.mask}'.")
                args.mask = candidates
                return args
        if not mask.exists():
            print(f"Mask '{args.mask}' cannot be found.")
            if args.verbose:
                for attempt in mask_attempts:
                    print(f"  could not find '{attempt}'")
            sys.exit(1)
        args.mask = mask

    return args


def extract_voxels_by_label(data, mask, label):
    """ Extract voxels from input where the value in mask matches label
    """

    if data.shape != mask.shape:
        raise ValueError("Data {data.shape} and mask {mask.shape} don't match")

    voxels = []
    dims = data.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if mask[x][y][z] == label:
                    voxels.append({
                        "x": x, "y": y, "z": z,
                        "value": data[x][y][z],
                        "label": mask[x][y][z],
                    })

    return pd.DataFrame(voxels)


def extract_voxels_by_threshold(data, mask, threshold):
    """ Extract voxels from input where the value in mask > threshold.
    """

    if data.shape != mask.shape:
        raise ValueError("Data {data.shape} and mask {mask.shape} don't match")

    voxels = []
    dims = data.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if mask[x][y][z] > threshold:
                    voxels.append({
                        "x": x, "y": y, "z": z,
                        "value": data[x][y][z],
                        "label": mask[x][y][z],
                    })

    return pd.DataFrame(voxels)


def main(args):
    """ Entry point """

    if args.verbose:
        print("Extracting betas from: {}".format(
            "all copes" if args.cope is None else str(args.cope),
        ))
        print("      masked @{:0.2f} by: {}".format(
            args.threshold,
            "all masks" if args.mask is None else str(args.mask),
        ))
        print("                   in: {}".format(
            str(args.featpath.resolve()),
        ))

    # What feat run are we dealing with? Different levels, different patterns.
    is_high_level = False
    feat_id = args.featpath.resolve().parent.name.split('.')[0]
    if ".gfeat" in str(args.featpath.resolve()):
        is_high_level = True
        pattern = re.compile(r"/([^/]+).gfeat")
        match = pattern.search(str(args.featpath.resolve()))
        if match:
            feat_id = match.group(1)

    if args.mask is None:
        masks = sorted(args.mask_path.glob("res-bold_*.nii.gz"))
        print(f"Found {len(masks)} masks in {str(args.mask_path)}:")
    elif isinstance(args.mask, list):
        masks = args.mask
    else:
        masks = [args.mask, ]
    if args.verbose:
        for mask in masks:
            print(f"  mask '{mask}'")

    if args.cope is None:
        copes = list(args.cope_path.glob("cope*.nii.gz"))
        copes.sort(key=lambda x: int(str(x.name)[4: -7]))
        print(f"Found {len(copes)} copes in {str(args.cope_path)}:")
    else:
        copes = [args.cope, ]
    if args.verbose:
        for cope in copes:
            print(f"  cope '{cope}'")

    # Ensure the path for writing exists.
    out_path = (
        args.mask_path /
        f"{feat_id}_t-{args.threshold:0.2f}"
    )
    out_path.mkdir(exist_ok=True)

    # For all specified copes and all specified masks, combine them and average
    if len(copes) == 0:
        print(f"No copes found at {args.cope_path}")
        return 0

    all_data = []
    for cope in copes:
        voxelwise_dataframes = []
        cope_img = nib.load(str(cope.resolve()))
        cope_name = cope.name
        if is_high_level:
            # High level gfeat runs call every file cope1.feat, be more specific.
            cope_name = cope.parent.parent.name.replace(".feat", "")
            print(f"getting name {cope.parent.parent.name} from {str(cope.resolve())}")
        cope_data = cope_img.get_fdata()
        for mask in masks:
            mask_img = nib.load(str(mask.resolve()))
            mask_data = mask_img.get_fdata()
            if args.verbose:
                print(f"Applying {mask.name} to {cope.name} (named {cope_name})...")
                print("  cope is {} x {} x {}".format(
                    cope_data.shape[0], cope_data.shape[1], cope_data.shape[2],
                ))
                print("  mean cope is {:0.2f}, from {:0.2f} to {:0.2f}".format(
                    np.mean(cope_data), np.min(cope_data), np.max(cope_data),
                ))
                print("  mask is {} x {} x {}".format(
                    mask_data.shape[0], mask_data.shape[1], mask_data.shape[2],
                ))

            lbls, ns = np.unique(mask_data, return_counts=True)
            if len(lbls) == 1:
                # An empty mask (all 0) has little value
                if args.verbose:
                    print("  mask is useless, all values are {ns[0]}")
            elif len(lbls) == 2:
                if args.verbose:
                    print(f"  mask looks binary ({len(lbls)} labels)")
                    print(f"    {ns[0]:,} {lbls[0]} and {ns[1]:,} {lbls[1]}")
                for lbl in lbls:
                    if lbl != 0:
                        voxels = extract_voxels_by_label(
                            cope_data, mask_data, lbl
                        )
                        voxels['cope'] = cope_name
                        voxels['mask'] = mask.name[: -7]
                        voxelwise_dataframes.append(voxels)
            elif len(lbls) >= 20:
                # Currently, the smallest probabilistic atlas contains
                # 37 voxels. The largest labelled atlas contains 19 regions.
                if args.verbose:
                    print(f"  mask looks probabilistic ({len(lbls)} labels)")

                voxels = extract_voxels_by_threshold(
                    cope_data, mask_data, args.threshold
                )
                if len(voxels) > 0:
                    if args.verbose:
                        print(f"  found {len(voxels)} voxels")
                else:
                    if args.verbose:
                        print(f"  No voxels meet {args.threshold} threshold!!")
                    # Even with no voxels, the empty mask still must be
                    # represented in the dataset, so make a placeholder.
                    voxels = pd.DataFrame([{
                        'label': lbls[0], 'value': np.NaN,
                        'x': np.NaN, 'y': np.NaN, 'z': np.NaN,
                    }, ])
                voxels['cope'] = cope_name
                voxels['mask'] = mask.name[: -7]
                voxelwise_dataframes.append(voxels)
            else:
                if args.verbose:
                    print(f"  mask looks like {len(lbls)} numeric labels")
                for lbl, n in zip(lbls, ns):
                    if args.verbose:
                        print(f"    mask label {lbl} has {n} voxels")
                    if lbl != 0:
                        voxels = extract_voxels_by_label(
                            cope_data, mask_data, lbl
                        )
                        voxels['cope'] = cope_name
                        voxels['mask'] = mask.name[: -7]
                        voxelwise_dataframes.append(voxels)
            if args.verbose and ('value' in voxels):
                print("  masked cope is {:0.2f}, from {:0.2f} to {:0.2f}".format(
                    np.mean(voxels['value']),
                    np.min(voxels['value']),
                    np.max(voxels['value']),
                ))

        # Save out complete set of all masked voxels
        voxelwise_dataframe = pd.concat(voxelwise_dataframes)
        all_data.append(voxelwise_dataframe)
        voxelwise_dataframe['subject'] = args.subject_id
        voxelwise_dataframe[
            ['subject', 'cope', 'mask', 'value', 'label', 'x', 'y', 'z', ]
        ].sort_values(['cope', 'mask', ]).to_csv(
            out_path / f"sub-{args.subject_id}_cope-{cope_name}_voxels_by_masks.csv",
            index=None
        )

    # Save out summarized stats from masked voxels
    all_data_for_summary = pd.concat(all_data)
    # Use cope_num to sort, then drop cope and replace it after sorting
    all_data_for_summary['cope_num'] = all_data_for_summary['cope'].apply(
        lambda x: int(x[4:])
    )
    all_data_for_summary = all_data_for_summary.sort_values(
        ['cope_num', 'mask', ]
    )
    grouping = all_data_for_summary[['cope_num', 'mask', 'value', ]].groupby(
        ['cope_num', 'mask', ]
    )['value']
    counts = grouping.count()
    counts.name = 'n'
    means = grouping.mean()
    means.name = 'mean'
    sds = grouping.std()
    sds.name = 'sd'

    # While sorted by cope_num and mask, lock in the index.
    stats_dataframe = pd.concat([means, sds, counts, ], axis=1).reset_index()
    stats_dataframe['subject'] = args.subject_id
    stats_dataframe['cope'] = stats_dataframe['cope_num'].apply(
        lambda x: f"cope{x}"
    )

    # But we saved empty masks, and don't want them to say they have 1 voxel.
    stats_dataframe['n'] = stats_dataframe.apply(
        lambda row: 0 if np.isnan(row['mean']) else row['n'], axis=1,
    )

    # Save to disk; append rather than overwriting
    summary_file = out_path / f"sub-{args.subject_id}_copes_by_masks.csv"
    if summary_file.exists():
        existing_data = pd.read_csv(summary_file)
        stats_dataframe = pd.concat([existing_data, stats_dataframe])
    stats_dataframe[
        ['subject', 'cope', 'mask', 'n', 'mean', 'sd', ]
    ].sort_index().to_csv(
        summary_file, index=None
    )

    print(stats_dataframe)


if __name__ == "__main__":
    main(get_arguments())
