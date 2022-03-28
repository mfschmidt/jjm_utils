#!/usr/bin/env python3

# extract_copes_per_mask.py

import sys
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
        "-t", "--threshold", type=float, default=0.9,
        help="thresholds for inclusion in probabilistic masks",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Ensure the feat directory exists
    args.featpath = Path(args.featpath)
    if not args.featpath.exists():
        print(f"Path '{args.featpath}' does not exist.")
        sys.exit(1)

    setattr(args, "mask_path",
            (args.featpath / ".." / "masks").resolve())
    setattr(args, "cope_path",
            (args.featpath / "stats").resolve())

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
    """ Extract voxels from input where the value in mask >= threshold.
    """

    if data.shape != mask.shape:
        raise ValueError("Data {data.shape} and mask {mask.shape} don't match")

    voxels = []
    dims = data.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if mask[x][y][z] >= threshold:
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

    if args.mask is None:
        masks = sorted(args.mask_path.glob("res-bold_*.nii.gz"))
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
    else:
        copes = [args.cope, ]
    if args.verbose:
        for cope in copes:
            print(f"  cope '{cope}'")

    # Ensure the path for writing exists.
    out_path = args.mask_path / f"{args.featpath.name}_t-{args.threshold:0.2f}"
    out_path.mkdir(exist_ok=True)

    # For all specified copes and all specified masks, combine them and average
    all_data = []
    for cope in copes:
        voxelwise_dataframes = []
        cope_img = nib.load(cope)
        cope_data = cope_img.get_fdata()
        for mask in masks:
            mask_img = nib.load(mask)
            mask_data = mask_img.get_fdata()
            if args.verbose:
                print(f"Applying {mask.name} to {cope.name}...")
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
                if args.verbose:
                    print("  mask is useless, all values are {ns[0]}")
            elif len(lbls) == 2:
                if args.verbose:
                    print("  mask looks binary")
                    print(f"    {ns[0]:,} {lbls[0]} and {ns[1]:,} {lbls[1]}")
                for lbl in lbls:
                    if lbl != 0:
                        voxels = extract_voxels_by_label(
                            cope_data, mask_data, lbl
                        )
                        voxels['cope'] = cope.name[: -7]
                        voxels['mask'] = mask.name[: -7]
                        voxelwise_dataframes.append(voxels)
            elif len(lbls) >= 20:
                # Currently, the smallest probabilistic atlas contains
                # 37 voxels. The largest labelled atlas contains 19 regions.
                if args.verbose:
                    print("  mask looks probabilistic")
                voxels = extract_voxels_by_threshold(
                    cope_data, mask_data, args.threshold
                )
                voxels['cope'] = cope.name[: -7]
                voxels['mask'] = mask.name[: -7]
                voxelwise_dataframes.append(voxels)
            else:
                if args.verbose:
                    print("  mask looks like numeric labels")
                for lbl, n in zip(lbls, ns):
                    if args.verbose:
                        print(f"    mask label {lbl} has {n} voxels")
                    if lbl != 0:
                        voxels = extract_voxels_by_label(
                            cope_data, mask_data, lbl
                        )
                        voxels['cope'] = cope.name[: -7]
                        voxels['mask'] = mask.name[: -7]
                        voxelwise_dataframes.append(voxels)

        # Save out complete set of all masked voxels
        voxelwise_dataframe = pd.concat(voxelwise_dataframes)
        all_data.append(voxelwise_dataframe)
        voxelwise_dataframe.to_csv(
            out_path / f"{cope.name[: -7]}_masked_voxels.csv",
            index=None
        )

    # Save out summarized stats from masked voxels
    all_data_for_summary = pd.concat(all_data)
    grouping = all_data_for_summary[['cope', 'mask', 'value', ]].groupby(
        ['cope', 'mask', ]
    )['value']
    means = grouping.mean()
    means.name = 'mean'
    sds = grouping.std()
    sds.name = 'sd'
    stats_dataframe = pd.concat([means, sds, ], axis=1)
    stats_dataframe.to_csv(
        out_path / "cope_stats_by_mask.csv",
    )

    print(stats_dataframe)


if __name__ == "__main__":
    main(get_arguments())
