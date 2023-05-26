#!/usr/bin/env python3

# extract_masked_ts.py
#
# This script will input bold data and binary masks and confounds,
# and will use them to extract mask-bound timeseries from cleaned
# bold data.

import re
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from pathlib import Path
from nilearn.maskers import NiftiMasker
from nilearn.image import index_img

# Trigger printing in red to highlight problems
red_on = '\033[91m'
green_on = '\033[92m'
color_off = '\033[0m'
err = f"{red_on}ERROR: {color_off}"


def get_arguments():
    """ Parse command line arguments """

    usage_str = """
        Typical usage:

        extract_masked_ts.py \\
        --bold-file /path/to/sub-U03280_STUFF_desc-preproc_bold.nii.gz \\
        --mask-files /path/to/masks/tpl-STUFF_dseg.nii.gz \\
        --confounds-file /path/to/regressors/mems/run1/sub-U03280_STUFF.tsv \\
        --output-dir /path/to/timeseries \\
        --trim-first-trs 7 \\
        --verbose

        That command would load the specified BOLD file, and trim the first 7
        TRs from the beginning of it. It would then smooth at
        5mm fwhm, hi-pass filters at 0.01Hz, and residualizes the timeseries
        with different confounds regressor collections. Finally, it would
        extract the timeseries from each voxel in each ROI, average across
        voxels within-ROI, and save the results in the output-dir.
    """

    parser = argparse.ArgumentParser(
        description="Regress confounds from a BOLD file and extract ts data.",
        usage=usage_str,
    )
    parser.add_argument(
        "--bold-file", "--bold_file", required=True,
        help="The 4D BOLD file containing fMRIPrep-preprocessed data.",
    )
    parser.add_argument(
        "--mask-files", "--mask_files", nargs="+",
        help="Specify as many masks as desired, each with an ROI",
    )
    parser.add_argument(
        "--confounds-file", "--confounds_file",
        help="A cropped and pruned version of the fMRIPrep confounds.",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default=".",
        help="Specify where to write output, defaults to current path.",
    )
    parser.add_argument(
        "--smooth", type=float, default=5.0,
        help="Specify full-width half-maximum of smoothing kernel.",
    )
    parser.add_argument(
        "--prob-threshold", type=float, default=0.50,
        help="Specify threshold if mask is probabilistic.",
    )
    parser.add_argument(
        "--trim-first-trs", "--trim_first_trs", type=int, default=0,
        help="Specify how many TRs to trim from the beginning of each run.",
    )
    parser.add_argument(
        "--save-correlations", action="store_true",
        help="set to write out correlation matrices for all voxels tses",
    )
    parser.add_argument(
        "--save-all-voxels", action="store_true",
        help="set to write out matrices with all voxels' tses per region",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    # Parse and validate arguments
    args = parser.parse_args()
    ok_to_run = True

    # Figure out the subject, session, and task.
    # We would also like run, but need to match even for task-study w/o run-#
    if Path(args.bold_file).exists():
        setattr(args, "bold_file", Path(args.bold_file))
        match = re.compile(r"sub-([A-Z][0-9]+)").search(str(args.bold_file))
        if match:
            setattr(args, "subject", match.group(1))
        else:
            setattr(args, "subject", "unknown")
            print(f"Found BOLD, '{str(args.bold_file)}' "
                  "but couldn't determine subject.")
        match = re.compile(r"ses-([0-9]+)").search(str(args.bold_file))
        if match:
            setattr(args, "session", match.group(1))
        else:
            setattr(args, "session", "unknown")
        match = re.compile(r"task-([a-z]+)").search(args.bold_file.name)
        if match:
            setattr(args, "task", match.group(1))
        else:
            setattr(args, "task", "unknown")
        match = re.compile(r"_run-([0-9]+)").search(args.bold_file.name)
        if match:
            setattr(args, "run", int(match.group(1)))
        else:
            print(f"  no run for {args.bold_file.name}")
    else:
        print(f"Could not find BOLD file at '{args.bold_file}'.")
        ok_to_run = False

    # Establish mask file names to match
    patterns = [
        # Schaefer2018 regions
        re.compile(r"_roi-([A-Za-z]+)_([a-z]+)\."),
        # FreeSurfer aseg regions (with Right_/Left_ indicating laterality)
        re.compile(r"res-bold_aseg_([A-Za-z]+)_mask\.T1\.nii\.gz"),
        # FreeSurfer second-level regions (with .lh/.rh indicating laterality)
        re.compile(r"res-bold_([a-z0-9]+)_([A-Za-z]+)_mask\.T1\.([a-z]+)\.nii\.gz"),
    ]
    masks = {}
    # Use Path objects rather than strings, and make sure they exist.
    for mask_file in [Path(_) for _ in args.mask_files]:
        if mask_file.exists():
            print(f"Mask path '{str(mask_file)}' exists.")
            mask_img = nib.load(mask_file)
            # If the mask is probabilistic, make it binary.
            lbls, ns = np.unique(mask_img.get_fdata(), return_counts=True)
            if len(lbls) > 2:
                bin_mask = np.zeros(mask_img.shape)
                bin_mask[mask_img.get_fdata() > args.prob_threshold] = 1
                mask_img = nib.Nifti1Image(
                    bin_mask, mask_img.affine, dtype=np.uint16,
                )
            for pattern in patterns:  # only one should match
                match = pattern.search(mask_file.name)
                if match:
                    print(f"{str(mask_file.name)} matched")
                    if len(match.groups()) == 1:
                        # This is an aseg region
                        if "Right" in mask_file.name:
                            roi, lat = match.group(1), "rh"
                        elif "Left" in mask_file.name:
                            roi, lat = match.group(1), "lh"
                        else:
                            roi, lat = match.group(1), "bi"
                    elif len(match.groups()) == 2:
                        # This is a Schaefer2018 region
                        roi, lat = match.group(1), match.group(2)
                    else:
                        # Must be a FreeSurfer second-level segmentation
                        roi, lat = match.group(2), match.group(3)
                    print(f"Key is '{roi}_{lat}'")

                    # Store the final binary mask
                    print(f"Storing mask {roi}_{lat}")
                    masks[f"{roi}_{lat}"] = mask_img
                else:
                    pass
        else:
            print(f"{err}Ignoring mask '{str(mask_file)}'; does not exist.")
    if len(masks.keys()) > 0:
        setattr(args, "masks", masks)
    else:
        print(f"{err}There are no regions to extract!")
        ok_to_run = False

    if Path(args.confounds_file).exists():
        print(f"Path '{args.confounds_file}' exists.")
        setattr(args, "confounds_file", Path(args.confounds_file))
        setattr(args, "confounds_data",
                pd.read_csv(args.confounds_file, sep='\t', index_col=None))
    else:
        print(f"{err}Path '{args.confounds_file}' does not exist.")
        ok_to_run = False

    if Path(args.output_dir).exists():
        print(f"Path '{args.output_dir}' exists.")
        setattr(args, "output_dir", Path(args.output_dir))
        setattr(args, "subject_dir",
                Path(args.output_dir) / f"sub-{args.subject}")
    else:
        print(f"{err}Path to output, '{args.output_dir}' does not exist.")
        ok_to_run = False

    if not ok_to_run:
        sys.exit(1)

    return args


def run_all_confounds(bold_img, confounds, roi_masks, tr_res, smooth):
    """ With a given mask, extract timeseries with different cleaning strategies

    We extract the timeseries from voxels within this binary mask.
    We could have done it all at once with a labelled mask,
    but using these thresholded binary masks seems to be a better
    way to treat each region independently, rather than having
    each region compete for voxel-hood in a nearest-neighbor
    resampling. Both approaches would be very similar,
    maybe identical at higher thresholds.

    We also standardize the data extracted from the mask.
    This simply zero-means the data, and scales to unit variance.
    By default, confounds are also standardized, which we accept.
    Additional parameters (filtering, confound strategy) are
    deployed as specified in each NiftiMasker constructor below.

    """

    rois = []
    for roi_name, roi_mask in roi_masks.items():
        num_voxels = int(np.sum(roi_mask.get_fdata() > 0))
        if num_voxels == 0:
            print(f"Masking an empty mask is pointless! skipping {roi_name}")
            continue
        print(f"Masking {num_voxels:,} voxels from {roi_name}.")

        # Extract the timeseries from the masks, and store everything in a dict.
        masker = NiftiMasker(
            mask_img=roi_mask, standardize=True,
            smoothing_fwhm=smooth, high_pass=0.01, t_r=tr_res,
        )
        ts = masker.fit_transform(bold_img, confounds=confounds)

        rois.append({
            "roi": roi_name.split("_")[0],
            "hemi": roi_name.split("_")[1],
            "mask": roi_mask,
            "num_voxels": num_voxels,
            "masker": masker,
            "timeseries": ts,
        })

    return rois


def main(args):
    """ Entry point """

    ts_dir = args.subject_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"  reading BOLD data for {args.subject}")
    bold_img = nib.load(args.bold_file)
    tr_res = bold_img.header['pixdim'][4]

    # Crop off the first TRs from both data and confounds
    if args.trim_first_trs > 0:
        if args.verbose:
            print(f"  cropping first {args.trim_first_trs} volumes "
                  f"from {args.bold_file.name}.")
        cropped_bold_img = index_img(
            bold_img, slice(args.trim_first_trs, None)
        )
    else:
        cropped_bold_img = bold_img

    # Determine whether to alter confounds to match BOLD
    extra_trs = len(args.confounds_data) - cropped_bold_img.shape[3]
    if extra_trs > 0:
        if args.verbose:
            print(f"  cropping {extra_trs} confounds from fMRIPrep")
            if extra_trs != args.trim_first_trs:
                print("  WARNING: "
                      f"this differs from {args.trim_first_trs} specified.")
        cropped_confounds = args.confounds_data.iloc[extra_trs:, :]
    else:
        if args.verbose:
            print(f"  confounds match BOLD, {cropped_bold_img.shape[3]} TRs.")
        cropped_confounds = args.confounds_data

    mask_dicts = run_all_confounds(
        cropped_bold_img, cropped_confounds, args.masks, tr_res, args.smooth
    )
    for mask_dict in mask_dicts:
        ts = mask_dict['timeseries']
        ts_file = "sub-{}_task-{}_{}roi-{}_hemi-{}_{}".format(
            args.subject, args.task,
            f"run-{args.run}_" if "run" in args else "",
            mask_dict['roi'], mask_dict['hemi'], "ts.tsv"
        )
        if args.save_all_voxels:
            np.savetxt(
                ts_dir / ts_file, ts, fmt="%0.5f", delimiter='\t'
            )
        if ts.shape[1] > 1:
            # Average the voxels within the mask.
            mean_ts_file = ts_file.replace("_ts.", "_mean_ts.")
            np.savetxt(
                ts_dir / mean_ts_file,
                np.mean(ts, axis=1), fmt="%0.5f", delimiter='\t'
            )

            if args.save_correlations:
                # Calculate correlations between a subsample of in-mask voxels
                sample_idx = np.random.randint(
                    np.max([512, ts.shape[1]]), size=512
                )
                corr_mat = np.corrcoef(ts[:, sample_idx], rowvar=False)
                corr_idx = np.tril_indices(corr_mat.shape[0], k=-1)
                corrs = corr_mat[corr_idx]
                if args.verbose:
                    if np.sum(corrs == 1.0) > 0:
                        z_corrs = np.concatenate((
                            np.arctanh(corrs[corrs < 1.0]), corrs[corrs == 1.0],
                        ))
                    else:
                        z_corrs = np.arctanh(corrs)
                    mean_r = np.tanh(np.mean(z_corrs))
                    print(f"      {mask_dict['roi']} {mask_dict['hemi']} "
                          f"raw mean voxel-wise r == {mean_r:0.2f}")
                # These matrices can be gigabytes and take forever to save.
                np.savetxt(
                    ts_dir / ts_file.replace("ts.tsv", "r.tsv"),
                    corrs, fmt="%0.5f", delimiter='\t'
                )
        else:
            if args.verbose:
                print("      has one voxel, no r")

    with open(args.subject_dir / "last_run.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main(get_arguments())
