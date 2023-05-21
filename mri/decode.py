#!/usr/bin/env python3

# decode.py

import sys
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from nilearn.signal import clean


"""
The main kernel of decoding is on one run with one mask/weights pair.
So the 'for each subject, for each run' stuff should happen in the shell,
where each iteration should execute this file. The current implementation
uses the 'decode_everything_with_python.sh' script alongside this one.
Note that this script uses several python libraries that must be
installed for successful execution, too. It would be best to activate
a virtual environment before running the shell script.

The original matlab decoder also spends a lot of lines of code on picking
out the TRs of interest. This decoder doesn't care. It will just decode
the entire BOLD file start to finish. The user can pick out their own
blocks/periods/trials however they like.

"""

# Trigger printing in red to highlight problems
red_on = '\033[91m'
green_on = '\033[92m'
color_off = '\033[0m'
err = f"{red_on}ERROR: {color_off}"


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Apply a decoder (mask and weights) to a 4D BOLD file.",
    )
    parser.add_argument(
        "bold_file",
        help="The 4D data, already pre-processed, cropped, smoothed, filtered",
        # Future upgrade would be to accept an fMRIPrep output file, then
        # do the crop, smooth, filter here to save the Feat step
        # and the space from the extra files Feat writes.
        # But for right now, this simply replaces the matlab version
        # in a way that saves me time writing debug info for visuals.
    )
    parser.add_argument(
        "decoder_mask",
        help="A 3D mask, in the same space as the 'bold_file'",
    )
    parser.add_argument(
        "decoder_weights",
        help="A 1D vector, with one scalar value per voxel in 'decoder_mask'",
    )
    parser.add_argument(
        "confounds",
        help="A tsv file, w/ header, usually extracted from fMRIPrep confounds",
        # A potential upgrade is accepting full fMRIPrep confounds and a spec
        # for which confounds to pull out of it and how many TRs to crop off
        # the top of it (which would match the non-steady-state TRs in BOLD).
        # This script would then do the crop/extract without the extra file.
    )
    parser.add_argument(
        "--name",
        help="override the decoder name with this string, for naming outputs",
    )
    parser.add_argument(
        "--output-path", default=".",
        help="write output files here, rather than in the current directory",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    setattr(args, "bold_file", Path(args.bold_file).resolve())
    setattr(args, "decoder_mask", Path(args.decoder_mask).resolve())
    setattr(args, "decoder_weights", Path(args.decoder_weights).resolve())
    setattr(args, "confounds", Path(args.confounds).resolve())
    setattr(args, "output_path", Path(args.output_path).resolve())

    return args


def validated(args):
    """ Validate arguments """

    if args.verbose:
        print(f"Running from {str(Path.cwd())}")

    we_have_a_fatal_error = False
    for p, desc in [
        (args.bold_file, 'bold_file'),
        (args.decoder_mask, 'decoder_mask'),
        (args.decoder_weights, 'decoder_weights'),
        (args.confounds, 'confounds'),
    ]:
        if p.exists():
            if args.verbose:
                print(f"Path '{str(p)}' exists.")
        else:
            print(f"{err}Path '{str(p)}' does not exist.")
            we_have_a_fatal_error = True

    # Store paths as Path objects rather than strings
    args.output_path.mkdir(parents=True, exist_ok=True)

    # If a name was not provided, make a good guess
    if "name" not in args or args.name is None:
        loc_underscore = args.decoder_mask.name.find("_mask")
        setattr(args, "name", args.decoder_mask.name[0:loc_underscore])
        if args.verbose:
            print(f"detected '{args.name}' as decoder name")
    else:
        if args.verbose:
            print(f"using provided '{args.name}' as decoder name")

    if we_have_a_fatal_error:
        sys.exit(1)

    return args


def load_and_mask_bold(args):
    """ Load the 4D bold data, and mask it with the decoder mask. """

    bold_img = nib.load(args.bold_file)
    mask_img = nib.load(args.decoder_mask)

    assert(bold_img.get_fdata()[:, :, :, 0].shape == mask_img.shape)

    # If the x-scale in the affine has a different sign in either image,
    # the bold data needs to be flipped on the x-axis to match.
    # We anticipate this, using an LAS FSL decoder with RAS data.
    if mask_img.affine[0, 0] * bold_img.affine[0, 0] < 0.0:
        bold_4d_data = np.flip(bold_img.get_fdata(), axis=0)
        print(f"{red_on}"
              "bold and mask have opposite x encoding order, "
              "flipping bold x-axis to match."
              f"{color_off}")
    else:
        bold_4d_data = bold_img.get_fdata()

    # Handle 4D data as [all_voxels x time] 2D matrix.
    # To match matlab and the weights, this MUST be done in fortran order
    dims = bold_img.shape
    voxels_per_volume = dims[0] * dims[1] * dims[2]
    bold_full_2d_data = np.reshape(
        bold_4d_data, (voxels_per_volume, dims[3]), order='F'
    )
    mask_full_2d_data = np.reshape(
        mask_img.get_fdata(), voxels_per_volume, order='F'
    )
    masked_bold_data = bold_full_2d_data[mask_full_2d_data != 0]

    if args.verbose:
        print(f"Masked BOLD data is shaped {masked_bold_data.shape} "
              f"and has {np.sum(masked_bold_data != 0.0):,} values.")

    return masked_bold_data


def remove_motion(data, args, scale='zscore', method='manual'):
    """ Regress out motion confounds, return scaled residuals. """

    if args.confounds.name.endswith(".tsv"):
        # fMRIPrep prepares a tab-separated table, with a header row
        confounds = pd.read_csv(args.confounds, sep='\t')
    elif args.confounds.name.endswith(".par"):
        # If motion correction was done by FSL Feat, double-spaces
        confounds = pd.read_csv(args.confounds, sep='  ', header=None)
    confounds['bias'] = 1.0

    # One way is to do this with nilearn, in one line:
    if method == 'nilearn':
        # Nilearn insists we should de-trend or standardize.
        # For now, I prevent it to ensure these results are identical to matlab.
        return clean(data.T, confounds=confounds.values, detrend=False,
                     standardize=scale, standardize_confounds=False).T

    # Another way is to replicate matlab exactly and do all of this manually:
    beta_motion = np.dot(data, np.linalg.pinv(confounds.values).T)
    motion_residuals = data - np.dot(beta_motion, confounds.values.T)

    if scale == 'zscore':
        # Compute z scores across voxel rows, NOT time columns
        # with population degrees of freedom, not sample

        # Python returns a row of NaN z-scores for a row of zero data.
        # Matlab returns a row of zeros, which is more useful.
        # Here, we zero out the NaNs to allow scoring via the other voxels.
        raw_z = zscore(motion_residuals, axis=1, ddof=0)
        safe_z = np.nan_to_num(raw_z, nan=0.0)
        return safe_z
    else:
        return motion_residuals


def predict_y(data, args, null_weights=False):
    """ Use measured BOLD data (cleaned) to predict y """

    # Normally, we use a decoder, which is a vector of weights.
    # But we may also want to use all ones for the decoder as a null comparison.
    if null_weights:
        words = "created", "as ones"
        weights = np.ones((data.shape[0] + 1, 1))
    else:
        if args.decoder_weights.name.endswith('nii.gz'):
            # The mask contains weights within the volume, use them.
            words = "extracted", "from decoder volume"
            weight_img = nib.load(args.decoder_weights)
            dims = weight_img.shape
            voxels_per_volume = dims[0] * dims[1] * dims[2]
            weight_2d_data = np.reshape(
                weight_img.get_fdata(), voxels_per_volume, order='F'
            )
            # Add a bias, for the intercept. This is never zero or one in Noam's
            # decoders, though. :( I am using a 0-intercept, and a brief
            # investigation looked like putting it AFTER the data fits best.
            # weights = np.append(weight_2d_data[weight_2d_data != 0.0], 0.0)
            # weights = np.insert(weight_2d_data[weight_2d_data != 0.0], 0, 0.0)
            weights = weight_2d_data[weight_2d_data != 0.0]
            weights = weights.reshape((len(weights), 1))
        else:
            words = "loaded", "from weights vector"
            weights = pd.read_csv(
                args.decoder_weights, sep='\t', header=None
            ).values

    if args.verbose:
        print(f"  {words[0]} {len(weights)} weights {words[1]}")

    if data.shape[0] == weights.shape[0]:
        # No intercept, use as-is
        x = data
    else:
        # The weights have an intercept, add ones to the data
        x = np.append(data, np.ones((1, data.shape[1])), axis=0)
    y_hat = np.dot(weights.T, x).T

    # This is the decoder score for each t, whatever it is the decoder is fit to
    return y_hat


def main(args):
    """ Entry point """

    print("Decoding {} with a '{}' decoder.".format(
        str(args.bold_file), args.name
    ))

    bold_data = load_and_mask_bold(args)
    final_data = remove_motion(bold_data, args, scale='zscore')
    for null_weights in [True, False, ]:
        label = "ones" if null_weights else "weights"
        predicted_y = predict_y(final_data, args, null_weights=null_weights)
        if np.sum(np.isnan(predicted_y)) > 0:
            print("NaN values in predicted y, no scores!")
        pd.DataFrame(predicted_y).to_csv(
            args.output_path / f"all_trs_{args.name}_{label}_scores.tsv",
            sep='\t', header=False, index=False,
        )

    return 0


if __name__ == "__main__":
    main(validated(get_arguments()))
