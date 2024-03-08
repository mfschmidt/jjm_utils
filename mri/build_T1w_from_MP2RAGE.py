#!/usr/bin/env python3

""" build_T1w_from_MP2RAGE.py

    The algorithms in this file were taken from Jose P Marques'
    code at https://github.com/JosePMarques/MP2RAGE-related-scripts and
    papers at https://doi.org/10.1016/j.neuroimage.2009.10.002 and
    https://doi.org/10.1371/journal.pone.0099676. We have a collection
    of scans from the Pittsburgh portion of the Fampath study that have
    only MP2RAGE anat scans, with no pure T1w images to feed FreeSurfer,
    fMRIPrep, or C-PAC, and we require something that can be skull-
    stripped.

    - Mike Schmidt Aug 7, 2022
"""

import sys
import json
import argparse
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn.plotting import plot_anat
from collections import namedtuple


NiftiPair = namedtuple('NiftiPair', ['image', 'sidecar', 'n', ])
Mp2rage = namedtuple('Mp2rage', ['uni', 'inv1', 'inv2', ])


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="From a collection of three files collected as an MP2RAGE "
                    "sequence (uni, inv1, inv2), create a fourth nifti/json "
                    "pair that resembles a T1w MPRAGE image.",
    )
    parser.add_argument(
        "anatpath",
        help="the path to a BIDS-valid sub-*/[ses-*/]anat/ directory",
    )
    parser.add_argument(
        "-r", "--regularization", type=int, default=12,
        help="Select a regularization parameter that determines how "
             "aggressive the denoising should be. It may be worth "
             "experimenting with this as each scanner may produce "
             "different noise patterns. This is a positive integer, "
             "usually from 1 to 10, but I've used 12 to 15 to OK "
             "effect in our Siemens 3T data.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Verify we are working with a real path
    if Path(args.anatpath).exists():
        # Use a Path rather than a str
        setattr(args, "anatpath", Path(args.anatpath))
    else:
        print(f"Path '{args.anatpath}' does not exist.")
        sys.exit(1)

    return args


def retype_image(image, datatype=np.float64):
    """ return same image at double resolution """

    new_img_data = np.copy(image.get_fdata())
    new_img_data = new_img_data.astype(datatype)
    image.set_data_dtype(datatype)
    if image.header['sizeof_hdr'] == 348:
        return nib.Nifti1Image(
            new_img_data, image.affine, header=image.header
        )
    elif image.header['sizeof_hdr'] == 540:
        return nib.Nifti2Image(
            new_img_data, image.affine, header=image.header
        )
    else:
        print("ERROR: image neader is neither Nifti1 nor Nifti2!")
        return None


def root_squares_pos(a, b, c):
    """ calculate and return the positive portion of the quadratic equation

        Ported from matlab:
        rootsquares_pos   = @(a, b, c)
            (-b + sqrt(b.^2 - 4*a.*c)) ./ (2*a);
    """
    return np.divide(
        ((-1.0 * b) + np.sqrt(np.square(b) - (4.0 * np.multiply(a, c)))),
        (2.0 * a)
    )


def root_squares_neg(a, b, c):
    """ calculate and return the negative portion of the quadratic equation

        Ported from matlab:
        rootsquares_neg   = @(a, b, c)
            (-b - sqrt(b.^2 - 4*a.*c)) ./ (2*a);
    """
    return np.divide(
        ((-1.0 * b) - np.sqrt(np.square(b) - (4.0 * np.multiply(a, c)))),
        (2.0 * a)
    )


def robustify(inv1, inv2, beta):
    """ no idea, have not thought it through, just copied from matlab

        Ported from matlab:
        MP2RAGErobustfunc = @(INV1, INV2, beta)
            (conj(INV1).*INV2-beta) ./ (INV1.^2 + INV2.^2 + 2*beta);

        :param inv1: the inv-1 MP2RAGE image, already cleaned up a bit
        :param inv2: the unaltered inv-2 MP2RAGE image
        :param beta: The noise level to clean up, higher is more aggressive
    """

    return (
        np.divide(
            np.multiply(np.conj(inv1), inv2) - beta,
            np.square(inv1) + np.square(inv2) + (2.0 * beta)
        )
    )


def threshold_mask(fake_t1w_data, inv2, threshold=100):
    """ Where values in inv2 < 250, cap fake_t1 at that value """

    new_data = fake_t1w_data.copy()
    new_data[inv2 < threshold] = -0.50
    # The following is tempting and seemingly elegant,
    # but inv2 values range from 0 to 1500;
    # t1w data here range from -0.5 to 0.5,
    # so will always be lower than inv2, avoiding any change whatsoever.
    # new_data[inv2 < threshold] = np.min(fake_t1w_data, inv2)[inv2 < threshold]
    return new_data


def report_image_stats(img, name):
    """ print some information about an image """
    if isinstance(img, nib.Nifti1Image) or isinstance(img, nib.Nifti1Image):
        img = img.get_fdata()
    print(f"{name} (shape {img.shape}) ranges "
          f"from {np.nanmin(img):0.2f} to {np.nanmax(img):0.2f}, "
          f"mean {np.nanmean(img):0.2f}, "
          f"{np.count_nonzero(np.isnan(img)):,} NaNs, "
          f"est noise level {img[:, -12:, -12:, ].mean():0.2f}")


def plot_steps(steps):
    """ Plot a list of steps {"title": title, "img": image} """

    fig, axes = plt.subplots(ncols=len(steps), figsize=(4 * len(steps), 4))

    for i, step in enumerate(steps):
        if "range" in step:
            vmin = step['range'][0]
            vmax = step['range'][1]
        else:
            vmin = np.min(step['img'].get_fdata())
            vmax = np.max(step['img'].get_fdata())
        plot_anat(step['img'], cut_coords=[20, ], display_mode='x',
                  vmin=vmin, vmax=vmax,
                  figure=fig, axes=axes[i], )
        axes[i].set_title(step['title'])

    return fig, axes


def robust_combination(mp2rage, regularization=1, verbose=True):
    """ Combine three mp2rage images to create a T1w-like image. """

    # Load all three MP2RAGE images
    img_uni = retype_image(nib.load(mp2rage.uni.image), np.float64)
    data_uni = img_uni.get_fdata()
    img_inv1 = retype_image(nib.load(mp2rage.inv1.image), np.float64)
    img_inv2 = retype_image(nib.load(mp2rage.inv2.image), np.float64)
    data_inv2 = img_inv2.get_fdata()

    if verbose:
        report_image_stats(img_uni, "Original UNI")
        report_image_stats(img_inv1, "Original INV1")
        report_image_stats(img_inv2, "Original INV2")

    # If values are amenable to it, normalize UNI between -0.5 and +0.5
    if np.min(data_uni) >= 0 and np.max(data_uni) >= 0.51:
        # Center values to zero, then divide by the former max,
        # resulting in -0.50 to 0.50 normalized array
        normalized_uni = (
            (data_uni - (np.max(data_uni) / 2.0)) / np.max(data_uni)
        )
        uni_was_normalized = True
    else:
        normalized_uni = data_uni
        uni_was_normalized = False
    if verbose:
        report_image_stats(normalized_uni, "Normalized UNI")

    # Correct the polarity to inv1, based on sign of UNI
    # This flips inv1's voxels to negative if uni's voxel is > the mean,
    # resulting in a very T1-like image in the brain, but with a gray
    # background as 0 is now between the -600ish and +900ish range.
    correct_inv1 = np.sign(normalized_uni) * img_inv1.get_fdata()

    # The inv1 and inv2 arrays are sum-of-squares data.
    # The uni is a phase-sensitive coil combination.
    # Improve the inv1 image by taking the root of each square and selecting
    # the lease likely to be noise at each voxel.
    pos_root_inv1 = root_squares_pos(
        a=(-1.0 * normalized_uni),
        b=data_inv2,
        c=np.multiply((-1.0 * np.square(data_inv2)), normalized_uni)
    )
    neg_root_inv1 = root_squares_neg(
        a=(-1.0 * normalized_uni),
        b=data_inv2,
        c=np.multiply((-1.0 * np.square(data_inv2)), normalized_uni)
    )

    # Wherever the root is very different from the original, exclude it,
    # keeping similar roots.
    pos_root_delta = np.abs(correct_inv1 - pos_root_inv1)
    neg_root_delta = np.abs(correct_inv1 - neg_root_inv1)
    pos_gt_filter = pos_root_delta > neg_root_delta
    neg_gt_filter = neg_root_delta > pos_root_delta

    # Start with the flipped INV1 image, T1w-like
    final_inv1 = correct_inv1

    # Replace values with roots that don't look like noise.
    final_inv1[pos_gt_filter] = neg_root_inv1[pos_gt_filter]
    final_inv1[neg_gt_filter] = pos_root_inv1[neg_gt_filter]

    # Determine noise in inv2 by averaging the empty space as far from
    # the head as possible, above and in front of the forehead, spread
    # all the way across left to right
    noise_level = regularization * data_inv2[:, -11:, -11:, ].mean()

    phase_sensitive_robust = robustify(
        final_inv1, data_inv2, noise_level**2
    )

    masked_phase_sensitive_robust = threshold_mask(
        phase_sensitive_robust, data_inv2,
    )

    if uni_was_normalized:
        # Normalize image back to (0 to 4095) from (-0.5 to +0.5)
        phase_sensitive_t1w = 4095.0 * (masked_phase_sensitive_robust + 0.5)
        phase_sensitive_img = retype_image(
            nib.Nifti1Image(
                phase_sensitive_t1w, img_uni.affine, header=img_uni.header
            ),
            datatype=np.float32,
        )
    else:
        phase_sensitive_img = retype_image(
            nib.Nifti1Image(
                masked_phase_sensitive_robust,
                img_uni.affine, header=img_uni.header
            ),
            datatype=np.float32,
        )

    if verbose:
        report_image_stats(phase_sensitive_img,
                           "Final phase-sensitive robust fake T1w")

    return phase_sensitive_img


def pair_from_nii(nifti_path):
    """ From a nifti_path, create a NiftiPair namedtuple. """

    if "gz" in str(nifti_path):
        json_path = Path(str(nifti_path).replace(".nii.gz", ".json"))
    else:
        json_path = Path(str(nifti_path).replace(".nii", ".json"))

    if nifti_path.exists() and json_path.exists():
        return NiftiPair(
            image=nifti_path, sidecar=json_path, n=2,
        )
    elif nifti_path.exists():
        return NiftiPair(
            image=nifti_path, sidecar=None, n=1
        )
    else:
        return NiftiPair(
            image=None, sidecar=None, n=0
        )


def find_mp2rage_files(anat_path):
    """ Given the path containing the files, return a labeled named tuple.
    """

    uni = NiftiPair(image=None, sidecar=None, n=0)
    inv1 = NiftiPair(image=None, sidecar=None, n=0)
    inv2 = NiftiPair(image=None, sidecar=None, n=0)
    for file in anat_path.glob("*.nii.gz"):
        if re.search(r"sub-(.+)_ses-(.+)_.*UNIT1.nii.gz", str(file)):
            uni = pair_from_nii(file)
        if re.search(r"sub-(.+)_ses-(.+)_.*inv-1_MP2RAGE.nii.gz", str(file)):
            inv1 = pair_from_nii(file)
        if re.search(r"sub-(.+)_ses-(.+)_.*inv-2_MP2RAGE.nii.gz", str(file)):
            inv2 = pair_from_nii(file)

    if uni.n > 0 and inv1.n > 0 and inv2.n > 0:
        print(f"Found uni image   : {uni.image}")
        print(f"Found uni sidecar : {uni.sidecar}")
        print(f"Found inv1 image  : {inv1.image}")
        print(f"Found inv1 sidecar: {inv1.sidecar}")
        print(f"Found inv2 image  : {inv2.image}")
        print(f"Found inv2 sidecar: {inv2.sidecar}")
        return Mp2rage(uni=uni, inv1=inv1, inv2=inv2)

    print(f"Could not find necessary files at '{str(anat_path)}'")
    print(f"  - UNI : {str(uni.image)}")
    print(f"  - INV1: {str(inv1.image)}")
    print(f"  - INV2: {str(inv2.image)}")
    sys.exit(2)


def adapt_sidecar(mp2rages, from_str, to_str, args):
    """ Copy a sidecar, add some notes, and save it with a new name. """

    # Generate a name for the new sidecar
    t1w_sidecar_name = mp2rages.uni.sidecar.name.replace(from_str, to_str)
    t1w_sidecar = mp2rages.uni.sidecar.parent / t1w_sidecar_name

    # Use the original UNI sidecar as a template, and write a new one.
    if mp2rages.uni.sidecar.exists():
        with open(mp2rages.uni.sidecar) as json_in:
            data = json.load(json_in)
        data['BasedOn'] = [
            mp2rages.uni.image.name,
            mp2rages.inv1.image.name,
            mp2rages.inv2.image.name,
        ]
        data['SeriesDescription'] = "_".join([
            data['ProtocolName'], "MP2RAGE_denoised_background"
        ])
        data['NoiseRegularization'] = args.regularization
        with open(t1w_sidecar, "w") as json_out:
            json.dump(data, json_out, indent=4)
        if args.verbose:
            print(f"Wrote sidecar '{t1w_sidecar.name}'")
    else:
        print(f"UNI sidecar '{mp2rages.uni.sidecar.name}' does not exist, "
              "so there is no template for a new T1w sidecar.")


def main(args):
    """ Entry point """

    # We know we'll be getting a bunch of NaNs and we handle them,
    # so we don't need a bunch of error messages polluting our output.
    np.seterr(divide='ignore', invalid='ignore')

    if args.verbose:
        print(f"Searching {str(args.anatpath)} for MP2RAGE files, "
              f"using r={args.regularization} regularization.")

    # Find input files
    mp2rages = find_mp2rage_files(args.anatpath)

    # build the fake T1w
    mprage_mimic = robust_combination(
        mp2rages, regularization=args.regularization, verbose=args.verbose
    )

    # Write out T1w files
    t1w_image_name = mp2rages.uni.image.name.replace("_UNIT1", "_T1w")
    nib.save(mprage_mimic, str(args.anatpath / t1w_image_name))
    if args.verbose:
        print(f"Wrote fake T1w image to '{t1w_image_name}'")
    adapt_sidecar(mp2rages, "_UNIT1", "_T1w", args)


if __name__ == "__main__":
    main(get_arguments())
