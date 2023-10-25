#!/usr/bin/env python3

# nifti_compare.py

import sys
from pathlib import Path
import argparse
from statistics import correlation, StatisticsError
import numpy as np

import nibabel as nib
from nibabel.processing import resample_from_to
from nilearn import image
from sklearn.metrics.pairwise import cosine_similarity


# Some globals

# These codes allow for turning color on and off
GREEN_ON = "\033[1;32m"
RED_ON = "\033[0;31m"
COLOR_OFF = "\033[0m"


def get_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Compare two nifti images.",
    )
    parser.add_argument(
        "image_a",
        help="a nifti file for comparison",
    )
    parser.add_argument(
        "image_b",
        help="a nifti file for comparison",
    )
    parser.add_argument(
        "--mask",
        default="",
        help="apply a specific mask to each image, compare voxels within it",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="set to trigger verbose output",
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    """Ensure the environment will support the requested workflows."""

    args_are_valid = True

    if Path(args.image_a).exists():
        setattr(args, "image_a", Path(args.image_a))
    else:
        print(f"Image '{args.image_a}' does not exist.")
        args_are_valid = False

    if Path(args.image_b).exists():
        setattr(args, "image_b", Path(args.image_b))
    else:
        print(f"Image '{args.image_b}' does not exist.")
        args_are_valid = False

    if args.mask == "":
        setattr(args, "mask", None)
    elif Path(args.mask).exists():
        setattr(args, "mask", Path(args.mask))
    else:
        print(f"A mask '{args.mask}' was specified, but does not exist.")
        args_are_valid = False

    if args_are_valid:
        return args
    else:
        sys.exit(1)


def get_voxels(image_path):
    """Load a nifti image and extract its voxels."""

    if image_path is None:
        return None, None

    img = nib.Nifti1Image.from_filename(image_path)
    if len(img.shape) > 3:
        print(f"  collapsing {len(img.shape)}-D image down to 3-D")
        img = image.mean_img(img)
    return img, img.get_fdata().ravel()


def main(args):
    """Entry point"""

    if args.verbose:
        print("Comparing '{}' vs '{}'.".format(
            str(args.image_a), str(args.image_b)
        ))

    # Load images and mask; ensure they are comparable
    img_a, img_a_voxels = get_voxels(args.image_a)
    img_b, img_b_voxels = get_voxels(args.image_b)
    mask_img, mask_voxels = get_voxels(args.mask)

    if img_a_voxels is None:
        print(f"could not load {str(args.image_a)}")
    if img_b_voxels is None:
        print(f"could not load {str(args.image_b)}")
    if mask_voxels is None and args.mask is not None:
        print(f"could not load {str(args.mask)}")

    # If affines don't match, resample the image and compare both versions
    resampled_img_b_voxels = None
    if np.array_equal(img_a.affine, img_b.affine):
        print(f"Affines match")
    else:
        print(f"{RED_ON}Affines differ; "
              f"images are not in the same space." f"{COLOR_OFF}")
        resampled_img_b = resample_from_to(img_b, img_a)
        resampled_img_b_voxels = resampled_img_b.get_fdata().ravel()

    # Detect problems and bail out if we can't do a comparison.
    if len(img_a_voxels) != len(img_b_voxels):
        print("Images are not the same size.")
        print(f"Image a is {img_a.shape}.")
        print(f"Image b is {img_b.shape}.")
        sys.exit(1)
    if (mask_voxels is not None) and len(mask_voxels) != len(img_a_voxels):
        print("The mask is not the same size as the images.")
        print(f"Images are {img_a.shape}.")
        print(f"The mask is {mask_img.shape}.")
        sys.exit(1)

    # Save the comparisons we need to do in a list
    tests = [
        ("Whole image", img_a_voxels, img_b_voxels),
    ]

    # Set up resampled vector
    if resampled_img_b_voxels is not None:
        tests.append(("Resampled", img_a_voxels, resampled_img_b_voxels))
    # Set up masked vectors
    if mask_voxels is not None:
        img_a_masked_voxels = img_a_voxels[mask_voxels != 0.0]
        img_b_masked_voxels = img_b_voxels[mask_voxels != 0.0]
        tests.append(("Within mask", img_a_masked_voxels, img_b_masked_voxels))
        if resampled_img_b_voxels is not None:
            resampled_img_b_masked_voxels = resampled_img_b_voxels[
                mask_voxels != 0.0
            ]
            tests.append(
                (
                    "Resampled, within mask",
                    img_a_masked_voxels,
                    resampled_img_b_masked_voxels,
                )
            )

    # Compare the images
    for label, vector_a, vector_b in tests:
        # print(f"  vector_a {vector_a.shape}, vector_b {vector_b.shape}")
        comment_str = ""
        if np.array_equal(vector_a, vector_b):
            r = 1.0
            comment_str = " (identical)"
        else:
            try:
                r = correlation(vector_a, vector_b)
                if r > 0.9999:
                    comment_str = " (not identical)"
            except StatisticsError:
                r = 0.0
                if np.std(vector_a) == 0.0:
                    comment_str = " (not computable; a has no variance)"
                elif np.std(vector_b) == 0.0:
                    comment_str = " (not computable; b has no variance)"
                elif (np.std(vector_a) == 0.0) and (np.std(vector_b) == 0.0):
                    comment_str = (" (not computable;"
                                   " neither a nor b has any variance)")
                else:
                    comment_str = " (not computable)"
        print(
            f"{label:<24}: {len(vector_a):>7,} voxels - "
            f"         Pearson r == {r:0.4f}{comment_str}"
        )

        vector_a_1d = vector_a.reshape(1, -1)
        vector_b_1d = vector_b.reshape(1, -1)
        sim = cosine_similarity(vector_a_1d, vector_b_1d).ravel()[0]
        print(
            f"{label:<24}: {len(vector_a):>7,} voxels - "
            f" Cosine similarity == {sim:0.4f}"
        )


if __name__ == "__main__":
    main(get_arguments())
