#!/usr/bin/env python3

# nifti_compare.py

import sys
from pathlib import Path
import argparse
from statistics import correlation

import nibabel as nib
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
    if len(img_a_voxels) != len(img_b_voxels):
        print("Images are not the same size.")
        print(f"Image a is {img_a.shape}.")
        print(f"Image b is {img_b.shape}.")
        sys.exit(1)
    if len(mask_voxels) != len(img_a_voxels):
        print("The mask is not the same size as the images.")
        print(f"Images are {img_a.shape}.")
        print(f"The mask is {mask_img.shape}.")
        sys.exit(1)

    # Set up masked vectors
    img_a_masked_voxels = img_a_voxels[mask_voxels != 0.0]
    img_b_masked_voxels = img_b_voxels[mask_voxels != 0.0]

    # Compare the images
    for label, vector_a, vector_b in [
        ("Whole image", img_a_voxels, img_b_voxels),
        ("Within mask", img_a_masked_voxels, img_b_masked_voxels),
    ]:
        r = correlation(vector_a, vector_b)
        print(
            f"{label}: {len(vector_a):>7,} voxels - "
            f"         Pearson r == {r:0.4f}"
        )

        vector_a_1d = vector_a.reshape(1, -1)
        vector_b_1d = vector_b.reshape(1, -1)
        sim = cosine_similarity(vector_a_1d, vector_b_1d).ravel()[0]
        print(
            f"{label}: {len(vector_a):>7,} voxels - "
            f" Cosine similarity == {sim:0.4f}"
        )


if __name__ == "__main__":
    main(get_arguments())
