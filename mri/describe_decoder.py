#!/usr/bin/env python3

# describe_decoder.py

from pathlib import Path
import argparse
import sys
import re
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from scipy.io import loadmat
from nilearn import plotting
from tinyhtml import html, h


# Some globals

# These codes allow for turning color on and off
GREEN_ON = "\033[1;32m"
RED_ON = "\033[0;31m"
COLOR_OFF = "\033[0m"

sep_map = {'.tsv': '\t', '.csv': ',', '.txt': ' ', }


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Build an html file and supporting images about a decoder.",
    )
    parser.add_argument(
        "input_file",
        help="mask or weights file from a decoder",
    )
    parser.add_argument(
        "--output_path", default="",
        help="The output path for writing, defaults to input_file location."
             "Overriding the default will break the relative paths to images "
             "inside the html. I don't know a better way, because the absolute "
             "path to a network drive may be different for each of us.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run all tasks, even if it means overwriting existing data",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    args = get_env(args)
    if validate_args(args):
        return args
    else:
        sys.exit(1)


def get_env(args):
    """ Integrate environment variables into our args. """

    return args


def validate_args(args):
    """ Ensure the environment will support the requested workflows. """

    args_are_valid = True

    # Valid decoders can take the following forms:
    # - One weights file, with non-zero weights representing a mask.
    # - A nifti mask and a weights tsv, paired
    # - A nifti mask and a weights matlab .mat file, paired
    setattr(args, "input_file", Path(args.input_file))
    if args.input_file.exists():
        if "_mask." in str(args.input_file):
            setattr(args, "mask_file", Path(args.input_file))
            if args.verbose:
                print(f"Found a mask file, '{str(args.mask_file)}'")
            for ext in [".tsv", ".mat", ]:
                weights_filename = args.mask_file.name\
                    .replace("_mask.", "_weights.").replace(".nii.gz", ext, )
                weights_file = Path(args.input_file).parent / weights_filename
                if weights_file.exists():
                    setattr(args, "weights_file", weights_file)
                    if args.verbose:
                        print(f"Found a weights file, '{str(args.weights_file)}'")
            if "weights_file" not in args:
                print("Warning: No weights found, just a mask.")
                setattr(args, "weights_file", None)
        else:  # typically if "weight" in str(args.input_file):
            setattr(args, "weights_file", Path(args.input_file))
            if args.verbose:
                print(f"Found a weights file, '{str(args.weights_file)}'")
            if ".tsv" in args.weights_file.name:
                mask_filename = args.weights_file.name.\
                    replace("_weights.", "_mask.").replace(".tsv", ".nii.gz")
            else:
                mask_filename = args.weights_file.name.\
                    replace("_weights.", "_mask.").replace(".mat", ".nii.gz")
            mask_file = Path(args.input_file).parent / mask_filename
            if mask_file.exists():
                setattr(args, "mask_file", mask_file)
                if args.verbose:
                    print(f"Found a mask file, '{str(args.mask_file)}'")
            else:
                print("Warning: No separate mask found, just weights.")
                setattr(args, "mask_file", None)
    else:
        print(f"{RED_ON}ERROR:{COLOR_OFF} "
              f"The input file '{args.input_file}' does not exist.")
        args_are_valid = False

    # Determine the decoder's name
    if (args.weights_file is not None) and args.weights_file.exists():
        match = re.search(r"(\w+)_weights\.", str(args.weights_file))
        setattr(args, "decoder_name", match.group(1))
    elif (args.mask_file is not None) and args.mask_file.exists():
        match = re.search(r"(\w+)_mask\.", str(args.mask_file))
        setattr(args, "decoder_name", match.group(1))
    else:
        setattr(args, "decoder_name", "Unknown")

    if args.output_path == "":
        args.output_path = args.input_file.parent
    if not Path(args.output_path).exists():
        if args.force:
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
        else:
            print(f"Path '{args.output_path}' does not exist.")
            args_are_valid = False

    setattr(args, "html_file",
            args.output_path / f"{args.decoder_name}.html")
    setattr(args, "mask_figure",
            args.output_path / f"{args.decoder_name}_mask.png")
    setattr(args, "weights_figure",
            args.output_path / f"{args.decoder_name}_weights.png")
    setattr(args, "hist_figure",
            args.output_path / f"{args.decoder_name}_histogram.png")

    return args_are_valid


def plot_histogram(weights, plot_file):
    """ Plot histograms of weights and save to plot_file. """

    fig, axes = plt.subplots(ncols=2, figsize=(9, 3), layout='tight')

    num_weights = len(weights)
    sns.histplot(weights, ax=axes[0])
    axes[0].set_title(f"All {num_weights:,} weights")

    num_weights = len(np.flatnonzero(weights))
    sns.histplot(weights[np.nonzero(weights)], ax=axes[1])
    axes[1].set_title(f"Just {num_weights:,} non-zero weights")

    fig.savefig(plot_file)


def main(args):
    """ Entry point """

    # Seaborn uses some deprecated pandas function calls and pandas warns
    # about them. Nothing we can do, and it works fine, so we choose to keep
    # the output clean by ignoring them.
    warnings.filterwarnings("ignore")

    # Default values should get overwritten, but we set them in case they don't
    mask_str = "No mask"
    mask_img = None
    weights_str = "No weights"
    weights = None
    weights_img = None

    # Handle the data we have, don't bother with what we don't
    if (args.mask_file is None) and (args.weights_file is None):
        print("Nothing to describe")
        return 1

    if (args.mask_file is not None) and args.mask_file.exists():
        mask_img = nib.load(args.mask_file)
        mask_str = (f"Mask is shaped {mask_img.shape} with "
                    f"{len(np.flatnonzero(mask_img.get_fdata())):,} voxels.")
        mask_plot = plotting.plot_roi(mask_img)
        mask_plot.savefig(args.mask_figure)
        if args.weights_file is None:
            args.weights_file = args.mask_file

    if (args.weights_file is not None) and args.weights_file.exists():
        weights = None
        # In reality, these are all one column, so don't have any
        # delimiters, and these will all do the same thing.
        if args.weights_file.suffix in [".tsv", ".csv", ".txt", ]:
            weights_df = pd.read_csv(
                args.weights_file,
                sep=sep_map[args.weights_file.suffix],
                header=None, index_col=None
            )
            weights = weights_df.values.ravel()
        elif args.weights_file.suffix in [".mat", ]:
            weights_dict = loadmat(args.weights_file)
            weights = weights_dict.get('w').ravel()

        if weights is not None:
            # To project a list of weights onto 3D space, we must have a 3D
            # mask. If we don't have a mask, this is just a list of numbers
            # without any context at all for where they go.
            n_vox = mask_img.shape[0] * mask_img.shape[1] * mask_img.shape[2]
            mask_vector = np.reshape(mask_img.get_fdata(), n_vox, order="F")
            weights_in_3d = np.zeros(mask_vector.shape)
            weights_in_3d[mask_vector != 0] = weights[: -1]
            weights_in_3d = np.reshape(weights_in_3d, mask_img.shape, order="f")
            weights_img = nib.Nifti1Image(weights_in_3d, affine=mask_img.affine)

        if args.weights_file.name.endswith(".nii.gz"):
            weights_img = nib.load(args.weights_file)
            # This file may also be the mask, so handle the mask first
            if mask_img is None:
                mask_img = nib.Nifti1Image(
                    (weights_img.get_fdata() != 0).astype(np.int16),
                    affine=weights_img.affine, dtype=np.int16
                )
                mask_str = f"Mask (non-zero weights) is shaped {mask_img.shape}"
                mask_plot = plotting.plot_roi(mask_img)
                mask_plot.savefig(args.mask_figure)

            # And now deal with the weights
            weights = weights_img.get_fdata().ravel()

        # Whether from the tsv or the nii.gz, plot the weights
        weights_plot = plotting.plot_stat_map(weights_img)
        weights_plot.savefig(args.weights_figure)

        weights_str = f"{len(weights):,} voxel weights"
        plot_histogram(weights, args.hist_figure)

    # Build html for description report
    html_content = html(lang="en")(
        h("head")(
            h("title")(args.decoder_name),
        ),
        h("body")(
            h("h1")(args.decoder_name),
            h("p")(mask_str),
            h("p")(
                h("img", src=str(args.mask_figure.name)),
            ),
            h("p")(weights_str),
            h("p")(
                h("img", src=str(args.weights_figure.name)),
            ),
            h("p")(
                h("img", src=str(args.hist_figure.name)),
            ),
        ),
    )
    if Path(args.output_path).exists():
        print(f"Writing description to '{str(args.html_file)}'")
        with open(args.html_file, "w") as f:
            f.write(html_content.render())


if __name__ == "__main__":
    main(get_arguments())
