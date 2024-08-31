#!/usr/bin/env python3

# concat_bold_runs.py

"""
Given a list of images and a path to save their concatenated output,
this script will concatenate them along the time axis and save the final image.
"""

import sys
from pathlib import Path
import argparse
import datetime
import numpy as np
import nibabel as nib
from nilearn import image

from nibabel.cifti2 import cifti2_axes


# Some globals

# These codes allow for turning color on and off
GREEN_ON = "\033[1;32m"
YELLOW_ON = "\033[1;33m"
RED_ON = "\033[0;31m"
COLOR_OFF = "\033[0m"


def get_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Concatenate BOLD runs along the time dimension.",
    )
    parser.add_argument(
        "input_files", type=str, nargs='+',
        help="The nifti or cifti files to concatenate",
    )
    parser.add_argument(
        "output_file",
        help="The path to save the final concatenated image",
    )
    parser.add_argument(
        "--crop-initial-volumes",
        default=0,
        type=int,
        help="From each 4D BOLD file, crop off the the first N volumes",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="De-mean and normalize each image to SD=1 before concatenation",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Run no tasks, just report on what would be run without this flag",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    args = get_env(args)
    args = validate_args(args)

    return args


def get_env(args):
    """Integrate environment variables into our args."""

    return args


def validate_args(args):
    """Ensure the environment will support the requested workflows."""

    errors = list()

    # Ensure we aren't overwriting anything important.
    setattr(args, "output_file", Path(args.output_file))
    if args.output_file.exists():
        errors.append(f"The proposed output file, '{str(args.output_file)}' "
                      "already exists. To re-build it, move or delete it.")
    elif not args.output_file.parent.exists():
        args.output_file.parent.mkdir(exist_ok=True, parents=True)
        if args.verbose:
            print(f"Creating output path '{args.output_path}'.")

    # Ensure the input files exist.
    if len(args.input_files) < 1:
        errors.append(f"There aren't any input files to concatenate.")
    elif len(args.input_files) < 2:
        print(f"{YELLOW_ON}WARNING: Only one file found; "
              f"Concatenating a single file seems pointless.{COLOR_OFF}")
    if len(args.input_files) > 0:
        setattr(args, "missing_files",
                [Path(f) for f in args.input_files if not Path(f).exists()])
        if len(args.missing_files) > 0:
            print(f"{YELLOW_ON}WARNING: Some files could not be found:"
                  f"{COLOR_OFF}")
            for f in args.missing_files:
                print(f"  {RED_ON}X{COLOR_OFF} - {str(f)}{COLOR_OFF}")
        setattr(args, "input_files",
                [Path(f) for f in args.input_files if Path(f).exists()])

    # Report the good news if verbose was turned on.
    if args.verbose:
        print(GREEN_ON)
        if len(args.input_files) > 2:
            print(f"Concatenating {len(args.input_files)} files.")
        if not args.output_file.exists():
            print(f"Output path '{str(args.output_file)}' is clear for writing.")
        if args.crop_initial_volumes > 0:
            print("Each input file will have its first "
                  f"{args.crop_initial_volumes} volumes removed before "
                  "inclusion.")
        print(COLOR_OFF)

    # Report the bad news with or without verbosity; these are fatal.
    if len(errors) > 0:
        for e in errors:
            print(f"{RED_ON}ERROR{COLOR_OFF}", e)
        sys.exit(1)

    return args


def get_cifti_axis(cifti_img, axis):
    """ From within a cifti image, get its anatomical or time axis.

        :param cifti_img: the cifti image object
        :param axis: which axis to retrieve

        :returns: The cifti2 SeriesAxis or BrainModelAxis
    """

    for i in range(cifti_img.ndim):
        ax = cifti_img.header.get_axis(i)
        if axis == 'brain' and isinstance(ax, cifti2_axes.BrainModelAxis):
            return ax
        if axis == 'series' and isinstance(ax, cifti2_axes.SeriesAxis):
            return ax
        if axis == 'label' and isinstance(ax, cifti2_axes.LabelAxis):
            return ax
        if axis == 'parcels' and isinstance(ax, cifti2_axes.ParcelsAxis):
            return ax
        if axis == 'scalar' and isinstance(ax, cifti2_axes.ScalarAxis):
            return ax
    return None


def tr_str(axis):
    """ From a cifti2 SeriesAxis object, return the TR and its str rep

        :param SeriesAxis axis: A cifti2 SeriesAxis object
        :returns: a (float, str) tuple with TR length and string representation
    """

    if axis.unit == 'SECOND':
        return axis.step, f"{axis.step:0.1f}s"
    else:
        print("I can only handle 'SECOND's, not '{axis.unit}'s")


def concatenate_cifti_images(cifti_images, crop_initial_volumes,
                             normalize=False, verbose=False):
    """ Concatenate all nifti files in the list after cropping each.

        :param cifti_images: list of cifti images to concatenate
        :param int crop_initial_volumes: volumes to crop from the beginning of each image
        :param bool normalize: set to True to normalize each image before adding
        :param bool verbose: set to True to trigger verbose output

        :returns: A large cifti image containing all concatenated data
    """

    final_data = None
    final_brain_axis = None
    tr_lengths = dict()

    for img in cifti_images:
        # Keep track of tr-lengths from each file
        tr_len, tr_key = tr_str(get_cifti_axis(img, 'series'))
        if tr_key not in tr_lengths.keys():
            tr_lengths[tr_key] = 1
        else:
            tr_lengths[tr_key] += 1

        # Extract data and crop along the time axis if necessary
        if crop_initial_volumes > 0:
            img_data = img.get_fdata()[crop_initial_volumes:, :]
        else:
            img_data = img.get_fdata()
        if verbose:
            print(f"{img_data.shape[0]}-TR ({tr_key}) image has "
                  f"mean {np.mean(img_data):0.4f}, "
                  f"sd {np.std(img_data):0.4f}")

        if normalize:
            img_data = img_data - np.mean(img_data)
            img_data = img_data / np.std(img_data)
            if verbose:
                print(f"  {img_data.shape[0]}-TR ({tr_key}) image normed to "
                      f"mean {np.mean(img_data):0.4f}, "
                      f"sd {np.std(img_data):0.4f}")

        # Add this image's data to the stack
        if final_data is None:
            # We'll copy the anatomical axis header from the first file
            final_brain_axis = get_cifti_axis(img, 'brain')
            final_data = img_data.copy()
        else:
            final_data = np.vstack((final_data, img_data))

    # Build the time/series axis for the new concatenated data
    final_tr_length = 0.0
    cur_hi_count = 0
    if len(tr_lengths.keys()) != 1:
        print(f"{RED_ON}TR lengths are inconsistent across images!{COLOR_OFF}")
        for tr_key, count in tr_lengths.items():
            print(f"  {count} files have {tr_key} TRs")
            if count > cur_hi_count:
                final_tr_length = float(tr_key[: -1])
                cur_hi_count = count
        print(f"Using the most frequent TR, {final_tr_length:0.1f}")
    final_series_axis = cifti2_axes.SeriesAxis(
        start=0.0, step=final_tr_length, size=final_data.shape[0]
    )
    final_header = nib.cifti2.Cifti2Header.from_axes((
        final_series_axis, final_brain_axis
    ))

    return nib.cifti2.Cifti2Image(final_data, final_header)


def guess_file_type(p):
    """ From the path, p, return the file type. """

    if p.name.endswith(".nii.gz"):
        img = nib.load(p)
        if isinstance(img, nib.nifti1.Nifti1Image):
            if (len(img.shape) < 3) or (len(img.shape) > 4):
                print(RED_ON, end="")
                print(f"Nifti1Image OK, but not 4D: shaped {str(img.shape)}")
                print(COLOR_OFF, end="")
                return img, "Nifti1Image", img.shape, None
            elif len(img.shape) == 3:
                return img, "Nifti1Image", img.shape, 1
            elif len(img.shape) == 4:
                return img, "Nifti1Image", img.shape[:3], img.shape[3]
            return img, "Confused", (), 0
        else:
            return img, f"Type {str(type(img))} for '.nii.gz'?", (), None
    elif p.name.endswith(".dtseries.nii"):
        img = nib.load(p)
        if isinstance(img, nib.cifti2.cifti2.Cifti2Image):
            return img, "Cifti2Image", img.shape[1], img.shape[0]
        else:
            return img, f"Type {str(type(img))} for '.dtseries.nii'?", (), None
    elif p.name.endswith(".dscalar.nii"):
        img = nib.load(p)
        if isinstance(img, nib.cifti2.cifti2.Cifti2Image):
            return img, "Cifti2Image", img.shape[1], img.shape[0]
        else:
            return img, f"Type {str(type(img))} for '.dscalar.nii'?", (), None
    else:
        return None, "unknown", (), None


def main(args):
    """Entry point"""

    start_dt = datetime.datetime.now()

    print("The order of concatenation:")
    volume_shapes = set()
    file_types = set()
    concatable_images = list()
    final_num_volumes = 0
    for i, file_path in enumerate(args.input_files):
        img, file_type, volume_shape, num_volumes = guess_file_type(file_path)
        if num_volumes is None:
            COLOR_ON = RED_ON
            marker = "✗"
        else:
            COLOR_ON = GREEN_ON
            marker = "✓"
            volume_shapes.add(volume_shape)
            file_types.add(file_type)
            concatable_images.append(img)
            final_num_volumes += num_volumes
        print(f" - {i + 1:> 3d}. {COLOR_ON}{marker} - {file_path}  -  "
              f"{num_volumes} {volume_shape}-shaped {file_type} volumes "
              F"{COLOR_OFF}")
    if len(volume_shapes) > 1 or len(file_types) > 1:
        print(f"{RED_ON}No way to concatenate volumes of different shapes!")
        print("Shapes:", volume_shapes)
        print("File types:", file_types)
        print(COLOR_OFF)
        return -1

    final_volume_shape = volume_shapes.pop()
    final_file_type = file_types.pop()
    print(f"{len(concatable_images)} good {final_file_type} images, "
          f"all shaped {final_volume_shape}, with {final_num_volumes} "
          f"total volumes; ready to concatenate")

    # Do the work on the files found
    final_image = None
    if args.dry_run:
        print(f"{GREEN_ON}NOT running because --dry-run is set.{COLOR_OFF}")
    elif isinstance(final_volume_shape, int):
        final_image = concatenate_cifti_images(
            concatable_images, args.crop_initial_volumes,
            args.normalize, args.verbose
        )
    elif isinstance(final_volume_shape, tuple) and len(final_volume_shape) == 3:
        final_image = image.concat_imgs(
            [i.slicer[:,:,:,args.crop_initial_volumes:]
             for i in concatable_images]
        )
    else:
        print(f"I can't figure out what to do with {final_volume_shape}-shaped"
              f" {final_file_type} volumes.")

    if final_image is not None:
        final_image.to_filename(args.output_file)
        print(f"New image shaped {final_image.shape} written to "
              f"{args.output_file}")
    else:
        print(RED_ON)
        print(f"There's no final image to save. Something's gone wrong.")
        print(COLOR_OFF)

    end_dt = datetime.datetime.now()
    if args.verbose:
        print(
            "Done. Ran from {} to {} ({})".format(
                start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt - start_dt,
            )
        )


if __name__ == "__main__":
    main(get_arguments())
