#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:55:17 2022

@author: mike

clear_wraparound.py

"""

import sys
import argparse
from pathlib import Path
from nilearn import image
from nibabel import nifti1
import numpy as np
# import matplotlib.pyplot as plt


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Clear wraparound artifact from BOLD images.",
    )
    parser.add_argument(
        "input",
        help="the path to a file or a func directory",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="The directory or filename to save",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Determine how to save output
    if args.output.endswith(".nii.gz") or args.output.endswith(".nii"):
        # Treat input and output as one file.
        if Path(args.input).is_file():
            # The file is specified, use it.
            setattr(args, "output_file", args.output)
            Path(args.output_dir)
        else:
            print("Output is a file, but input is not. If you want to output "
                  "one file, you need to input one file.")
            sys.exit(1)
    else:
        # Treat output as a directory
        setattr(args, "output_file", "")
        Path(args.input).mkdir(parents=True, exist_ok=True)
        setattr(args, "output_dir", args.output)
        
    return args


def flatten_4d_to_2d(data_4d):
    """ From a 4d BOLD run, return a 2d flat image of all slices, averaged. """
    data_3d = np.mean(data_4d, axis=3)
    data_2d = np.mean(data_3d, axis=2)
    return data_2d

    
def gradient_to_background(data_4d):
    """ Smoothly reduce signal to zero in front of the head. """
    
    # Determine the normal background from empty corners
    bg_val = np.mean(np.concatenate([
        data_4d[2:5, 5:50, :, :].ravel(),
        data_4d[data_4d.shape[0] - 5:data_4d.shape[0] - 2, 5:50, :, :].ravel(),
    ]))
    
    # Determine low-signal point anterior to the head.
    data_ap = np.mean(np.mean(np.mean(data_4d, axis=3), axis=2), axis=0)
    mid_point = int(len(data_ap) / 2)
    min_val_y = np.argmin(data_ap[mid_point:]) + mid_point
    
    # Every voxel in that A-P plane must decrease to the mean anteriorly
    for t in range(data_4d.shape[3]):
        for x in range(data_4d.shape[1]):
            for z in range(data_4d.shape[2]):
                # As x goes from min_val_z to the end, f(x) goes to bg_val
                xp = [min_val_y, data_4d.shape[0]]
                fp = [data_4d[x, min_val_y, z, t], bg_val]
                data_4d[x, min_val_y:, z, t] = np.interp(
                    list(range(min_val_y, data_4d.shape[0])),
                    xp, fp
                )
                
    return data_4d


def main(args):
    """ App entry """
    
    func_image_files = []
    if Path(args.input).is_file():
        if args.verbose:
            print(f"Found a file at '{str(Path(args.input).resolve())}'.")
        func_image_files = [Path(args.input), ]
    elif Path(args.input).is_dir():
        if args.verbose:
            print(f"Found a directory at '{str(Path(args.input).resolve())}'.")
        func_image_files = Path(args.input).glob(
            "sub-*_task-*_run-*_bold.nii.gz"
        )
    else:
        print(f"Path '{args.somepath}' does not exist.")

    for i, file in enumerate(func_image_files):
        
        # Load original data
        bold_image = nifti1.load(file)
        bold_data = image.get_data(bold_image)
        axial_mean = flatten_4d_to_2d(bold_data)
        print(f"{i+1:>02d}. "
              f"{file.name} is shaped {bold_data.shape}, "
              f"with {axial_mean.shape} axial slices.")
        
        # Alter original data
        cleared_data = gradient_to_background(bold_data)
        
        # Save cleared data
        cleared_nifti = nifti1.Nifti1Image(
            cleared_data, bold_image.affine, bold_image.header
        )
        if len(args.output_file) > 0:
            # A file was specified, write to it.
            print(f"  writing to {args.output_file}")
            nifti1.save(cleared_nifti, args.output_file)
        else:
            # No file was specified, use input filename
            print(f"  writing to {args.output_dir}/{file.name}")
            nifti1.save(cleared_nifti, str(Path(args.output_dir) / file.name))


if __name__ == "__main__":
    main(get_arguments())
