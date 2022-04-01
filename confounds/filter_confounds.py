#!/usr/bin/env python3

# filter_confounds.py

import os
import pathlib
import argparse
import pandas as pd


""" The first several functions return lists of columns for use
    in filtering the dataframe.
"""


def motion_confounds(data, dof=6):
    """ Return the specified number of motion regressor columns from data. """

    motion_prefixes = ["trans", "rot", ]
    motion_axes = ["x", "y", "z", ]
    motion_suffixes = ["derivative1", "power2", "derivative1_power2", ]

    column_candidates = []
    if dof in [6, 12, 18, 24, ]:
        for pre in motion_prefixes:
            for ax in motion_axes:
                column_candidates.append(f"{pre}_{ax}")
                if dof == 12:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[0]}"
                    )
                if dof == 18:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[1]}"
                    )
                if dof == 24:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[2]}"
                    )
    else:
        print("Only motion degrees-of-freedom in multiples of 6 are handled.")
        print("The first 6 are 3-dimensions each of translation and rotation.")
        print("The next 6 are derivatives of each of the first six.")
        print("The next 6 are powers of each of the first six.")
        print("The next 6 are derivatives of powers of each of the first six.")

    return data[
        [col for col in column_candidates if col in data.columns]
    ]


def basic_confounds(data):
    """ Return only the basic regressor columns from data. """
    column_candidates = ["framewise_displacement", ] +\
        [col for col in data.columns if col.startswith("t_comp_cor_")] +\
        [col for col in data.columns if col.startswith("a_comp_cor_")][:6]
    return data[
        [col for col in column_candidates if col in data.columns]
    ]


def curious_confounds(data):
    """ Return the regressors we are interested in testing. """
    column_candidates = ["global_signal", "csf", "white_matter", ]
    return data[
        [col for col in column_candidates if col in data.columns]
    ]


def scrubbed_confounds(data):
    """ Return the regressors we are interested in testing. """
    column_candidates = [
        col for col in data.columns if col.startswith("motion_outlier")
    ]
    return data[
        [col for col in column_candidates if col in data.columns]
    ]


def mod_confounds(args):
    """ Read confounds and write out a subset, based on specified level. """

    full_confounds_data = pd.read_csv(args.input, sep="\t", header=0)
    full_shape = full_confounds_data.shape

    if args.level == "motion":
        confounds_data = motion_confounds(full_confounds_data, args.motion)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")
    elif args.level == "basic":
        confounds_data = pd.concat([
            basic_confounds(full_confounds_data),
            motion_confounds(full_confounds_data, args.motion),
        ], axis=1, sort=False)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")
    elif args.level == "curious":
        confounds_data = pd.concat([
            basic_confounds(full_confounds_data),
            motion_confounds(full_confounds_data, args.motion),
            curious_confounds(full_confounds_data),
        ], axis=1, sort=False)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")

    if args.scrub:
        confounds_data = pd.concat([
            confounds_data,
            scrubbed_confounds(full_confounds_data),
        ], axis=1, sort=False)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via 'scrubbing'")

    print("Read [{} TRs x {} regressors] confounds, writing [{} x {}]".format(
        full_shape[0], full_shape[1],
        confounds_data.shape[0], confounds_data.shape[1]
    ))
    if args.force or not os.path.isfile(args.output):
        os.makedirs(pathlib.Path(args.output).parents[0], exist_ok=True)
        confounds_data.to_csv(args.output, sep="\t", index=False)


""" And finally, the interface components
"""


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Extract regressor columns from confounds file.",
    )
    parser.add_argument(
        "-i", "--input",
        help="Input, the complete confound file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output, the filtered regressor columns",
    )
    parser.add_argument(
        "-l", "--level", default="basic",
        help="A level label specifying which regressor columns to include"
             "('motion', 'basic', 'curious')",
    )
    parser.add_argument(
        "-m", "--motion", type=int, default=6,
        help="How many motion degrees of freedom to regress (6, 9, 12, 15)",
    )
    parser.add_argument(
        "-s", "--scrub", action="store_true",
        help="Set to include motion outlier TRs as one-hot regressors",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Set to overwrite the output file, if it exists",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Set to trigger verbose output",
    )

    return parser.parse_args()


def main(args):
    """ Entry point """

    if args.verbose:
        scrubbing = " and scrubbing" if args.scrub else ""
        print(f"Reading {args.input} to write {args.output}.")
        print(f"Extracting {args.level} "
              f"with {args.motion} motion dof{scrubbing}.")

    mod_confounds(args)


if __name__ == "__main__":
    main(get_arguments())
