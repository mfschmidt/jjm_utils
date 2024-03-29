#!/usr/bin/env python3

# filter_confounds.py

import os
import pathlib
import argparse
import pandas as pd


""" The first several functions return lists of columns for use
    in filtering the dataframe.
"""


def motion_confounds(data, dof=6, verbose=False):
    """ Return the specified number of motion regressor columns from data. """

    motion_prefixes = ["trans", "rot", ]
    motion_axes = ["x", "y", "z", ]
    motion_suffixes = ["derivative1", "power2", "derivative1_power2", ]

    column_candidates = []
    if verbose:
        print(f" finding motion columns for dof of '{dof}' ({type(dof)})")
    if dof in [6, 12, 18, 24, ]:
        for pre in motion_prefixes:
            for ax in motion_axes:
                column_candidates.append(f"{pre}_{ax}")
                if dof in [12, 18, 24, ]:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[0]}"
                    )
                if dof in [18, 24, ]:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[1]}"
                    )
                if dof in [24, ]:
                    column_candidates.append(
                        f"{pre}_{ax}_{motion_suffixes[2]}"
                    )
    else:
        print("Only motion degrees-of-freedom in multiples of 6 are handled.")
        print("The first 6 are 3-dimensions each of translation and rotation.")
        print("The next 6 are derivatives of each of the first six.")
        print("The next 6 are powers of each of the first six.")
        print("The next 6 are derivatives of powers of each of the first six.")

    included_columns = [col for col in column_candidates
                        if col in data.columns]
    if verbose:
        print(f" found {len(column_candidates)} motion columns,"
              f" {len(included_columns)} are ok to include.")

    return data[included_columns]


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
        col for col in data.columns
        if (col.startswith("motion_outlier") and data[col].sum() > 0)
    ]
    return data[
        [col for col in column_candidates if col in data.columns]
    ]


def mod_confounds(args):
    """ Read confounds and write out a subset, based on specified level. """

    full_confounds_data = pd.read_csv(args.input, sep="\t", header=0)
    full_shape = full_confounds_data.shape

    if args.start_tr is not None and args.start_tr > 0:
        full_confounds_data = full_confounds_data.iloc[args.start_tr:, :]
        print(f" {len(full_confounds_data.index)} x {len(full_confounds_data.columns)}  "
              "via '--start-tr'")

    if args.total_trs is not None and args.total_trs > 0:
        full_confounds_data = full_confounds_data.iloc[:args.total_trs, :]
        print(f" {len(full_confounds_data.index)} x {len(full_confounds_data.columns)}  "
              "via '--total-trs'")

    if args.level == "motion":
        confounds_data = motion_confounds(
            full_confounds_data, args.motion, verbose=args.verbose
        )
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")
    elif args.level == "basic":
        confounds_data = pd.concat([
            basic_confounds(full_confounds_data),
            motion_confounds(full_confounds_data, args.motion,
                             verbose=args.verbose),
        ], axis=1, sort=False)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")
    elif args.level == "curious":
        confounds_data = pd.concat([
            basic_confounds(full_confounds_data),
            motion_confounds(full_confounds_data, args.motion,
                             verbose=args.verbose),
            curious_confounds(full_confounds_data),
        ], axis=1, sort=False)
        if args.verbose:
            print(f" {len(confounds_data.columns)} cols via '{args.level}'")
    elif args.level == "csf_wm":
        if "csf_wm" in full_confounds_data.columns:
            confounds_data = pd.concat([
                full_confounds_data[["csf_wm", ]],
                motion_confounds(full_confounds_data, args.motion,
                                 verbose=args.verbose),
            ], axis=1, sort=False)
        else:
            confounds_data = motion_confounds(
                full_confounds_data, args.motion, verbose=args.verbose
            )
    else:
        confounds_data = pd.DataFrame()

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
        confounds_data.fillna(0.0).to_csv(
            args.output, sep="\t", float_format='%.6f', index=False
        )


""" And finally, the interface components
"""


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="\n".join([
            "Extract regressor columns from confounds file.",
            "",
            "--level csf_wm   : csf_wm",
            "--level basic   : fd + 6 tCompCors, 6 aCompCors, and 6 motion",
            "--level curious : <basic> + global + csf + whitematter",
            "--level motion  : defaults to 6 motion parameters",
            "                : --motion {6,12,18,24} increases that",
            "--motion N      : adds {6,12,18,24} motion vectors to --level",
            "--scrub         : adds however many frames have excess motion",
        ])
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
             "('csfwm', 'motion', 'basic', 'curious')",
    )
    parser.add_argument(
        "-m", "--motion", type=int, default=6,
        help="How many motion degrees of freedom to regress (6, 12, 18, 24)",
    )
    parser.add_argument(
        "-s", "--scrub", action="store_true",
        help="Set to include motion outlier TRs as one-hot regressors",
    )
    parser.add_argument(
        "--start-tr", type=int, default=0,
        help="Crop TRs before this, a '--start-tr 5' removes first five TRs.",
    )
    parser.add_argument(
        "--total-trs", type=int, default=None,
        help="Set to number of TRs, by default all TRs to the end are used.",
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
