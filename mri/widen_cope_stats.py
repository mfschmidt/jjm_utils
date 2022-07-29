#!/usr/bin/env python3

# widen_cope_stats.py

import sys
import argparse
from pathlib import Path
import pandas as pd
from numpy import float64


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Find all copes, extracted by mask, "
                    "and save them out as one row of data.",
    )
    parser.add_argument(
        "summary_file",
        help="The input csv file containing mask-averaged cope data",
    )
    parser.add_argument(
        "-o", "--output-file",
        help="The wide output csv file to write",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Validate input
    if Path(args.summary_file).exists():
        if args.verbose:
            print(f"File '{args.summary_file}' exists.")
        setattr(args, "summary_file", Path(args.summary_file))
    else:
        print(f"File '{args.summary_file}' does not exist.")
        sys.exit(1)

    # Validate output
    if args.output_file is None:
        if args.verbose:
            print("output-file not set, calculating a default")
        output_filename = args.summary_file.name.replace(".csv", "_wide.csv")
        setattr(args, "output_file",
                args.summary_file.parent / output_filename)

    if args.verbose:
        print(f"Writing wide file to '{args.output_file}'.")

    return args


def main(args):
    """ Entry point """

    # Create empty Series objects to fill with data
    region_means = pd.Series(
        data=[], dtype=float64, name=args.summary_file.name[:10]
    )
    voxel_counts = pd.Series(
        data=[], dtype=float64, name=args.summary_file.name[:10]
    )

    # Load the csv file specified from the command-line
    df = pd.read_csv(args.summary_file, sep=",")
    if args.verbose:
        print(f"A sample of the {df.shape}-shaped input:")
        print(df)
    
    # Iterate over rows of input data, extracting specific columns
    for idx, row in df.sort_values(['cope', 'mask', ]).iterrows():
        region_means[f"{row['cope']}_{row['mask'][9:]}"] = row['mean']
        voxel_counts[f"{row['cope']}_{row['mask'][9:]}_voxnum"] = row['n']

    # Combine means and counts into one long Series
    combined = pd.concat([region_means, voxel_counts, ], axis=0)

    if args.verbose:
        print(f"Values are in a Series shaped {region_means.shape}.")
        print(f"Counts are in a Series shaped {voxel_counts.shape}.")
        print(f"Combined, they're a {type(combined)} shaped {combined.shape}.")
    
    # Transpose the long Series into a DataFrame with one wide row.
    wide_df = pd.DataFrame(combined).transpose()
    if args.verbose:
        print(f"The output is {wide_df.shape}-shaped.")
    
    # Write the wide DataFrame to disk
    wide_df.to_csv(args.output_file, sep=",")


if __name__ == "__main__":
    main(get_arguments())
