#!/usr/bin/env python3

# aggregate_timeseries.py

import sys
import re
from pathlib import Path
import argparse
import pandas as pd


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="For a given subject's timeseries directory, read all "
                    "mean timeseries files and organize them by ROI."
    )
    parser.add_argument(
        "ts_path",
        help="The path to timeseries files",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Ensure the timeseries directory exists
    args.ts_path = Path(args.ts_path).resolve()
    if args.ts_path.exists():
        if args.verbose:
            print(f"Path is '{args.ts_path}'")
    else:
        print(f"Path '{args.ts_path}' does not exist.")
        sys.exit(1)

    if args.verbose:
        print(f"Timeseries are at '{args.ts_path}'.")

    return args


def main(args):
    """ Entry point """

    if args.verbose:
        print("Collecting timeseries from: {}".format(
            str(args.ts_path.resolve())
        ))

    # Find all the timeseries
    bold_values = []
    subject_id, task, run, roi, hemi = "NA", "NA", "NA", "NA", "NA"
    for ts_file in args.ts_path.glob("sub-*_mean_ts.tsv"):
        pattern = re.compile(
            r"sub-([A-Z][0-9]+)_task-([a-z]+)_run-([0-9]+)_roi-([A-Za-z]+)_hemi-([a-z]+)_mean_ts.tsv"
        )
        match = pattern.match(ts_file.name)
        if match:
            subject_id = match.group(1)
            task = match.group(2)
            run = match.group(3)
            roi = match.group(4)
            hemi = match.group(5)
        else:
            print("Could not determine region from regex of filename.")
        ts_data = pd.read_csv(ts_file, index_col=False, header=None)
        for i, bold_value in enumerate(ts_data.values):
            bold_values.append({
                "subject": subject_id,
                "task": task,
                "run": run,
                "roi": roi,
                "hemi": hemi,
                "orig_tr": i + 1,
                "bold": bold_value[0],
            })

    # For all specified copes and all specified masks, combine them and average
    if len(bold_values) == 0:
        print(f"No timeseries found at {args.ts_path}")
        return 0

    all_bold_df = pd.DataFrame(bold_values)
    all_bold_df[
        ['subject', 'task', 'run', 'roi', 'hemi', 'orig_tr', 'bold', ]
    ].sort_values(['subject', 'task', 'run', 'roi', 'hemi', 'orig_tr', ]).to_csv(
        args.ts_path / f"sub-{subject_id}_all_ts.tsv",
        index=None, sep='\t'
    )


if __name__ == "__main__":
    main(get_arguments())
