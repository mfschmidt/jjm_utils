#!/usr/bin/env python3

# bids_events_to_feat.py

import sys
import re
import argparse
import pandas as pd
from pathlib import Path


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="From a BIDS-valid events.tsv file, "
                    "export txt files for feat.",
    )
    parser.add_argument(
        "events_file",
        help="the BIDS-valid events file",
    )
    parser.add_argument(
        "output_path",
        help="the path for writing out multiple feat-friendly event files",
    )
    parser.add_argument(
        "-s", "--shift", type=float, default=0.0,
        help="subtract this amount from each onset time",
    )
    parser.add_argument(
        "--as-block", action='append', default=[],
        help="use --as-block for each trial_type that should be aggregated "
             "into a single block, rather than single events. This implies "
             "the use of a dummy numeral 1 for the block."
    )
    parser.add_argument(
        "--split-on-stimulus", action='append', default=[],
        help="By default, a different file will be generated for each "
             "trial_type in the events.tsv file. Optionally, by setting "
             "'--split-on-stimulus trial_type', events matching trial_type "
             "will also be split across stimuli."
    )
    parser.add_argument(
        "--use-response", action='append', default=[],
        help="By default '1' will be written in the third column."
             "specify '--use-response question' to use the response "
             "to the 'question' trial_type in the third column of "
             "the output timing file. Be careful and review your data "
             "because any 'nan' values will be treated as 1. This may "
             "not be appropriate for your models."
    )
    parser.add_argument(
        "--long-name", action="store_true",
        help="By default, text files are written with short names, "
             "but setting this to true causes text files to be written "
             "with full bids key-value pairs for sub, ses, task, run."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    return parser.parse_args()


def main(args):
    """ Entry point """

    if args.verbose:
        print("Extracting from {}, shifting by {:0.2f} seconds.".format(
            args.events_file, args.shift,
        ))

    metadata = metadata_from_path(args.events_file)
    data = pd.read_csv(args.events_file, sep='\t')

    # Check some assumptions
    available_trial_types = list(data['trial_type'].unique())

    def print_trial_types():
        print("       Available trial_types:")
        for t in available_trial_types:
            print(f"       - {t}")

    for block_name in args.as_block:
        if block_name not in available_trial_types:
            print(f"ERROR: Trial type '{block_name}' requested to be grouped "
                  "into a single block, but no events match this trial_type. ")
            print_trial_types()
            sys.exit(1)
    for stimulus in args.split_on_stimulus:
        if stimulus not in available_trial_types:
            print(f"ERROR: Trial type '{stimulus}' requested to be split "
                  "into separate trial types, but no events match this trial_type. ")
            print_trial_types()
            sys.exit(2)
    for stimulus in args.use_response:
        if stimulus not in available_trial_types:
            print(f"ERROR: Trial type '{stimulus}' requested to be quantified "
                  "with response values, but no events match this trial_type. ")
            print_trial_types()
            sys.exit(3)

    # Separate trial_type events, but in a way that separates blocks if needed
    timing_tables = {}
    last_trial_type = ''
    for idx, row in data.iterrows():
        # Extract just the data we need for our smaller feat-friendly file.
        if row['trial_type'] in args.use_response:
            try:
                third_value = int(row['response'])
            except ValueError:
                third_value = 1
                print(f"  WARNING: converted '{row['response']}' as response "
                      f"to '{row['trial_type']}':'{row['stimulus']}' "
                      f"in '{str(args.events_file)}' to a 1 value for Feat.")
        else:
            third_value = 1
        if args.shift > 0.0:
            onset_value = float(row['onset']) - args.shift
        else:
            onset_value = float(row['onset'])
        event = {
            "onset": onset_value,
            "duration": float(row['duration']),
            "value": third_value,
        }

        # Figure out how to split up trial_types and stimuli
        if row['trial_type'] in args.split_on_stimulus:
            if row['trial_type'] in args.as_block:
                print("WARNING: treating '{row['trial_type']}' as a block "
                      "precludes splitting on stimulus. Entire blocks of "
                      "{row['trial_type']} events will be grouped without "
                      "regard to stimulus.")
                # ignore stimulus, just worry about trial_type
                group_name = row['trial_type']
            else:
                group_name = "_".join([
                    row['trial_type'],
                    "-".join([
                        "stimulus",
                        "".join([c for c in row['stimulus'].lower() if c.isalpha()]),
                    ]),
                ])
        else:
            group_name = row['trial_type']

        # Depending on context, store this event or append it to one.
        if group_name in timing_tables:
            if row['trial_type'] in args.as_block:
                if last_trial_type == group_name:
                    same_event = timing_tables[group_name][-1]
                    same_event['duration'] = (
                        event['onset'] + event['duration'] - same_event['onset']
                    )
                else:
                    timing_tables[group_name].append(event)
            else:
                timing_tables[group_name].append(event)
        else:
            timing_tables[group_name] = [event, ]

        # Remember last_trial_type to detect continuing blocks of same trials
        # same as group_name, only matters with multi-event blocks
        last_trial_type = row['trial_type']

    # Save the separate event types to separate files
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    for group_name in sorted(timing_tables.keys()):
        relevant_data = pd.DataFrame(timing_tables[group_name])[
            ['onset', 'duration', 'value']
        ].sort_values('onset')

        orig_trial_type = group_name.split("_")[0]
        if orig_trial_type in args.as_block:
            descriptor = "blocks"
        else:
            descriptor = "events"
        if orig_trial_type in args.use_response:
            weight = "as-responses"
        else:
            weight = "as-ones"

        if args.long_name:
            filename = "_".join([
                f"sub-{metadata['sub']}",
                f"task-{metadata['task']}",
                f"run-{metadata['run']}",
                f"trial-{group_name}",
                weight,
                descriptor,
            ]) + ".txt"
        else:
            filename = "_".join([
                f"trial-{group_name}",
                weight,
                descriptor,
            ]) + ".txt"

        relevant_data.to_csv(
            Path(args.output_path) / filename,
            sep='\t', index=None, header=None, float_format="%.3f",
        )


def metadata_from_path(path):
    """ Return all key/value pairs from BIDS path

        :param str path:
            The full path to an events.tsv file
        :return dict:
            A map of key-value pairs found in the BIDS-style path
    """

    bids_map = {}
    pairs = re.findall(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", path)
    for pair in pairs:
        if pair[0] in bids_map:
            # This is a duplicate, like sub or ses being in path and file
            if pair[1] != bids_map[pair[0]]:
                print(f"ERROR: No way to tell if {pair[0]} is "
                      f"'{pair[1]}' or '{bids_map[pair[0]]}'.")
        else:
            bids_map[pair[0]] = pair[1]

    return bids_map


if __name__ == "__main__":
    main(get_arguments())
