#!/usr/bin/env python3

# ims_bids_events_to_feat.py

import re
import argparse
import textwrap
import pandas as pd
from pathlib import Path


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            From a BIDS-valid events.tsv file, export txt files for feat.
            
            The simplest example:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv .
            
            The above command will generate separate Feat-compatible files
            for each trial_type in the events.tsv file specified. It will
            write out multiple Feat-appropriate txt files to ./
            
            An example to treat different stimuli as separate event types:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --split-on-stimulus "*a.jpg"
             
            The above command will treat different stimuli, each with
            'stimulus' == '*a.jpg' as different trial types, saving them
            to separate Feat files.
            
            An example to use the response in one trial as the value for
            another:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --use-response-from "How badly do you feel?" \\
                --use-response-to instruct
             
            The above command will use the response to the question nearest the
            instruct as the value in the instruct record.
            
            An example to group contiguous events as a single block:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --as-block arrow
             
            The above command will treat multiple repeated arrow trials as one
            large block, adding together each arrow duration into the long
            block.
            
            A ppi example:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --ppi-trial-types neut_new --ppi-trial-types emot_old \\
                --ppi-stimuli-from "5*.jpg"
             
            The above will create separate ppi files for memory and instruct
            events, and for each stimulus saved with 'instruct' events. The
            events will have +1 and -1 values representing each trial_type
            or stimulus.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--trial-types", nargs='*', default=[],
        help="By default, a different file will be generated for each "
             "trial_type in the events.tsv file. Optionally, by setting "
             "'--trial-types a b c', only events matching trial_type "
             "of 'a', 'b', or 'c' will be extracted."
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
        "--use-response-to", action='append', default=[],
        help="See --use-response-from - these must be used together."
    )
    parser.add_argument(
        "--use-response-from", action='append', default=[],
        help="If a stimulus is provided in one event, but you want to "
             "describe that stimulus with a response to a later event, "
             "like a question asking about the initial stimulus, you can "
             "specify the trial_type OR stimulus for reading the response "
             "value with --use-response-from and the location for writing "
             "it with --use-response-to. Both arguments must be supplied "
             "to use this feature. Because only one event can "
             "be set at a time, this feature only writes out the file for "
             "the event specified. For each event matching "
             "--use-response-to, the response value from the next event "
             "in temporal order matching --use-response-from will be used. "
             "And the number of --use-response-to and --use-response-from "
             "events must be the same to align values properly."
    )
    parser.add_argument(
        "--long-name", action="store_true",
        help="By default, text files are written with short names, "
             "but setting this to true causes text files to be written "
             "with full bids key-value pairs for sub, ses, task, run."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    # Assess/tweak arguments
    parsed_args = parser.parse_args()

    # Use Path objects rather than strings
    setattr(parsed_args, "events_file", Path(parsed_args.events_file))
    setattr(parsed_args, "output_path", Path(parsed_args.output_path))

    # The --use-response argument is essentially a shortcut to implement both
    # --use-response-from and --use-response-to as the same trial_type.
    assert (
        len(parsed_args.use_response_from) == len(parsed_args.use_response_to)
    )
    for trial_type in parsed_args.use_response:
        parsed_args.use_response_from.append(trial_type)
        parsed_args.use_response_to.append(trial_type)
    parsed_args.use_response = []

    return parsed_args


def metadata_from_path(path):
    """ Return all key/value pairs from BIDS path

        :param str path:
            The full path to an events.tsv file
        :return dict:
            A map of key-value pairs found in the BIDS-style path
    """

    bids_map = {}
    pairs = re.findall(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", str(path))
    for pair in pairs:
        if pair[0] in bids_map:
            # This is a duplicate, like sub or ses being in path and file
            if pair[1] != bids_map[pair[0]]:
                print(f"ERROR: No way to tell if {pair[0]} is "
                      f"'{pair[1]}' or '{bids_map[pair[0]]}'.")
        else:
            bids_map[pair[0]] = pair[1]

    return bids_map


def make_feats(data, column_name, value):
    """ Generate feat regressors """

    output_cols = ['onset', 'duration', 'ones', ]
    if value == 'ALL':
        return data[output_cols]
    else:
        return data[data[column_name] == value][output_cols]


def get_events_data(filepath, shift=0.0):
    """ Load and format the BIDS-compatible events file. """

    data = pd.read_csv(
        filepath, sep='\t', dtype=str, keep_default_na=False
    )

    data['ones'] = "1"

    data['onset'] = data['onset'].astype(float)
    data['onset'] = data['onset'] - shift
    num_mangled_trs = (data['onset'] < 1.0).sum()
    if num_mangled_trs > 0:
        print(f"  {num_mangled_trs} TRs have onsets before {shift} "
              "and would shift to less than zero. "
              "They have been set to 0.0 instead.")
    data.loc[data['onset'] < 0.0, 'onset'] = 0.0

    data['duration'] = data['duration'].astype(float)

    return data


def main(args):
    """ Entry point """

    if args.verbose:
        print("Extracting from {}, shifting by {:0.2f} seconds.".format(
            args.events_file, args.shift,
        ))

    # Load data
    metadata = metadata_from_path(args.events_file)
    data = get_events_data(args.events_file, args.shift)

    # Do the work
    args.output_path.mkdir(parents=True, exist_ok=True)
    # if metadata['task'] == 'test':
    #     build_test_regressors(data)
    # elif metadata['task'] == 'study':
    #     build_study_regressors(data)

    # We hard-code explicit assumptions about columns that must be in the file.
    # Every run needs all files, even if they're empty.
    no_filter = pd.Series([True, ] * len(data.index))
    for trial in ['all', 'emot', 'neut', ]:
        if trial == 'all':
            trial_filter = no_filter
        else:
            trial_filter = data['trial_type'] == trial
        for response in ['all', '3', '4', 'n/a', ]:
            if response == 'all':
                resp_filter = no_filter
            else:
                resp_filter = data['response'] == response
            for sim in ['all', 'old', 'hsim', 'lsim', 'new', ]:
                if sim == 'all':
                    sim_filter = no_filter
                else:
                    sim_filter = data['sim_type'] == sim
                for correct in ['all', '0', '1', 'n/a', ]:
                    if correct == 'all':
                        corr_filter = no_filter
                    else:
                        corr_filter = data['correct'] == correct
                    # Filter out regressors for this specific context
                    regressors = data[
                        trial_filter & sim_filter & corr_filter & resp_filter
                    ][['onset', 'duration', 'ones']]
                    # And if they're empty, create a null regressor
                    if len(regressors.index) == 0:
                        regressors = pd.DataFrame([{
                            "onset": 0.0, "duration": 0.0, "ones": 0,
                        }])
                    # Save it to disk
                    if metadata['task'] == 'study' and sim == 'new':
                        # No such thing as 'new' similarity in study task
                        # We have chosen not to write these empty files.
                        pass
                    else:
                        filename = "_".join([
                            f"trial-{trial}",
                            f"response-{response.replace('/', '')}",
                            f"sim-{sim}",
                            f"correct-{correct.replace('/', '')}",
                        ]) + ".txt"
                        regressors.to_csv(
                            args.output_path / filename,
                            sep='\t', index=False, header=False,
                            float_format='%0.3f'
                        )


if __name__ == "__main__":
    main(get_arguments())
