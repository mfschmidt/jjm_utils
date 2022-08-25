#!/usr/bin/env python3

# bids_events_to_feat.py

import sys
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
                --split-on-stimulus question
             
            The above command will treat different questions, each with
            'trial_type' == 'question' as different trial types, saving them
            to separate Feat files.
            
            An example to use the response in one trial as the value for another:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --use-response-from "How badly do you feel?" --use-response-to instruct
             
            The above command will use the response to the question nearest the
            instruct as the value in the instruct record.
            
            An example to group contiguous events as a single block:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --as-block arrow
             
            The above command will treat multiple repeated arrow trials as one
            large block, adding together each arrow duration into the long block.
            
            A ppi example:
            
                bids_events_to_feat.py sub-1_ses-1_task-a_run-01_events.tsv . \\
                --ppi-trial-types memory --ppi-trial-types instruct \\
                --ppi-stimuli-from instruct
             
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
        "--as-block", nargs='*',
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
        "--ppi-trial-types", nargs='*',
        help="Specify the trial_types to be included as a connected block. "
             "for ppi analyses. For example, in a memory task with a memory "
             "event, followed by an instruct event, you would use "
             "`--ppi-blocks memory instruct` to cause those two events "
             "to be treated as monolithic blocks. Combine this with "
             "`--ppi-stimuli`."
    )
    parser.add_argument(
        "--ppi-stimuli-from",
        help="Specify the stimuli to discriminate `--ppi-trial-types`."
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

    # If nothing is to be blocked, use an empty list rather than None.
    # This allows loops to simply not run rather than raise exceptions.
    if parsed_args.as_block is None:
        parsed_args.as_block = []

    # The --use-response argument is essentially a shortcut to implement both
    # --use-response-from and --use-response-to as the same trial_type.
    assert len(parsed_args.use_response_from) == len(parsed_args.use_response_to)
    for trial_type in parsed_args.use_response:
        parsed_args.use_response_from.append(trial_type)
        parsed_args.use_response_to.append(trial_type)
    parsed_args.use_response = []

    return parsed_args


def trial_type_plus_stimulus(trial_type, stimulus):
    """ Create a unique safe identifier combining trial_type and stimulus.

        :param str trial_type:
            The trial_type label
        :param str stimulus:
            The stimulus label, will have all non-alpha characters stripped
        :returns:
            A concatenation of trial_type and stimulus, separated by an
            underscore, and stimulus keyed by 'stimulus-'
    """

    return "_".join([
        f"trial-{trial_type}",
        "-".join([
            "stimulus",
            "".join([c for c in stimulus.lower() if c.isalpha()]),
        ]),
    ])


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


def handle_errors(args, available_trial_types, available_stimuli):
    """ Assess arguments and report any problems with them. """

    # If any of the options specifies a trial_type that does not exist,
    # set the error_response, then when all options are checked, we can
    # print available trial_types just once, and exit
    error_response = 0
    for block_name in args.as_block:
        if block_name not in available_trial_types:
            print(f"ERROR: Trial type '{block_name}' requested to be grouped "
                  "into a single block, but no events match this trial_type. ")
            error_response = 1
    for stimulus in args.split_on_stimulus:
        if stimulus not in available_trial_types:
            print(f"ERROR: Trial type '{stimulus}' requested to be split "
                  "into separate trial types, but no events match this trial_type. ")
            error_response = 2
            
    assert len(args.use_response_from) == len(args.use_response_to), \
        "Each --use-response-from must be matched to a --use-response-to."
    for trial_type in args.use_response_from + args.use_response_to:
        if (
                (trial_type not in available_trial_types) and
                (trial_type not in available_stimuli)
        ):
            print(f"ERROR: '{trial_type}' specified, "
                  "but no events match this as a trial_type or stimulus. ")
            error_response = 3
    # If multiple levels are specified in either of these 'use'
    # arguments, make sure they get combined appropriately to
    # discriminate between them.
    if len(args.use_response_to) > 1:
        args.split_on_stimulus.append(args.use_response_to[0])
    if len(args.use_response_from) > 1:
        args.split_on_stimulus.append(args.use_response_from[0])

    return error_response


def do_ppi(data, args):
    """ Extract events for PPI analyses. """

    # Separate trial_type events, but in a way that separates blocks if needed
    timing_tables = {}
    trial_types = set()
    stimuli = set()
    ppi_events = []
    cur_event = {"rows": [], }
    for idx, row in data.iterrows():
        # Extract just the data we need for our smaller feat-friendly files.

        # In the mem trial example used to write this,
        # we'll hit 8 rows constituting 4 events, each with a 'memory' and an 'instruct'
        # Each 'instruct' will have 'distance' or 'immerse' as stimulus.
        # These need to be organized into 8 files with 4 events each,
        # coded with onsets from different rows and each with a different
        # paradigm for +/- 1

        # Gather multi-row events
        # Only bother with rows included in --ppi-trial-types
        if row['trial_type'] in args.ppi_trial_types:
            if row['trial_type'] in cur_event['rows']:
                # The current event is probably complete. Save it and create a new one.
                ppi_events.append(cur_event)
                cur_event = {"rows": [], }
            trial_types.add(row['trial_type'])
            cur_event[f"{row['trial_type']}_onset"] = float(row['onset']) - args.shift
            cur_event[f"{row['trial_type']}_duration"] = float(row['duration'])
            cur_event[f"{row['trial_type']}_stimulus"] = row['stimulus']
            if row['trial_type'] == args.ppi_stimuli_from:
                cur_event["stimulus"] = row['stimulus']
            cur_event['rows'].append(row['trial_type'])
            if row['trial_type'] in args.ppi_stimuli_from:
                stimuli.add(row['stimulus'])

    # Save the final event from the loop, not yet appended
    ppi_events.append(cur_event)

    if args.verbose:
        print(f"Found {len(ppi_events)} ppi events.")

    # Create the eight (assuming 2 values for each of 3 variables, 2**3) files
    # we need an all-memory and all-instruct,
    # each with opposite distance/immerse encoding.
    for trial in sorted(trial_types):
        # Hackily, we need to know what the 'other' trial is for file naming
        if trial == sorted(trial_types)[0] and len(trial_types) > 1:
            off_trial = sorted(trial_types)[1]
        else:
            off_trial = sorted(trial_types)[0]
        for stimulus in sorted(stimuli):
            # Hackily, we need to know what the 'other' stimulus is for file naming
            if stimulus == sorted(stimuli)[0] and len(stimuli) > 1:
                off_stimulus = sorted(stimuli)[1]
            else:
                off_stimulus = sorted(stimuli)[0]
            for event in ppi_events:

                # Configure trial-based_'stimulus-all' entries.
                name = f"trial-{trial}_stimulus-all_pos-{stimulus}_neg-{off_stimulus}"
                if name not in timing_tables:
                    timing_tables[name] = []
                timing_tables[name].append({
                    "onset": event[f"{trial}_onset"],
                    "duration": event[f"{trial}_duration"],
                    "value": 1 if event["stimulus"] == stimulus else -1,
                })

                # Configure 'trial-all'_stimulus-based entries.
                name = f"trial-all_stimulus-{stimulus}_pos-{trial}_neg-{off_trial}"
                if name not in timing_tables:
                    timing_tables[name] = []
                if event["stimulus"] == stimulus:
                    timing_tables[name].append({
                        "onset": event[f"{trial}_onset"],
                        "duration": event[f"{trial}_duration"],
                        "value": 1,  # obviously (trial == trial)
                    })
                    for other_trial in [tt for tt in trial_types if tt != trial]:
                        timing_tables[name].append({
                            "onset": event[f"{other_trial}_onset"],
                            "duration": event[f"{other_trial}_duration"],
                            "value": -1,  # obviously (other_trial != trial)
                        })

    if args.verbose:
        print(f"Created {len(timing_tables)} ppi tables for saving.")

    return timing_tables


def do_feat(data, args):
    """ Extract events for FSL Feat analyses. """

    # For a quick first-pass, collect source values, if necessary,
    # from anything specified as --use-response-from
    val_sources = []
    for idx, row in data.iterrows():
        # If we were asked to use a particular response, same or other,
        # store that response until it's used. This must happen in a first
        # pass, before the actual pass, because the source value may come
        # after the row where it is to be used.

        # Liberally match stimuli, forgiving capitalization or punctuation.
        supplied_stimuli = [
            "".join([c for c in stim.lower() if c.isalpha()])
            for stim in args.use_response_from
        ]
        this_stimulus = "".join([c for c in row['stimulus'].lower() if c.isalpha()])
        if (
                (row['trial_type'] in args.use_response_from) or
                (this_stimulus in supplied_stimuli)
        ):
            try:
                val_sources.append(str(int(row['response'])))
            except ValueError:
                val_sources.append("nan")

    if args.verbose:
        print(f"Found {len(val_sources)} source values.")

    # Now that we've collected the value sources, in order, proceed.
    # Separate trial_type events, but in a way that separates blocks if needed
    timing_tables = {}
    last_trial_type = ''
    for idx, row in data.iterrows():
        # Extract just the data we need for our smaller feat-friendly file.

        if row['trial_type'] in args.use_response_to:
            try:
                third_value = val_sources.pop(0)
            except ValueError:
                third_value = "1"
                print(f"  WARNING: converted '{row['response']}' as response "
                      f"to '{row['trial_type']}':'{row['stimulus']}' "
                      f"in '{str(args.events_file)}' to a 1 value for Feat.")
        else:
            third_value = "1"
            
        event = {
            "onset": float(row['onset']) - args.shift,
            "duration": float(row['duration']),
            "value": third_value,
        }

        # Figure out how to split up trial_types and stimuli
        if row['trial_type'] in args.split_on_stimulus:
            if row['trial_type'] in args.as_block:
                print(f"WARNING: treating '{row['trial_type']}' as a block "
                      "precludes splitting on stimulus. Entire blocks of "
                      f"{row['trial_type']} events will be grouped without "
                      "regard to stimulus.")
                # ignore stimulus, just worry about trial_type
                event_name = f"trial-{row['trial_type']}"
            else:
                event_name = trial_type_plus_stimulus(row['trial_type'],
                                                      row['stimulus'])
        else:
            event_name = f"trial-{row['trial_type']}"

        # Depending on context, store this event or append it to one.
        if event_name in timing_tables:
            if row['trial_type'] in args.as_block:
                if last_trial_type == event_name:
                    # This event is part of an existing contiguous block,
                    # so we should add its duration to the prior event,
                    # and not store it by itself.
                    same_event = timing_tables[event_name][-1]
                    same_event['duration'] = (
                        event['onset'] + event['duration'] - same_event['onset']
                    )
                else:
                    # This event_name exists, and is specified as a block,
                    # but is not contiguous with this event, so this event is new.
                    timing_tables[event_name].append(event)
            else:
                # This event_name exists, but is not specified as a block,
                # so it is a new event.
                timing_tables[event_name].append(event)
        else:
            # event_name not yet in timing_Tables, create a new list of events
            timing_tables[event_name] = [event, ]

        # Remember last_trial_type to detect continuing blocks of same trials
        # same as event_name, only matters with multi-event blocks
        last_trial_type = row['trial_type']

    return timing_tables


def main(args):
    """ Entry point """

    if args.verbose:
        print("Extracting from {}, shifting by {:0.2f} seconds.".format(
            args.events_file, args.shift,
        ))

    # Load data
    metadata = metadata_from_path(args.events_file)
    data = pd.read_csv(args.events_file, sep='\t')

    # Check some assumptions
    available_trial_types = list(data['trial_type'].unique())
    available_stimuli = list(data['stimulus'].unique())

    # Get errors and mis-formed requests out of the way first
    error_response = handle_errors(args, available_trial_types, available_stimuli)

    if error_response > 0:
        print("       Available trial_types:")
        for t in available_trial_types:
            print(f"       - {t}")
        sys.exit(error_response)

    # Figure out whether we need to build standard feat event files or
    # ppi style files.
    if args.ppi_trial_types:
        timing_tables = do_ppi(data, args)
        did_ppi = True
    else:
        timing_tables = do_feat(data, args)
        did_ppi = False

    # Save the separate event types to separate files
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    for event_name in sorted(timing_tables.keys()):
        # Convert each list of dicts to an ordered 3-column dataframe.
        # 'onset' has been stored as a float, so ordering is robust.
        relevant_data = pd.DataFrame(timing_tables[event_name])[
            ['onset', 'duration', 'value']
        ].sort_values('onset')

        orig_trial_type = event_name.split("_")[0].split("-")[1]

        # Name the files to indicate how their rows were generated
        if orig_trial_type in args.as_block:
            descriptor = "blocks"
        else:
            descriptor = "events"

        # Name the files to indicate how their third column was generated
        if did_ppi:
            weight = "as-ppi"
        elif orig_trial_type in args.use_response_to:
            idx = args.use_response_to.index(orig_trial_type)
            response_from = args.use_response_from[idx]
            from_str = "".join([c for c in response_from.lower() if c.isalpha()])
            weight = f"as-{from_str}responses"
        else:
            weight = "as-ones"

        # Name the files as long-and-BIDSy or short
        if args.long_name:
            filename = "_".join([
                f"sub-{metadata['sub']}",
                f"task-{metadata['task']}",
                f"run-{metadata['run']}",
                event_name,
                weight,
                descriptor,
            ]) + ".txt"
        else:
            filename = "_".join([
                event_name,
                weight,
                descriptor,
            ]) + ".txt"

        # We may need to avoid writing data without responses to some files.
        writable_data = relevant_data[relevant_data['value'] != 'nan']
        writable_data.to_csv(
            Path(args.output_path) / filename,
            sep='\t', index=None, header=None, float_format="%.3f",
        )
        if args.verbose:
            print(filename)
            print(writable_data)

        # And write them to their own non-response files.
        errant_data = relevant_data[relevant_data['value'] == 'nan']
        if len(errant_data) > 0:
            errant_data.to_csv(
                Path(args.output_path) / filename.replace("responses_events", "responsefailure_events"),
                sep='\t', index=None, header=None, float_format="%.3f",
            )
            if args.verbose:
                print(filename.replace("responses_events", "responsefailure_events"))
                print(errant_data)



if __name__ == "__main__":
    main(get_arguments())
