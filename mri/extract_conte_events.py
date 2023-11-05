#!/usr/bin/env python3

""" extract_conte_events.py

    Originally written to extract events from an ePrime-derived Excel
    file. Later modified to handle either Excel or text, with different
    formats. Either will generate multiple BIDS-compliant events.tsv files.

    Use `extract_conte_events.py --help` for usage details
"""

import argparse
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from numbers import Number

# Trigger printing in red to highlight problems
RED_ON = '\033[91m'
GREEN_ON = '\033[92m'
COLOR_OFF = '\033[0m'


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="From an excel or text file exported from ePrime and "
                    "containing events and timings of an experiment, "
                    "export BIDS-compatible tsvs and jsons.",
    )
    parser.add_argument(
        "events_file",
        help="the events file, could be xlsx or txt",
    )
    parser.add_argument(
        "output_dir",
        help="where to write out the tsv and json files",
    )
    parser.add_argument(
        "--subject", default="None",
        help="optionally, specify the subject id to use in BIDS naming",
    )
    parser.add_argument(
        "--session", default="None",
        help="optionally, specify the session id to use in BIDS naming",
    )
    parser.add_argument(
        "-s", "--shift", type=float, default=0.0,
        help="subtract this amount from each start time",
    )
    parser.add_argument(
        "--rawdata", default="/mnt/rawdata/new_conte",
        help="the path to BIDS-valid rawdata folder for looking up sessions",
    )
    parser.add_argument(
        "--run-duration", type=float, default="0.0",
        help="if a waitfornextrun event duration is desired, subtract from this"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    parsed_args = parser.parse_args()

    # Figure out subject and session from output_dir
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    for path_piece in str(parsed_args.output_dir).split("/"):
        if path_piece.startswith("sub-"):
            if parsed_args.subject == "None":
                setattr(parsed_args, "subject", path_piece[4:])
        elif path_piece.startswith("ses-"):
            if parsed_args.session == "None":
                setattr(parsed_args, "session", path_piece[4:])

    if (parsed_args.subject != "None") and (parsed_args.session == "None"):
        setattr(parsed_args, "session",
                find_session(parsed_args.subject, rawdata=parsed_args.rawdata))

    if parsed_args.subject == "None":
        print(f"{RED_ON}"
              f"Can't determine subject in '{parsed_args.output_dir}'"
              f"{COLOR_OFF}")
        print(f"{RED_ON}"
              "Outputs will NOT be BIDS compatible without sub- and ses-."
              f"{COLOR_OFF}")
        print(f"{RED_ON}"
              "You can specify a subject with [--subject SUBJECT_ID]."
              f"{COLOR_OFF}")
    if parsed_args.session == "None":
        print(f"{RED_ON}"
              f"Can't determine session in '{parsed_args.output_dir}'"
              f"{COLOR_OFF}")
        print(f"{RED_ON}"
              "Outputs will NOT be BIDS compatible without sub- and ses-."
              f"{COLOR_OFF}")
        print(f"{RED_ON}"
              "You can specify a session with [--session SESSION_ID]."
              f"{COLOR_OFF}")

    if parsed_args.verbose:
        print(
            f"subject '{parsed_args.subject}', session '{parsed_args.session}'"
        )

    return parsed_args


def find_session(subject_id, rawdata="/mnt/rawdata"):
    """ From a subject id, find and return a session.

        :param str subject_id:
            A subject_id, without the "sub-" portion
        :param str rawdata:
            The base path to a BIDS-valid rawdata directory
        :returns str:
            The first session id corresponding to the subject
    """

    possibilities = list(Path(rawdata).glob(f"sub-{subject_id}/ses-*"))
    if len(possibilities) == 1:
        return possibilities[0].name[possibilities[0].name.rfind("-") + 1:]
    elif len(possibilities) == 0:
        return "None"
    else:
        print(f"  Subject {subject_id} has {len(possibilities)} sessions")
        for possibility in possibilities:
            niftis = list(possibility.glob(
                "func/sub-*_ses-*_task-*_run-*_bold.nii.gz"
            ))
            if len(niftis) > 0:
                return possibility.name[possibility.name.rfind("-") + 1:]


def row_as_mem_event(row):
    """ Interpret dataframe row as event, and return dict with relevant items.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to be treated as an event in a memory trial.
        :returns list:
            A list of dicts, each dict containing key-value pairs intended
            to become one row in a BIDS-valid events.tsv file
    """

    assert "Procedure[SubTrial]" in row
    assert "QText.OnsetTime" in row
    assert "QText.OnsetToOnsetTime" in row
    assert "Question" in row
    assert "QText.RESP" in row
    assert "QText.RT" in row
    assert "Arrow.OnsetTime" in row
    assert "Arrow.OnsetToOnsetTime" in row
    assert "Arrow" in row
    assert "Arrow.RESP" in row
    assert "Arrow.RT" in row
    assert "Block" in row
    assert "Trial" in row
    assert "SubTrial" in row

    # In the key column, 'Procedure[SubTrial]':
    # Memory sheets have only three values
    # 'Rating', 'Arrow', and 'ArrowB'

    values = []

    # What kind of event is this row describing?
    if (
            (row["Procedure[SubTrial]"] == "Rating")
            and np.isfinite(row["QText.OnsetTime"])
    ):
        response = "n/a"
        if str(row['QText.RESP']) != "nan":
            response = str(row["QText.RESP"])
        values.append({
            "block": int(row["Block"]),
            "onset": float(row["QText.OnsetTime"]) / 1000,
            "duration": float(row["QText.OnsetToOnsetTime"]) / 1000,
            "trial_type": "question",
            "stimulus": row["Question"],
            "response": response,
            "response_time": float(row["QText.RT"]) / 1000,
        })
    elif (
            row["Procedure[SubTrial]"].startswith("ArrowTask")
            and np.isfinite(row["Arrow.OnsetTime"])
    ):
        response = "n/a"
        if str(row['Arrow.RESP']) != "nan":
            response = str(row["Arrow.RESP"])
        values.append({
            "block": int(row["Block"]),
            "onset": float(row["Arrow.OnsetTime"]) / 1000,
            "duration": float(row["Arrow.OnsetToOnsetTime"]) / 1000,
            "trial_type": "arrow",
            "stimulus": row["Arrow"],
            "response": response,
            "response_time": float(row["Arrow.RT"]) / 1000,
        })

    if len(values) == 0:
        print("Procedure == {}, (block {}, trial {}, subtrial {})".format(
            row["Procedure[SubTrial]"],
            row["Block"], row["Trial"], row["SubTrial"]
        ))
        raise ValueError("Record not recognized!!")

    return values


def row_as_pic_event(row):
    """ Interpret dataframe row as event, and return dict with relevant items.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to be treated as an event in a picture trial.
        :returns list:
            A list of dicts, each dict containing key-value pairs intended
            to become one row in a BIDS-valid events.tsv file
    """

    assert "Procedure[SubTrial]" in row
    assert "neutStim.OnsetTime" in row
    assert "neutStim.OnsetToOnsetTime" in row
    assert "Stimulus" in row
    assert "emoStim.OnsetTime" in row
    assert "emoStim.OnsetToOnsetTime" in row
    assert "sameDiff.OnsetTime" in row
    assert "sameDiff.OnsetToOnsetTime" in row
    assert "sameDiff.RESP" in row
    assert "sameDiff.RT" in row
    assert "Block" in row
    assert "Trial" in row
    assert "SubTrial" in row

    # In the key column, 'Procedure[SubTrial]':
    # Picture sheets have only three values:
    # 'NeutTrialProc', 'EmoTrialProc', and ''

    values = []

    # What kind of event is this row describing?
    if "TrialProc" in str(row["Procedure[SubTrial]"]):
        # Both emot and neut have same_diff responses
        response, response_time = "n/a", "n/a"
        if str(row['sameDiff.RESP']) != "nan":
            response = str(row["sameDiff.RESP"])
        if str(row['sameDiff.RT']) != "nan":
            response_time = str(row["sameDiff.RT"])
        values.append({
            "block": int(row["Block"]),
            "onset": float(row["sameDiff.OnsetTime"]) / 1000,
            "duration": float(row["sameDiff.OnsetToOnsetTime"]) / 1000,
            "trial_type": "same_diff",
            "stimulus": row["Stimulus"],
            "response": response,
            "response_time": response_time,
        })
        # But emot and neut differ in where they keep their timings
        if (
                (row["Procedure[SubTrial]"] == "NeutTrialProc")
                and np.isfinite(row["neutStim.OnsetTime"])
        ):
            values.append({
                "block": int(row["Block"]),
                "onset": float(row["neutStim.OnsetTime"]) / 1000,
                "duration": float(row["neutStim.OnsetToOnsetTime"]) / 1000,
                "trial_type": "neut_stim",
                "stimulus": row["Stimulus"],
                "response": "n/a",
                "response_time": "n/a",
            })
        elif (
                (row["Procedure[SubTrial]"] == "EmoTrialProc")
                and np.isfinite(row["emoStim.OnsetTime"])
        ):
            values.append({
                "block": int(row["Block"]),
                "onset": float(row["emoStim.OnsetTime"]) / 1000,
                "duration": float(row["emoStim.OnsetToOnsetTime"]) / 1000,
                "trial_type": "emot_stim",
                "stimulus": row["Stimulus"],
                "response": "n/a",
                "response_time": "n/a",
            })

    # We should know 'Repeated', several 'sameDiff'-related columns,
    # 'arrowDisplay'-related columns

    # if len(values) == 0:
    # In picture tasks, we know we are missing RateNeg, RatePos,
    # and several RestStates. That's fine for now.
    # warning = (
    #     "Record unrecognized: "
    #     f"Procedure == {row['Procedure[SubTrial]']}, "
    #     f"(block {row['Block']}, trial {row['Trial']}, "
    #     f"subtrial {row['SubTrial']}, "
    #     f"sameDiff.OnsetTime {row.get('sameDiff.OnsetTime', 'None')}, "
    #     f"neutStim.OnsetTime {row.get('neutStim.OnsetTime', 'None')}, "
    #     f"emoStim.OnsetTime {row.get('emoStim.OnsetTime', 'None')})"
    # )
    # print(warning)
    # raise ValueError("Record not recognized!!")
    return values


def row_as_mem_primary_event(row):
    """ Interpret dataframe row as a scanner marker, not an experimental event.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to be interpreted as a 'memory' and 'instruct' in a memory trial.
        :returns list:
            A list of dicts, each dict containing key-value pairs intended
            to become one row in a BIDS-valid events.tsv file
    """

    assert "MemCue.OnsetTime" in row
    assert "MemCue.OnsetToOnsetTime" in row
    assert "Instruct.OnsetTime" in row
    assert "Instruct.OnsetToOnsetTime" in row
    assert "Block" in row
    assert "MemCue" in row
    assert "Word" in row

    # Collect all onset times and offset times.
    # Collecting these row-by-row creates duplicates that need to be removed.
    nonevents = [
        {
            "block": int(row["Block"]),
            "onset": float(row["MemCue.OnsetTime"]) / 1000,
            "duration": float(row["MemCue.OnsetToOnsetTime"]) / 1000,
            "trial_type": "memory",
            "stimulus": row["MemCue"],
            "response": "n/a",
            "response_time": 0.0,
        },
        {
            "block": int(row["Block"]),
            "onset": float(row["Instruct.OnsetTime"]) / 1000,
            "duration": float(row["Instruct.OnsetToOnsetTime"]) / 1000,
            "trial_type": "instruct",
            "stimulus": row["Word"],
            "response": "n/a",
            "response_time": 0.0,
        },
    ]
    return nonevents


def row_as_pic_arrow_event(row):
    """ Interpret dataframe row as a scanner marker, not an experimental event.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to be treated as an arrow event in a picture trial.
        :returns list:
            A list of dicts, each dict containing key-value pairs intended
            to become one row in a BIDS-valid events.tsv file
    """

    assert "arrowDisplay.OnsetTime" in row
    assert "arrowDisplay.OnsetToOnsetTime" in row
    assert "arrowDisplay.RESP" in row
    assert "arrowDisplay.RT" in row
    assert "Block" in row
    assert "Trial" in row
    assert "SubTrial" in row
    assert "FlankersDir" in row

    # Collect all onset times and offset times.
    # Collecting these row-by-row creates duplicates that need to be removed.
    nonevents = []
    if np.isfinite(row['arrowDisplay.OnsetTime']):
        response, response_time = "n/a", "n/a"
        if str(row['arrowDisplay.RESP']) != "nan":
            response = str(row["arrowDisplay.RESP"])
        if str(row['arrowDisplay.RT']) != "nan":
            response_time = str(row["arrowDisplay.RT"])
        nonevents.append({
            "block": int(row["Block"]),
            "onset": float(row["arrowDisplay.OnsetTime"]) / 1000,
            "duration": float(row["arrowDisplay.OnsetToOnsetTime"]) / 1000,
            "trial_type": "arrow",
            "stimulus": row["FlankersDir"],
            "response": response,
            "response_time": response_time,
        })
    return nonevents


def row_timestamps(row):
    """ Get the beginning timestamp for each block.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to extract timestamp data, used to delineate blocks of events.
        :returns list:
            A list of dicts, each dict containing block-boundary timing
            information
    """

    timestamps = []
    for col in row.index:
        if (
                col.startswith("SynchWithScanner")
                and col.endswith("OffsetTime")
                and np.isfinite(row[col])
        ):
            # This is a timestamp for a memory block
            timestamps.append({
                "block": int(row["Block"]),
                "onset": float(row[col]) / 1000,
                "duration": 0.0,
                "trial_type": "begin_block",
                "stimulus": "n/a",
                "response": "n/a",
                "response_time": 0.0,
            })
        if (
                (col == "scannerTrig.RTTime")
                and np.isfinite(row[col])
        ):
            # This is a timestamp for a pictures/images block
            timestamps.append({
                "block": int(row["Block"]),
                "onset": float(row[col]) / 1000,
                "duration": 0.0,
                "trial_type": "begin_block",
                "stimulus": "n/a",
                "response": "n/a",
                "response_time": 0.0,
            })

    return timestamps


def timestamp_map(sheet):
    """ From an excel-like sheet, extract a map of block_id->start_time.

        :param DataFrame sheet:
            The contents of an Excel sheet, read as a pandas DataFrame,
            should have been exported directly from ePrime
        :returns dict:
            A dict with block numbers as keys and onset values as values
    """

    ts_df = pd.DataFrame(list(
        sheet.apply(row_timestamps, axis=1).explode().dropna().drop_duplicates()
    ))
    ts_map = {}
    # The blocks must be sequential, starting with 1, to be used as run #s
    for i, block_id in enumerate(sorted(ts_df['block'].unique())):
        ts_map[i + 1] = ts_df[ts_df['block'] == block_id]['onset'].values[0]
    return ts_map


def make_sidecar(task):
    """ For a given task, create a json file describing tsv fields

        :param str task:
            The task descriptor for a trial: 'mem' or 'pic' in new_conte
        :returns dict:
            A nested dict appropriate for direct json serialization
    """

    # The events.tsv files created in this script contain four columns
    # expected by BIDS. They also contain additional columns that should
    # be described in an accompanying sidecar file.
    if task == "mem":
        levels = {
            "memory": "A 10 second period after a memory cue is presented",
            "instruct": "A 10 second period during which the participant "
                        "is instructed to immerse themselves in the "
                        "memory or distance themselves from the memory.",
            "question": "A 3-ish second period in which the participant "
                        "is asked about the memory's effect.",
            "arrow": "A variable duration period where the participant "
                     "is asked to identify the direction of an arrow.",
        }
    elif task == "pic":
        levels = {
            "neut_stim": "A 2 second period while a neutral picture is "
                         "presented to the participant",
            "emot_stim": "A 2 second period while an emotional picture is "
                         "presented to the participant",
            "same_diff": "A 1.5 second period in which the participant "
                         "is asked whether the picture is a repeat.",
            "arrow": "A variable duration period where the participant "
                     "is asked to identify the direction of an arrow.",
        }
    else:
        levels = {}
    data = {
        "trial_type": {
            "LongName": "Event category",
            "Description": "Indicator of type of action that is expected",
            "Levels": levels
        },
        "stimulus": {
            "Description": "The stimulus presented to the participant",
        },
        "response": {
            "Description": "The participant's response to the stimulus",
        }
    }
    return data


def ms_to_sec(ms, precision=3, to_str=False):
    """ Convert milliseconds to seconds

        :param ms:
            Any type that can be converted to float, written for a str
            from an ePrime Excel sheet
        :param int precision:
            The precision used for formatting the seconds value to a str,
            defaults to 3
        :param bool to_str:
            By default, return seconds as a float, but if to_str is True,
            return the seconds formatted to a str with precision specified
        :returns str:
            ms divided by 1000, and formatted as specified
    """
    if to_str:
        return f"{(float(ms) / 1000.0):0.{precision}f}"
    else:
        return float(ms) / 1000.0


def parse_file(file, verbose=False):
    """ Make sense of the file's contents and return relevant data.

        :param Path file:
            The path to the ePrime-emitted file to parse. Text files
            are preferred as they include more duration information.
            Duration must be inferred by gaps between events when
            using Excel files.
        :param bool verbose:
            Verbosity flag, set true for text output to console
        :returns list:
            A list of dicts, each key-value item representing a 'frame'
            from an ePrime text output file. One frame is all data contained
            between a '*** Start _ ***' and the next '*** End _ ***' line.
    """

    # Match things like '*** Header Start ***' or '    *** LogFrame End ***'
    pattern_wrapper = re.compile(r"[*][*][*]\s+(\S+)\s+(\S+)\s+[*][*][*]")
    # Match things like '    Procedure: Rating' or '    QText.RT: 500'
    key_value = re.compile(r"(\S+?.*?):\s+(.*)$")
    line_count = 0

    # Most of the ePrime files are not utf-8.
    # They may be 'ISO-8859-1', which is an old default from 40 years ago.
    # Or they are sometimes utf-16-LE. We need to figure out which is which.
    # The most common problem is when they have a special character, 0xa0.
    # These are handled by the errors="replace" and subsequent .replace().
    default_encoding = 'utf-16'
    f = open(file, "r", encoding=default_encoding)
    try:
        f.readline()
        encoding = 'utf-16'
        if verbose:
            print(f"File '{file}' must be '{encoding}'-encoded.")
    except UnicodeDecodeError as e:
        print(f"caught a UnicodeDecodeError {e}")
        encoding = 'utf-8'
    except UnicodeError as e:
        print(f"caught a UnicodeError {e}")
        encoding = 'utf-8'
    f.close()

    # Now that we have a guess for the file type, try to read it.
    # Using errors="replace" automatically replaces the bad characters with
    # another legal-but-unusable character chr(65533).
    # I wanted to be smart and swap spaces for known space-like characters,
    # but using 'surrogateescape' seems to make it even more difficult
    # to figure out what the hell the bad characters were and what to do
    # with them.
    try:
        error_handling = "replace"  # not "surrogateescape"
        with open(file, "r", encoding=encoding, errors=error_handling) as f:
            frames = []
            for i, line in enumerate(f):
                # Manage Start and Stop markers for frames
                match = pattern_wrapper.search(line)
                if match:
                    if match.group(2) == "Start":
                        frame = {'line': i, 'type': match.group(1), }
                    elif match.group(2) == "End":
                        frames.append(frame)

                # Manage data between frames
                # Many of these should be encountered between "Start" and "End"
                match = key_value.search(line)
                if match:
                    # The replace function handles 'replaced' bad characters
                    clean_val = match.group(2).strip()
                    clean_val = clean_val.replace("\"", "")
                    clean_val = clean_val.replace(chr(65533), " ")
                    frame[match.group(1)] = clean_val
                line_count = i

    except UnicodeDecodeError as ue:
        print(f"  skipping file with unknown encoding '{file.name}' "
              f"({i}, '{line.rstrip()}') '{ue}'")

    if verbose:
        print(f"Read {line_count} lines, found {len(frames)} frames.")
        # This level of verbosity only appropriate for extensive debugging
        # for i, frame in enumerate(frames):
        #     for k in frame.keys():
        #         if "Onset" in k:
        #             print(f"  frame {i:>3}. '{k}': '{frame[k]}'")

    return frames


def frames_to_dataframe(task, frames, verbose=False):
    """ Convert list of dicts to dataframe, keeping only what we want.

        :param str task:
            The task label for the trial being extracted
        :param list frames:
            A list of dicts, each dict representing one 'frame' from an
            ePrime-exported text file
        :param bool verbose:
            Set to True for more verbose output

        :returns: tuple (data, metadata)
            WHERE
            pd.DataFrame data is A concatenation of all dicts in frames
            dict metadata is timing information for splitting a long text file
                             into multiple fMRI runs
    """

    rows = []
    metadata = {'task': task, }
    for i, frame in enumerate(frames):
        if verbose:
            pass
            # Very verbose if debugging
            # print(f"Reading frame {i}, from line {frame['line'] + 1}")
        if (frame['type'] == "Header") and ("SessionDate" in frame):
            print(f"  Found header with subject '{frame['Subject']}' "
                  f"from {frame['SessionDate']} at {frame['SessionTime']}.")
        elif "Running" in frame:

            if task == "image":
                if frame['Running'] == "Intro":
                    pass
                elif frame['Running'] in [
                    "RunN1", "RunN2", "RunN3", "RunP1", "RunP2", "RunP3",
                ]:
                    # Just a label of a group of trials with no trial data
                    pass
                elif frame['Running'] in [
                    "NegativeImageTask", "PositiveImageTask",
                ]:
                    # A wrapper frame for bunches of trials with no trial data
                    # But the trigger and fixation screens still have onset data
                    # One of these is the start time for each of six runs.
                    metadata[frame['Procedure']] = ms_to_sec(
                        frame["LeadingFixation.OnsetTime"]
                    )
                elif frame['Running'] == "Tasks":
                    # The final label with no real trial data
                    pass
                elif frame['Running'] == "ratings":
                    if frame['Procedure'] == "rn":
                        prefix = "ratingnegSlide."
                    elif frame['Procedure'] == "rp":
                        prefix = "ratingposSlide."
                    else:
                        prefix = ""
                        print("  ERROR: "
                              f"No interpretation of '{frame['Procedure']}' "
                              f"procedure in 'ratings' frame on line "
                              f"{frame['line']}")
                    if len(prefix) > 0:
                        rows.append({
                            'trial_type': "rating",
                            'onset': ms_to_sec(frame[prefix + 'OnsetTime']),
                            'duration': ms_to_sec(frame[prefix + 'RT']),
                            'stimulus': frame['Procedure'],
                            'response': frame[prefix + 'RESP'],
                            'response_time': ms_to_sec(frame[prefix + 'RT']),
                        })
                elif frame['Running'] in [
                    "EmoPics", "NeutPics1", "NeutPics2",
                ]:
                    # These are more like section headings than data frames.
                    # None of them have event data within, but they do indicate
                    # run boundaries.
                    if "scannerTrig.RTTime" in frame:
                        metadata[frame["Procedure"]] = ms_to_sec(
                            frame["scannerTrig.RTTime"]
                        )
                    # else ignore this frame
                elif frame['Running'].startswith("Rate"):
                    # The example I used has Rate1-Rate8,
                    # Ignore RateProc because they occur before/after BOLD;
                    # and they therefore cannot fit into models.
                    if "ratePos.OnsetTime" in frame:
                        prefix = "ratePos"
                        if verbose:
                            print("  ignoring rating @ "
                                  f"{frame[prefix + '.OnsetTime']}")
                    elif "rateNeg.OnsetTime" in frame:
                        prefix = "rateNeg"
                        if verbose:
                            print("  ignoring rating @ "
                                  f"{frame[prefix + '.OnsetTime']}")
                    # rows.append({
                    #     'trial_type': "rating",
                    #     'onset': ms_to_sec(frame[prefix + '.OnsetTime']),
                    #     'duration': ms_to_sec(frame[prefix + '.RT']),
                    #     'stimulus': frame['Procedure'],
                    #     'response': frame[prefix + '.RESP'],
                    #     'response_time': ms_to_sec(frame[prefix + '.RT']),
                    # })
                elif frame['Running'].startswith("RestState"):
                    # A typical session has 12 of these,
                    # but they have no timing data,
                    # no response data, nothing useful at all
                    pass
                elif frame['Running'] in [
                    "Neut1ProcsList", "Neut2ProcsList",
                    "ArrowProcsList", "EmoProcsList",
                ]:
                    # These are more like section headings than data frames.
                    # None of them have actionable data within.
                    pass
                elif frame['Running'].startswith("DirsList"):
                    # The first example file had 405 of these,
                    # DirsList{4s,6s,8s}
                    rows.append({
                        'trial_type': "arrow",
                        'onset': ms_to_sec(frame['arrowDisplay.OnsetTime']),
                        'duration': ms_to_sec(frame['arrowDisplay.Duration']),
                        'stimulus': frame['FlankersDir'],
                        'response': frame['arrowDisplay.RESP'],
                        'response_time': ms_to_sec(frame['arrowDisplay.RT']),
                    })
                elif frame['Running'].startswith("NeutStims"):
                    # The example I used has 90 of these,
                    # 45 NeutStimsA, 45 NeutStimsB
                    rows.append({
                        'trial_type': "neutral_image",
                        'onset': ms_to_sec(frame['neutStim.OnsetTime']),
                        'duration': ms_to_sec(frame['neutStim.Duration']),
                        'stimulus': frame['Stimulus'],
                        'response': frame.get('neutStim.RESP', 'n/a'),
                        'response_time': ms_to_sec(frame['neutStim.RT']),
                    })
                    # I encountered only one file that did not have a matching
                    # 'sameDiff' set, but it caused a failure. The next clause
                    # should run in every other case.
                    if 'sameDiff.OnsetTime' in frame:
                        rows.append({
                            'trial_type': "same_diff",
                            'onset': ms_to_sec(
                                frame['sameDiff.OnsetTime']
                            ),
                            'duration': ms_to_sec(
                                frame['sameDiff.OnsetToOnsetTime']
                            ),
                            'stimulus': "n/a",
                            'response': frame['sameDiff.RESP'],
                            'response_time': ms_to_sec(
                                frame['sameDiff.RT']
                            ),
                        })
                elif frame['Running'].startswith("EmoStims"):
                    # The example I used has 45 of these, all EmoStimsX
                    rows.append({
                        'trial_type': "emo_image",
                        'onset': ms_to_sec(frame['emoStim.OnsetTime']),
                        'duration': ms_to_sec(frame['emoStim.Duration']),
                        'stimulus': frame['Stimulus'],
                        'response': frame.get('emoStim.RESP', 'n/a'),
                        'response_time': ms_to_sec(frame['emoStim.RT']),
                    })
                    rows.append({
                        'trial_type': "same_diff",
                        'onset': ms_to_sec(
                            frame['sameDiff.OnsetTime']
                        ),
                        'duration': ms_to_sec(
                            frame['sameDiff.OnsetToOnsetTime']
                        ),
                        'stimulus': "n/a",
                        'response': frame['sameDiff.RESP'],
                        'response_time': ms_to_sec(
                            frame['sameDiff.RT']
                        ),
                    })
                elif frame['Running'] in [
                    "N1a", "N2a", "N3a", "N1b", "N2b", "N3b",
                    "P1a", "P2a", "P3a", "P1b", "P2b", "P3b",
                ]:
                    # This frame contains several pieces:
                    # cueSlide, stimSlide, isiSlide, itiSlide
                    rows.append({
                        'trial_type': frame['trialtype'],
                        'onset': ms_to_sec(
                            frame.get('cueSlide.OnsetTime', '0')
                        ),
                        'duration': 0.0,
                        'stimulus': frame['cue'],
                        'response': "n/a", 'response_time': 0.0,
                    })
                    rows.append({
                        'trial_type': frame['trialtype'],
                        'onset': ms_to_sec(
                            frame.get('stimSlide.OnsetTime', '0')
                        ),
                        'duration': 0.0,
                        'stimulus': frame['stim'],
                        'response': "n/a",
                        'response_time': 0.0,
                    })
                    rows.append({
                        'trial_type': "isi",
                        'onset': ms_to_sec(
                            frame.get('isiSlide.OnsetTime', '0')
                        ),
                        'duration': ms_to_sec(frame['isi']),
                        'stimulus': "n/a",
                        'response': "n/a",
                        'response_time': 0.0,
                    })
                    rows.append({
                        'trial_type': "iti",
                        'onset': ms_to_sec(
                            frame.get('itiSlide.OnsetTime', '0')
                        ),
                        'duration': ms_to_sec(frame['iti']),
                        'stimulus': "n/a",
                        'response': "n/a",
                        'response_time': 0.0,
                    })
                else:
                    print(f"  Frame {i} (line {frame['line']}) missed. "
                          f"{task}: Running=='{frame['Running']}'")

            if task == "mem":
                if frame['Running'] in ["RatingList", "QList", "QList2", ]:
                    # In these 'Running's, either QText.RT or
                    # QText.OnsetToOnsetTime is always '0'
                    # and the other is non-zero.
                    # So for 'duration', we will use whichever is non-zero.
                    reaction_time = ms_to_sec(
                        frame.get('QText.RT', frame.get('Qtext.RT', '0'))
                    )
                    if reaction_time > 0.001:
                        duration = reaction_time
                    else:
                        duration = ms_to_sec(
                            frame.get('QText.OnsetToOnsetTime', '0')
                        )
                    rows.append({
                        'trial_type': "question",
                        'onset': ms_to_sec(
                            frame.get('QText.OnsetTime',
                                      frame.get('Qtext.OnsetTime', '0'))
                        ),
                        'duration': duration,
                        'stimulus': frame.get('Question', "n/a"),
                        'response': frame.get('QText.RESP',
                                              frame.get('Qtext.RESP', "n/a")),
                        'response_time': reaction_time,
                        'block': "n/a",
                    })
                elif frame['Running'].startswith("Arrow"):
                    rows.append({
                        'trial_type': "arrow",
                        'onset': ms_to_sec(frame['Arrow.OnsetTime']),
                        'duration': ms_to_sec(frame['Arrow.RT']),
                        'stimulus': frame['Arrow'],
                        'response': frame['Arrow.RESP'],
                        'response_time': ms_to_sec(frame['Arrow.RT']),
                        'block': "n/a",
                    })
                elif frame['Running'].startswith("WordList"):
                    rows.append({
                        'trial_type': "memory",
                        'onset': ms_to_sec(
                            frame.get('MemCue.OnsetTime', '0')
                        ),
                        'duration': ms_to_sec(
                            frame.get('MemCue.OnsetToOnsetTime', '0')
                        ),
                        'stimulus': frame['MemCue'],
                        'response': "n/a",
                        'response_time': "n/a",
                        'block': frame['Running'][-1],
                    })
                    rows.append({
                        'trial_type': "instruct",
                        'onset': ms_to_sec(
                            frame.get('Instruct.OnsetTime', '0')
                        ),
                        'duration': ms_to_sec(
                            frame.get('Instruct.OnsetToOnsetTime', '0')
                        ),
                        'stimulus': frame['Word'].lower(),
                        'response': "n/a",
                        'response_time': "n/a",
                        'block': frame['Running'][-1],
                    })
                elif frame['Running'] == "RecallTaskProc":
                    rows.append({
                        'trial_type': "memory",
                        'onset': ms_to_sec(
                            frame['MemCue.OnsetTime']
                        ),
                        'duration': ms_to_sec(
                            frame.get('MemCue.OnsetToOnsetTime', '0')
                        ),
                        'stimulus': frame['Memory'],
                        'response': "n/a",
                        'response_time': "n/a",
                        'block': "n/a",
                    })
                elif frame['Running'] in [
                    "SelfPracProc", "TimePracProc",
                ]:
                    rows.append({
                        'trial_type': "instruct",
                        'onset': 0.0,
                        'duration': 0.0,
                        'stimulus': frame['Instruction'].lower(),
                        'response': "n/a",
                        'response_time': "n/a",
                        'block': "n/a",
                    })
                elif frame['Running'].startswith("RunList"):
                    # Synchronization info between ePrime and the scanner
                    last_proc = frame['Procedure'][-1]
                    off_key = f"SynchWithScanner{last_proc}.OffsetTime"
                    if off_key in frame:
                        metadata[frame['Procedure']] = ms_to_sec(frame[off_key])
                        if verbose:
                            print(f"  RunList synch @ {int(frame[off_key]):,}")
                    else:
                        # This is still a block boundary
                        metadata[frame['Procedure']] = 0.0
                        if verbose:
                            print(f"  RunList block without synch")
                else:
                    print(f"  Frame {i} (line {frame['line']}) missed. "
                          f"{task}: Running=='{frame['Running']}'")

        else:
            pass
            # Every normal file has exactly one "LogFrame" without 'Running'
            # No need to report it as it's not a problem.
            # print(f"  Frame {i}, line {frame['line']}. "
            #       "'Running' key not in frame.")

    if verbose:
        print(f"  collected {len(rows)} frames")

    return pd.DataFrame(rows), metadata


def finalize_dataframe(df, offsets, verbose=False):
    """ Use the offsets to shift dataframe's 'onset' values
        and fill in missing data.

        :param pd.DataFrame df:
            A pandas DataFrame containing all frames from the first pass
        :param dict offsets:
            metadata containing timepoints for splitting one text file into
            multiple events.tsv files, one for each fMRI run
        :param verbose:
            Set to True for more verbose console output

        :returns pd.DataFrame:
            A cleaned, validated, filled-in dataframe derived from 'df'
    """

    # Rows must be in chronological order to calculate durations
    # And indices from 0-length allow for correct indexing with loc
    df = df.sort_values(
        'onset', key=lambda x: x.astype(float)
    ).reset_index(drop=True)

    # Fill in durations of 0 with calculated durations.
    additions = []
    for i in range(len(df)):
        # Fill in zeroes with the size of the temporal gap they fall within
        if (i < len(df) - 1) and (df.loc[i, 'duration'] == 0.0):
            # These studies use 10- or 20-second events, but when the ISI
            # is missing, it's hard to tell. So fix them here.
            # First, use the calculated value as a default, then override it
            # if it looks reasonable to do so.
            duration = df.loc[i + 1, 'onset'] - df.loc[i, 'onset']
            df.loc[i, 'duration'] = duration
            if df.loc[i, 'trial_type'] == "memory":
                if (duration >= 9.99) and (duration <= 17.50):
                    df.loc[i, 'duration'] = 10.000
            if df.loc[i, 'trial_type'] == "instruct":
                if (duration >= 9.99) and (duration <= 17.50):
                    df.loc[i, 'duration'] = 10.000
                if (duration >= 19.99) and (duration <= 27.50):
                    df.loc[i, 'duration'] = 20.000
            if df.loc[i, 'trial_type'] == "question":
                if duration > 3.000:
                    df.loc[i, 'duration'] = 3.000

        # In the 6-second gaps where the Arrows directions go, add that block
        if (
               (i < len(df) - 1) and
               (df.loc[i, 'trial_type'] == 'question') and
               (df.loc[i + 1, 'trial_type'] == 'arrow')
        ):
            ai_onset_estimate = df.loc[i, 'onset'] + df.loc[i, 'duration'] + 1.0
            additions.append({
                "onset": ai_onset_estimate,
                "duration": df.loc[i + 1, 'onset'] - ai_onset_estimate,
                "trial_type": "directions",
                "response_time": "n/a",
                "stimulus": "arrows directions",
                "response": "n/a",
            })

        # Blanks are not BIDS-valid; fill them with "n/a"s
        if df.loc[i, 'response'] == '':
            df.loc[i, 'response'] = 'n/a'
        if df.loc[i, 'response_time'] == '':
            df.loc[i, 'response_time'] = 'n/a'
        if df.loc[i, 'response'] == 'n/a':
            df.loc[i, 'response_time'] = 'n/a'

    if verbose:
        print(f"  added {len(additions)} events to {len(df)}-event dataframe")
    df = pd.concat([df, pd.DataFrame(additions), ])
    df = df.sort_values('onset').reset_index(drop=True)

    # Format offsets nicely for splitting timing into separate fMRI BOLD runs
    sync_times = []
    sync_df = pd.DataFrame(data=None)
    for k, v in offsets.items():
        if k != 'task':
            if v > 0.0:
                # This is a run boundary, and we have synch information
                sync_times.append({'t': v, 'p': k, })
            else:
                # We have a run boundary, but did not get Synch information.
                # This only runs when we have a BPD file missing Synch info,
                # and there is no "fixed" table to pull it from.
                earliest_onset = df[df['block'] == k[-1]]['onset'].min()
                sync_times.append({'t': earliest_onset - 8.0, 'p': k, })
                print("WARNING: Forced to guess scanner synch at 8 seconds.")

    try:
        # Extract runs from metadata
        sync_df = pd.DataFrame(sync_times)
        sync_df = sync_df.sort_values('t').reset_index(drop=True)
        sync_df['run'] = [i + 1 for i in range(len(sync_df))]

        # Center each 'onset' to its run's start time.
        df = df.rename(columns={'onset': 'original_onset'})
        df['onset'] = df['original_onset'].apply(
            lambda onset: onset - sync_df[sync_df['t'] < onset].max().iloc[0]
        )
        df['onset'] = df['onset'].fillna(0.0)
        df['run'] = df['original_onset'].apply(
            lambda onset: 0 if onset < sync_df['t'].min() else sync_df.loc[
                sync_df[sync_df['t'] < onset]['t'].idxmax(),
                'run'
            ]
        )

    except KeyError as key_error:
        print(f"  ERROR: KeyError {key_error}")
    except ValueError as value_error:
        print(f"  ERROR: ValueError {value_error}")
        print(sync_df)
        print("Onsets range: ", df['onset'].min(), df['onset'].max())
    return df


def prettify_floats_to_strings(df):
    """ """

    # Ensure we write out pretty text to the tsv
    # This converts numerics to strings,
    # so best to do this after all calculations
    columns = df.columns
    df = df.rename(
        columns={
            'onset': '_onset',
            'duration': '_duration',
            'response_time': '_response_time',
        }
    )
    df['onset'] = df['_onset'].apply(
        lambda t: "n/a" if not isinstance(t, Number) else f"{float(t):0.3f}"
    )
    df['duration'] = df['_duration'].apply(
        lambda t: "n/a" if not isinstance(t, Number) else f"{float(t):0.3f}"
    )
    df['response_time'] = df['_response_time'].apply(
        lambda t: "n/a" if not isinstance(t, Number) else f"{float(t):0.3f}"
    )
    return df[columns]


def add_end_of_run_wait_events(df, run_length):
    """ Add 'wait' event when participant completed a run """

    fixation_duration = 2.0  # These are variable, but we lack any data

    if "waitfornextrun" not in df['trial_type'].unique():
        last_event = df.loc[df['onset'].idxmax()]
        end_of_run = last_event['onset'] + last_event['duration']
        if run_length > 0.0:
            wait_duration = run_length - end_of_run - fixation_duration - 1.0
        else:
            wait_duration = 0.0
        final_event = pd.DataFrame([
            {
                "onset": end_of_run + 1.0,
                "duration": fixation_duration,
                "trial_type": "fixation",
                "response_time": "n/a",
                "stimulus": "+",
                "response": "n/a",
            },
            {
                'onset': end_of_run + 1.0 + fixation_duration,
                'duration': wait_duration,
                'trial_type': "waitfornextrun",
                'response_time': "n/a",
                'stimulus': "The next run of the task will begin in a moment.",
                'response': "n/a",
            },
        ])
        df = pd.concat([df, final_event, ], axis=0, ignore_index=True)
    return df


def split_into_runs(events_data, task, run_length=0.0, verbose=False):
    """ One ePrime file covers multiple fMRI runs; split them out. """

    bids_columns = [
        'onset', 'duration', 'trial_type',
        'response_time', 'stimulus', 'response',
    ]

    # Split into runs
    runs = []
    try:
        run_nums = sorted(events_data['run'].unique())
        if verbose:
            print(f"Found {len(run_nums)} blocks/runs in timing file")
        for run_num in run_nums:
            one_run_df = events_data[
                events_data['run'] == run_num
            ][
                bids_columns
            ].sort_values(by='onset', key=lambda x: x.astype(float))
            if task == "mem":
                one_run_df = add_end_of_run_wait_events(one_run_df, run_length)
            if verbose:
                run_start = one_run_df['onset'].astype(float).min()
                run_end = one_run_df['onset'].astype(float).max()
                last_duration = float(
                    one_run_df[
                        one_run_df['onset'] == run_end
                    ]['duration'].values[0]
                )
                timespan = run_end + last_duration - run_start
                print(f"  Run {run_num} occupies {timespan:0.1f} seconds "
                      f"with {len(one_run_df)} events; "
                      f"earliest event at {run_start:0.2f}s.")
            runs.append({
                "run": run_num,
                "dataframe": prettify_floats_to_strings(one_run_df),
            })
    except KeyError as key_error:
        print(f"  ERROR: {key_error}")

    return runs


def extract_txt_timing(filename, supp_table=None, shift=0.0, run_length=0.0,
                       verbose=False):
    """ Given a known OK ePrime txt output file, parse it

        This is the top-of-hierarchy file for each text file

        :param pathlib.Path filename:
            The file to parse
        :param pd.Dataframe supp_table:
            An alternative table with memory cue onsets
        :param float shift:
            How many seconds to subtract from each timestamp in filename
        :param float run_length:
            How many seconds in a complete run
        :param bool verbose:
            If this is set to True, print information about processing

        :returns:
            A list of dicts, each containing a DataFrame, one for each run
    """

    task = "none"
    if (
            ("automem" in str(filename).lower()) or
            ("memory" in str(filename).lower())
    ):
        task = "mem"
    elif (
            ("picture" in str(filename).lower())
            or ("image" in str(filename).lower())
    ):
        task = "image"

    frames = parse_file(filename, verbose=verbose)
    event_df, md = frames_to_dataframe(task, frames, verbose=verbose)
    # In the case where we are determining synch times with an extra table,
    # overwrite the synch times before finalizing the dataframe.
    if supp_table is not None:
        for k, v in md.items():
            if k != "task":
                blocks = supp_table[supp_table['block'] == k[-1]]
                if np.isfinite(blocks['block_onset'].min()):
                    md[k] = blocks['block_onset'].min() / 1000.0
    event_df = finalize_dataframe(event_df, md, verbose=verbose)
    event_df['onset'] = event_df['onset'] - shift

    return split_into_runs(event_df, task, run_length, verbose=verbose)


def extract_xl_timing(xl_file, shift=0.0, run_length=0.0, verbose=False):
    """ Extract events from a Conte task Excel sheet

        :param pathlib.Path xl_file:
            The file to parse
        :param float shift:
            How many seconds to subtract from each timestamp in filename
        :param float run_length:
            How many seconds in a complete run
        :param bool verbose:
            If this is set to True, print information about processing

        :returns:
            A list of dicts, each containing a DataFrame, one for each run
    """

    # Load Excel data
    sheet = pd.read_excel(
        xl_file, sheet_name=0, engine="openpyxl", skiprows=1,
    )
    if verbose:
        print(f"read {len(sheet)} rows and {len(sheet.columns)} columns "
              f"from {xl_file}.")

    if "picture" in xl_file.name:
        row_as_event = row_as_pic_event
        row_as_nonevent = row_as_pic_arrow_event
        task = "image"
    elif "memory" in xl_file.name:
        row_as_event = row_as_mem_event
        row_as_nonevent = row_as_mem_primary_event
        task = "mem"
    else:
        raise ValueError(f"{xl_file.name} is neither picture nor memory.")

    # Retrieve timestamps to shift each block's start
    block_onset_map = timestamp_map(sheet)

    # Extract experiment events and scanner events
    events = sheet.apply(row_as_event, axis=1)
    events = events.explode().dropna().drop_duplicates()
    nonevents = sheet.apply(row_as_nonevent, axis=1)
    nonevents = nonevents.explode().dropna().drop_duplicates()
    combined_events = list(pd.concat([events, nonevents, ]))
    event_df = pd.DataFrame(combined_events).sort_values('onset')
    event_df = event_df.reset_index(drop=True)

    # Correct for mismatched blocks and extra blocks in ePrime files.
    # Discontiguous blocks must be contiguous run numbers.
    block_id_map = {}
    for i, block_id in enumerate(sorted(event_df['block'].unique())):
        block_id_map[block_id] = i + 1
    event_df['run'] = event_df['block'].map(block_id_map)

    # Adjust event timing relative to block start points
    event_df['shift'] = event_df['run'].map(block_onset_map)
    event_df['shift'] = event_df['shift'] - shift
    event_df['onset'] = event_df['onset'] - event_df['shift']

    return split_into_runs(event_df, task, run_length, verbose=verbose)


def find_extra_bpd_mem_table(orig_events_file):
    """ Find a supplemental file to fill in details about timing.

        In BPD, the ePrime txt file only has scanner synchronization data for
        the first run. The rest are blank. But there's another table for most
        subjects that has accurate onsets for the memory cue. By using
        information from both files, we can create an accurate events.tsv file.
    """

    fixed_file, spaced_file, data, df = None, None, None, None
    sep = ","
    candidates = list(
        orig_events_file.parent.glob("BPD_MEMORY_fmri_*.txt")
    )
    candidates += list(
        (orig_events_file.parent / "mem").glob("BPD_MEMORY_fmri_*.txt")
    )
    for f in candidates:
        print(f"  supplementing with {f.name}")
        if "fixed" in f.name:
            fixed_file = f
            sep = ";"
            data = open(fixed_file, "r", errors="replace", )
            break
        else:
            spaced_file = f
            sep = " "
            data = open(spaced_file, "r", errors="replace", )

    if data:
        df = pd.read_csv(data, sep=sep, header=None)
        df = df.rename(
            columns={
                2: "block",
                3: "block_onset",
                4: "cue",
                5: "mem_onset",
            }
        )

    return df


def main(args):
    """ Entry point """

    if args.verbose:
        print("Extracting data from {}, shifting by {:0.2f} seconds.".format(
            args.events_file, args.shift,
        ))

    file = Path(args.events_file)
    out_path = Path(args.output_dir)
    if ("picture" in file.name.lower()) or ("reapp" in file.name.lower()):
        # "picture" or "picture_task"
        task = "image"
    elif "mem" in file.name.lower():
        # "memory" or "automem_task"
        task = "mem"
    else:
        print("Only two tasks are supported, 'picture' and 'memory'.")
        raise ValueError(f"I cannot figure out what task '{file.name}' holds.")

    # Do the extraction
    out_path.mkdir(exist_ok=True)
    if file.name.endswith(".xlsx") or file.name.endswith(".xls"):
        print("WARNING:")
        print("Using the Excel version is less reliable than the txt version.")
        print("This code is forced to make some assumptions about durations ")
        print("that are not necessarily true.")
        runs = extract_xl_timing(
            file, shift=args.shift, run_length=args.run_duration,
            verbose=args.verbose
        )
    else:
        runs = extract_txt_timing(
            file, supp_table=find_extra_bpd_mem_table(file),
            shift=args.shift, run_length=args.run_duration,
            verbose=args.verbose
        )

    file_root = f"sub-{args.subject}_ses-{args.session}_task-{task}"
    for run in runs:
        tsv_file = f"{file_root}_run-{run['run']:02d}_events.tsv"
        run['dataframe'].to_csv(
            out_path / tsv_file, sep='\t', header=True, index=False,
            float_format='%0.3f',
        )
    json_file = f"{file_root}_events.json"
    json.dump(
        make_sidecar(task), open(out_path / json_file, "w"),
        sort_keys=False, indent=4,
    )


if __name__ == "__main__":
    main(get_arguments())
