#!/usr/bin/env python3

""" extract_ims_events.py

    The IMS project has a 'study' task followed by a 'test' task.
    Ideally, both events files will have information from the other task,
    so they need to be created in parallel. If either xlsx file is opened
    with this program, it will look for the other task's xlsx file and
    build both events files. So this does not need to be run on each file;
    either will do.

    Use `extract_ims_events.py --help` for usage details
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number
from warnings import warn


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
        help="the xlsx events file, could be from study or test.",
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
        "--rawdata", default="/mnt/rawdata/ims",
        help="the path to BIDS-valid rawdata folder for looking up sessions",
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
        print(f"subject '{parsed_args.subject}', "
              f"session '{parsed_args.session}'")

    # Did we get a study or events file, and where's the other one?
    study_file, test_file = find_both_excel_files(parsed_args.events_file)
    setattr(parsed_args, "study_events_file", study_file)
    setattr(parsed_args, "test_events_file", test_file)

    return parsed_args


def find_both_excel_files(one_excel_file):
    """ Given the file specified, can we find the other one? """

    one_excel_file = Path(one_excel_file)
    study_file, test_file = (None, None)

    if "study" in one_excel_file.name:
        study_file = one_excel_file
        found_task = "study"
        task_to_find = "test"
    elif "test" in one_excel_file.name:
        test_file = one_excel_file
        found_task = "test"
        task_to_find = "study"
    else:
        raise ValueError(
            "Neither 'study' nor 'test' is in the event file name. "
            "I need one or the other to proceed."
        )

    # In the future, the excel files ought to be in the same place, so
    # the first attempt will be to look in the same folder for both files.
    # But if that's not the case, make a second attempt with new assumptions.
    # Typically, the excel file is in .../regressors/task/excel/file.xlsx
    # So we need to glob from the parent's parent's parent (regressors)
    # and only files starting with a letter or number to avoid the .,~ files
    related_excel_files = list(
        # This will be the case after moving files to sourcedata
        one_excel_file.parent.parent.parent.glob(
            f"*/*/[A-Za-z0-9]*{task_to_find}*.xlsx"
        )
    )
    if len(related_excel_files) == 0:
        # This will always be the case in Christina's home directory
        related_excel_files = list(
            one_excel_file.parent.parent.parent.glob(
                f"*/*/[A-Za-z0-9]*{task_to_find}*.xlsx"
            )
        )

    if len(related_excel_files) == 0:
        warn(f"No {task_to_find} file to match {found_task}")
    elif len(related_excel_files) > 1:
        warn(f"Too many {task_to_find} files; I need only one.")
        for i, file in enumerate(related_excel_files):
            print(f" - {str(file.name)}")
    else:
        if task_to_find == "study":
            study_file = related_excel_files[0]
        elif task_to_find == "test":
            test_file = related_excel_files[0]
        else:
            raise ValueError(
                f"This should never happen. 'task_to_find' is '{task_to_find}'"
            )

    return study_file, test_file


def find_session(subject_id, rawdata="/mnt/rawdata"):
    """ From a subject id, find and return a session.

        :param str subject_id:
            A subject_id, without the sub- portion
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
        glob_str = "func/sub-*_ses-*_task-*_bold.nii.gz"
        for possibility in possibilities:
            if len(list(possibility.glob(glob_str))) > 0:
                return possibility.name[possibility.name.rfind("-") + 1:]


def row_as_event(row):
    """ Interpret dataframe row as event, and return dict with relevant items.

        :param row:
            A pandas DataFrame row, read from an Excel file,
            to be treated as an event in a memory trial.
        :returns list:
            A list of dicts, each dict containing key-value pairs intended
            to become one row in a BIDS-valid events.tsv file
    """

    assert "Stim" in row
    assert "StimType" in row
    assert "TypeLabel" in row
    assert "iscorr" in row
    assert "image.started" in row
    assert "image.stopped" in row
    assert "jitter.started" in row
    assert "jitter.stopped" in row
    assert "key_resp_img.keys" in row
    assert "key_resp_img.corr" in row
    assert "key_resp_img.rt" in row
    assert "key_resp_img.started" in row
    assert "key_resp_img.stopped" in row
    assert "image.onset" in row
    assert "image.dur" in row

    try:
        response = str(int(row["key_resp_img.keys"]))
    except ValueError:
        response = "n/a"

    try:
        correct = str(int(row["key_resp_img.corr"]))
    except ValueError:
        correct = "n/a"

    # Float values will be truncated to strings at the very end,
    # but they should remain floats here, so they can be manipulated
    return pd.Series({
        "onset": float(row["image.onset"]),
        "duration": float(row["image.dur"]),
        "trial_type": row["TypeLabel"],
        "stimulus": str(row["Stim"]).replace("stim/", ""),
        "response": response,
        "response_time": float(row["key_resp_img.rt"]),
        "correct": correct,
    })


def make_sidecar(task):
    """ For a given task, create a json file describing tsv fields

        :param str task:
            The task descriptor for a run: 'study' or 'test' in IMS
        :returns dict:
            A nested dict appropriate for direct json serialization
    """

    # The events.tsv files created in this script contain four columns
    # expected by BIDS. They also contain additional columns that should
    # be described in this accompanying sidecar file.
    if task == "study":
        levels = {
            "neut_stim": "A 2.5ish second period while a neutral picture is "
                         "presented to the participant",
            "emot_stim": "A 2.5ish second period while an emotional picture is "
                         "presented to the participant",
        }
        stimulus_description = (
            "The stimulus presented to the participant. "
            "Images beginning with '1' are coded by IAPS to be emotional "
            "and are displayed in 'emot' trial_types. "
            "Images beginning with '2' are coded as neutral and are in "
            "'neut' trial_types. "
            "All study images end in 'a', indicating they are "
            "presented in the 'study' phase."
        )
        response_description = (
            "The participant's response to the stimulus. "
            "The '3' button indicates the participant perceived "
            "the stimulus to be emotional. "
            "The '4' button indicates the participant perceived "
            "the stimulus to be neutral."
        )
        correct_description = (
            "If the participant and IAPS agree on the emotionality of "
            "the image, 'correct' is '1', otherwise it is '0'."
        )
    elif task == "test":
        levels = {
            "neut_new": "A 2.5ish second period while a neutral picture "
                        "that has not been seen before is presented to the "
                        "participant.",
            "neut_old": "A 2.5ish second period while a neutral picture "
                        "that is identical to one seen in the study phase "
                        "is presented to the participant.",
            "neut_hsim": "A 2.5ish second period while a neutral picture "
                         "that is highly similar, but not identical, to one "
                         "presented in the study phase is presented to the "
                         "participant.",
            "neut_lsim": "A 2.5ish second period while a neutral picture "
                         "with low similarity to one presented in "
                         "the study phase is presented to the participant.",
            "emot_new": "A 2.5ish second period while a neutral picture "
                        "that has not been seen before is presented to the "
                        "participant.",
            "emot_old": "A 2.5ish second period while a neutral picture "
                        "that is identical to one seen in the study phase "
                        "is presented to the participant.",
            "emot_hsim": "A 2.5ish second period while a neutral picture "
                         "that is highly similar, but not identical, to one "
                         "presented in the study phase is presented to the "
                         "participant.",
            "emot_lsim": "A 2.5ish second period while a neutral picture "
                         "with low similarity to one presented in "
                         "the study phase is presented to the participant.",
        }
        stimulus_description = (
            "The stimulus presented to the participant. "
            "Images beginning with '1' or '4' are coded by IAPS to be "
            "emotional and are displayed in 'emot' trial_types. "
            "Images beginning with '2' or '5' are coded as neutral and are in "
            "'neut' trial_types. "
            "Images ending with 'a' are original images, meaning that if they "
            "start with '1' or '2', they were seen in the study phase; if "
            "they start with '4' or '5', they are new in this phase. All 'a' "
            "images should correspond with 'new' or 'old' trial_types. "
            "Images ending with 'b' are low-similarity lures of an 'a' image "
            "with the same number in the study phase. "
            "Images ending with 'c' are high-similarity lures of an 'a' image "
            "with the same number in the study phase. "
        )
        response_description = (
            "The participant's response to the stimulus. "
            "The '3' button indicates the participant remembers this exact "
            "image from the study phase. "
            "The '4' button indicates the participant does not remember this "
            "exact image from the study phase. "
        )
        correct_description = (
            "If the participant correctly identifies an image from the "
            "study phase being re-shown in the test phase, by responding "
            "with the '3' button, 'correct' is '1'. If the participant "
            "correctly identifies that an image shown in this test phase "
            "was not shown in the study phase ('new', 'hsim', 'lsim') by "
            "responding with the '4' button, 'correct' is also '1'. "
            "Responding '3' to a non-'old' image or '4' to an 'old' image "
            "is incorrect, and 'correct' is '0'."
        )
    else:
        levels = {}
        stimulus_description = "n/a"
        response_description = "n/a"
        correct_description = "n/a"

    data = {
        "trial_type": {
            "LongName": "Event category",
            "Description": "Indicator of type of action that is expected",
            "Levels": levels
        },
        "stimulus": {
            "Description": stimulus_description,
        },
        "response": {
            "Description": response_description,
        },
        "correct": {
            "Description": correct_description,
        },
    }
    return data


def float_to_str(num):
    """ Convert a float to a string, but use BIDS 'n/a' when necessary. """

    if isinstance(num, Number):
        if np.isnan(num):
            return "n/a"
        else:
            return f"{num:0.3f}"
    elif isinstance(num, str):
        return num
    else:
        return "n/a"


def prettify_floats_to_strings(df):
    """ Ensure we write out pretty text to the tsv """

    # converts numerics to strings, so best to do this after all calculations
    columns = df.columns
    df = df.rename(
        columns={
            'onset': '_onset',
            'duration': '_duration',
            'response_time': '_response_time',
        }
    )
    df['onset'] = df['_onset'].apply(float_to_str)
    df['duration'] = df['_duration'].apply(float_to_str)
    df['response_time'] = df['_response_time'].apply(float_to_str)
    return df[columns]


def extract_timing(xl_file, shift=0.0, verbose=False):
    """ Extract events from an IMS Excel sheet

        :param pathlib.Path xl_file:
            The file to parse
        :param float shift:
            How many seconds to subtract from each timestamp in filename
        :param bool verbose:
            If this is set to True, print information about processing

        :returns:
            A list of dicts, each containing a proper DataFrame for each run
    """

    # Load Excel data
    sheet = pd.read_excel(
        xl_file, sheet_name=0, engine="openpyxl", skiprows=0,
    )
    if verbose:
        print(f"read {len(sheet)} rows and {len(sheet.columns)} columns "
              f"from {xl_file}.")

    # Extract experiment events and scanner events
    events = sheet.apply(row_as_event, axis=1)
    events = events.dropna(axis=0, thresh=4)
    events = events.sort_values('onset').reset_index(drop=True)

    # Adjust event timing relative to cropping of initial settling frames
    events['onset'] = events['onset'] - shift

    return events


def get_one_matching_row(iaps_num, dataframe):
    iaps_filter = dataframe['stimulus'].str.contains(iaps_num)
    df = dataframe[iaps_filter][['trial_type', 'stimulus', 'response', ]]
    # There should be only one event in test matching the iaps number
    if len(df) == 1:
        return df.iloc[0]
    warn(f"  IAPS #{iaps_num} has {len(df)} events in test.")
    return None


def finalize_study_runs(study_runs, test_runs):
    """ Add variables and finalize the study events dataframe. """

    # The 'correct' column actually indicates whether the participant
    # perceived the image as emotional or not.
    # study_runs = study_runs.rename(columns={"correct": "perceived_emot"})

    # Modeling encoding, it might be useful to know if retrieval events
    # using each image were successful.
    def remembered_in_test(stim):
        s = get_one_matching_row(stim[:5], test_runs)
        if s is not None:
            if str(s['trial_type']).endswith("old"):
                if "a" in str(s['stimulus']):
                    if str(s['response']) == '3':
                        return '1'
                    else:
                        return '0'
                else:
                    warn("trial_type old, but stimulus not 'a'")
            else:
                if "a" in str(s['stimulus']):
                    warn("stimulus 'a', but trial_type not 'old'")
        return "n/a"

    def discriminated_hsim_in_test(stim):
        s = get_one_matching_row(stim[:5], test_runs)
        if s is not None:
            if str(s['trial_type']).endswith("hsim"):
                if "c" in str(s['stimulus']):
                    if str(s['response']) == '4':
                        return '1'
                    else:
                        return '0'
                else:
                    warn("trial_type 'hsim', but stimulus not 'c'")
            else:
                if "c" in str(s['stimulus']):
                    warn("stimulus 'c', but trial_type not 'hsim'")
        return "n/a"

    def discriminated_lsim_in_test(stim):
        s = get_one_matching_row(stim[:5], test_runs)
        if s is not None:
            if str(s['trial_type']).endswith("lsim"):
                if "b" in str(s['stimulus']):
                    if str(s['response']) == '4':
                        return '1'
                    else:
                        return '0'
                else:
                    warn("trial_type 'hsim', but stimulus not 'b'")
            else:
                if "b" in str(s['stimulus']):
                    warn("stimulus 'b', but trial_type not 'lsim'")
        return "n/a"

    study_runs['test_corr_remembered'] = study_runs['stimulus'].apply(
        remembered_in_test
    )
    study_runs['test_corr_hsim_discriminated'] = study_runs['stimulus'].apply(
        discriminated_hsim_in_test
    )
    study_runs['test_corr_lsim_discriminated'] = study_runs['stimulus'].apply(
        discriminated_lsim_in_test
    )

    return study_runs


def finalize_test_runs(test_runs, study_runs):
    """ Add variables and finalize the study events dataframe. """

    # Modeling decoding, it might be useful to know if encoding events
    # were perceived by the participant as emotional, regardless of coding.
    def perceived_emot(stim):
        if str(stim)[0] in ['4', '5', ]:
            # It's a new image, don't bother looking it up.
            return "n/a"
        s = get_one_matching_row(stim[:5], study_runs)
        if s is not None:
            if str(s['response']) == '3':
                return '1'
            else:
                return '0'
        return "n/a"

    test_runs['perceived_emot'] = test_runs['stimulus'].apply(
        perceived_emot
    )

    return test_runs


def main(args):
    """ Entry point """

    if args.verbose:
        print(f"Extracting study data from \n '{args.study_events_file}'")
        print(f"Extracting test data from \n '{args.test_events_file}'")
        print(f"Shifting both by {args.shift:0.2f} seconds.")

    # Do the first-pass extraction
    study_runs = extract_timing(
        args.study_events_file, shift=args.shift, verbose=args.verbose
    )
    test_runs = extract_timing(
        args.test_events_file, shift=args.shift, verbose=args.verbose
    )

    # Do the second-pass to fill in extra variables
    # Study runs must be done first so test runs can refer to changed col names
    final_study_runs = finalize_study_runs(study_runs, test_runs)
    final_test_runs = finalize_test_runs(test_runs, final_study_runs)

    # Write the output events files
    outpath = Path(args.output_dir)
    outpath.mkdir(exist_ok=True)
    file_root = f"sub-{args.subject}_ses-{args.session}"
    for task, df in [('study', final_study_runs), ('test', final_test_runs), ]:
        tsv_file = f"{file_root}_task-{task}_events.tsv"
        prettify_floats_to_strings(df).to_csv(
            outpath / tsv_file, sep='\t', header=True, index=False,
            float_format='%0.3f',
        )
        json_file = f"{file_root}_task-{task}_events.json"
        json.dump(
            make_sidecar(task), open(outpath / json_file, "w"),
            sort_keys=False, indent=4,
        )


if __name__ == "__main__":
    main(get_arguments())