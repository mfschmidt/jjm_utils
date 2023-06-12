#!/usr/bin/env python3

"""
conte_one_eprime_to_bids_events.py (formerly eprimetext_to_bidsevents.py)

From a primary ePrime text file, or secondary csv,
generate multiple BIDS events.tsv files
"""

import re
from pathlib import Path
from numbers import Number
import pandas as pd
import argparse


base = Path("/home/mike/Desktop/old_conte_timing_from_jjm2")
bids_columns = [
    'onset', 'duration', 'trial_type', 'stimulus', 'response', 'response_time',
]


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Generate BIDS events files from ePrime text output. "
                    "This also handles secondary csv files for image tasks.",
    )
    parser.add_argument(
        "eprime_text",
        help="A required ePrime text file (or Reapp csv).",
    )
    parser.add_argument(
        "--rawdata", default="/data/BI/human/rawdata/old_conte",
        help="Specify where the rawdata files are, "
             "for determining subject's session number and "
             "for writing out completed events.tsv files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="If set, existing BIDS timing files will be overwritten.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Set to trigger verbose output.",
    )

    args = parser.parse_args()

    return args


def ms_to_sec(ms, precision=3, to_str=False):
    """ Convert milliseconds to seconds """
    if to_str:
        return f"{(float(ms) / 1000.0):0.{precision}f}"
    else:
        return float(ms) / 1000.0


def parse_file(file):
    """ Make sense of the file's contents and return relevant data.
    """

    pattern_wrapper = re.compile(r"[*][*][*]\s+(\S+)\s+(\S+)\s+[*][*][*]")
    key_value = re.compile(r"(\S+.*):\s+(.*)$")
    try:
        with open(file, "r") as f:
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
                    frame[match.group(1)] = match.group(2).strip()

    except UnicodeDecodeError:
        print(f"  skipping non-UTF-8 file '{file.name}' ({i}, {line})")

    return frames


def frames_to_dataframe(task, frames, verbose=False):
    """ Convert list of dicts to dataframe, keeping only what we want.
    """

    rows = []
    metadata = {'task': task, }
    for i, frame in enumerate(frames):
        if verbose:
            print(f"Reading frame {i}, from line {frame['line'] + 1}")
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
                        'response': "n/a", 'response_time': 0.0,
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
                    rows.append({
                        'trial_type': "question",
                        'onset': ms_to_sec(frame.get('Qtext.OnsetTime', '0')),
                        'duration': ms_to_sec(frame.get('Qtext.RT', '0')),
                        'stimulus': frame.get('Question', "n/a"),
                        'response': frame.get('Qtext.RESP', "n/a"),
                        'response_time': ms_to_sec(frame.get('Qtext.RT', '0')),
                    })
                elif frame['Running'].startswith("Arrow"):
                    rows.append({
                        'trial_type': "arrow",
                        'onset': ms_to_sec(frame['Arrow.OnsetTime']),
                        'duration': ms_to_sec(frame['Arrow.RT']),
                        'stimulus': frame['Arrow'],
                        'response': frame['Arrow.RESP'],
                        'response_time': ms_to_sec(frame['Arrow.RT']),
                    })
                elif frame['Running'].startswith("WordList"):
                    rows.append({
                        'trial_type': "memory",
                        'onset': ms_to_sec(
                            frame.get('MemCue.OnsetTime', 0)
                        ),
                        'duration': 0.0,
                        'stimulus': frame['MemCue'],
                        'response': "n/a",
                        'response_time': "n/a",
                    })
                    rows.append({
                        'trial_type': "instruct",
                        'onset': ms_to_sec(
                            frame.get('Instruct.OnsetTime', '0')
                        ),
                        'duration': 0.0,
                        'stimulus': frame['Word'].lower(),
                        'response': "n/a",
                        'response_time': "n/a",
                    })
                    # Subtract the delay from the onset time for this Memory Cue
                    onset = ms_to_sec(frame['MemCue.OnsetTime'])
                    delay = ms_to_sec(frame['MemCue.OnsetDelay'])
                    base_time = onset - delay - 8.0
                    if (
                            (frame['Procedure'] not in metadata) or
                            (base_time < metadata[frame['Procedure']])
                    ):
                        metadata[frame['Procedure']] = base_time
                elif frame['Running'] == "RecallTaskProc":
                    rows.append({
                        'trial_type': "memory",
                        'onset': ms_to_sec(frame['MemCue.OnsetTime']),
                        'duration': 0.0,
                        'stimulus': frame['Memory'],
                        'response': "n/a",
                        'response_time': "n/a",
                    })
                    # Subtract the delay from the onset time for this Memory Cue
                    onset = ms_to_sec(frame['MemCue.OnsetTime'])
                    delay = ms_to_sec(frame['MemCue.OnsetDelay'])
                    base_time = onset - delay - 8.0
                    if (
                            (frame['Procedure'] not in metadata) or
                            (base_time < metadata[frame['Procedure']])
                    ):
                        metadata[frame['Procedure']] = base_time
                elif frame['Running'] in ["SelfPracProc", "TimePracProc", ]:
                    rows.append({
                        'trial_type': "instruct",
                        'onset': 0.0,
                        'duration': 0.0,
                        'stimulus': frame['Instruction'].lower(),
                        'response': "n/a",
                        'response_time': "n/a",
                    })
                elif frame['Running'].startswith("RunList"):
                    # One of these has synch info, which is redundant,
                    # the rest are worthless.
                    pass
                    # on_key = f"SynchWithScanner{frame['RunName']}.OnsetTime"
                    # if on_key in frame:
                    #     metadata[on_key] = ms_to_sec(frame[on_key])
                    # off_key = f"SynchWithScanner{frame['RunName']}.OffsetTime"
                    # if off_key in frame:
                    #     metadata[off_key] = ms_to_sec(frame[off_key])
                    #     metadata['offset_time'] = metadata[off_key]
                    #     metadata[frame['Procedure']] = ms_to_sec(
                    #         frame[off_key]
                    #     )
                else:
                    print(f"  Frame {i} (line {frame['line']}) missed. "
                          f"{task}: Running=='{frame['Running']}'")

        else:
            print(f"  Frame {i}, line {frame['line']}. "
                  f"'Running' key not in frame.")

    return metadata, pd.DataFrame(rows)


def finalize_dataframe(df, offsets):
    """ Use the offsets to shift dataframe's 'onset' values & fill in durations.
    """

    # Rows must be in chronological order to calculate durations
    # And indices from 0-length allow for correct indexing with loc
    df = df.sort_values('onset').reset_index(drop=True)

    # Fill in durations of 0 with calculated durations.
    for i in range(len(df)):
        if (i < len(df) - 1) and (df.loc[i, 'duration'] == 0.0):
            df.loc[i, 'duration'] = df.loc[i + 1, 'onset'] - df.loc[i, 'onset']

    # Format offsets nicely for splitting timing into separate fMRI BOLD runs
    sync_times = []
    if offsets:
        sync_df = pd.DataFrame(data=None)
        for k, v in offsets.items():
            if k != 'task':
                sync_times.append({'t': v})
        try:
            # Extract runs from metadata
            sync_df = pd.DataFrame(sync_times)
            sync_df = sync_df.sort_values('t').reset_index(drop=True)
            sync_df['run'] = [i + 1 for i in range(len(sync_df))]

            # Center each 'onset' to its run's start time.
            df = df.rename(columns={'onset': 'original_onset'})
            df['onset'] = df['original_onset'].apply(
                lambda onset: onset - sync_df[sync_df['t'] < onset].max()[0]
            )
            df['run'] = df['original_onset'].apply(
                lambda onset: sync_df.loc[
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

    # Ensure we write out pretty text to the tsv
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

    return df


def translate_csv(filename):
    """ From Noam's csv, load into a BIDS-compatible table. """

    raw_df = pd.read_csv(filename, sep=',', skipinitialspace=True)

    cues = {
        'LookNeu': 'LOOK',
        'LookNeg': 'LOOK',
        'LookPos': 'LOOK',
        'ReappNeg': 'INCREASE POSITIVE',
        'ReappPos': 'INCREASE POSITIVE',
    }
    # The input csv is just 90 rows, each with an image. We need to transform
    # this into six tsv files, each with 15 images, each image with 6 events.

    # Individual row calculations
    runs = [_ for run in range(1, 7) for _ in [run, ] * 15]
    images = []
    for run_idx, (idx, row) in enumerate(raw_df.iterrows()):
        # Standardize some variable data
        if row['ratingneg_onset'] < row['ratingpos_onset']:
            first_rating_onset = row['ratingneg_onset']
        else:
            first_rating_onset = row['ratingpos_onset']
        try:
            pos_rating = str(int(row['ratingpos']))
        except ValueError:
            pos_rating = "n/a"
        try:
            neg_rating = str(int(row['ratingneg']))
        except ValueError:
            neg_rating = "n/a"
        images.append({
            'onset': ms_to_sec(row['cue_onset'] - row['run_onset']),
            'duration': ms_to_sec(row['stim_onset'] - row['cue_onset']),
            'trial_type': row['trialtype'],
            'stimulus': cues[row['trialtype']],  # 'LOOK' or 'INCREASE POSITIVE'
            'response': "n/a",
            'response_time': "n/a",
            "run": runs[run_idx],
        })
        images.append({
            'onset': ms_to_sec(row['stim_onset'] - row['run_onset']),
            'duration': ms_to_sec(row['isi_onset'] - row['stim_onset']),
            'trial_type': row['trialtype'],
            'stimulus': row['stim'],  # image name ('###.bmp')
            'response': "n/a",
            'response_time': "n/a",
            "run": runs[run_idx],
        })
        images.append({
            'onset': ms_to_sec(row['isi_onset'] - row['run_onset']),
            'duration': ms_to_sec(first_rating_onset - row['isi_onset']),
            'trial_type': 'isi',
            'stimulus': "n/a",
            'response': "n/a",
            'response_time': "n/a",
            "run": runs[run_idx],
        })
        if row['ratingneg_onset'] < row['ratingpos_onset']:
            images.append({
                'onset': ms_to_sec(row['ratingneg_onset'] - row['run_onset']),
                'duration': ms_to_sec(
                    row['ratingpos_onset'] - row['ratingneg_onset']
                ),
                'trial_type': 'rating',
                'stimulus': 'rn',
                'response': neg_rating,
                'response_time': "n/a",
                "run": runs[run_idx],
            })
            images.append({
                'onset': ms_to_sec(row['ratingpos_onset'] - row['run_onset']),
                'duration': ms_to_sec(
                    row['iti_onset'] - row['ratingpos_onset']
                ),
                'trial_type': 'rating',
                'stimulus': 'rp',
                'response': pos_rating,
                'response_time': "n/a",
                "run": runs[run_idx],
            })
        else:
            images.append({
                'onset': ms_to_sec(row['ratingpos_onset'] - row['run_onset']),
                'duration': ms_to_sec(
                    row['ratingneg_onset'] - row['ratingpos_onset']
                ),
                'trial_type': 'rating',
                'stimulus': 'rp',
                'response': pos_rating,
                'response_time': "n/a",
                "run": runs[run_idx],
            })
            images.append({
                'onset': ms_to_sec(row['ratingneg_onset'] - row['run_onset']),
                'duration': ms_to_sec(
                    row['iti_onset'] - row['ratingneg_onset']
                ),
                'trial_type': 'rating',
                'stimulus': 'rn',
                'response': neg_rating,
                'response_time': "n/a",
                "run": runs[run_idx],
            })
        images.append({
            'onset': ms_to_sec(row['iti_onset'] - row['run_onset']),
            'duration': ms_to_sec(row['iti']),
            'trial_type': "iti",
            'stimulus': "n/a",
            'response': "n/a",
            'response_time': "n/a",
            "run": runs[run_idx],
        })

    # Create a dataframe with each dict saved above as its own row.
    prepped_dataframe = pd.DataFrame(images).sort_values(['run', 'onset', ])

    return None, prepped_dataframe


def find_session(subject_id, raw_data_dir):
    """ From a subject id, find and return a session. """

    rawdata = Path(raw_data_dir) / f"sub-{subject_id}"
    possibilities = list(rawdata.glob("ses-*"))
    if len(possibilities) == 1:
        return possibilities[0].name[possibilities[0].name.rfind("-") + 1:]
    elif len(possibilities) == 0:
        return "na"
    else:
        print(f"  Subject {subject_id} has {len(possibilities)} sessions")
        for possibility in possibilities:
            possible_niftis = possibility.glob(
                "func/sub-*_ses-*_task-*_run-*_bold.nii.gz"
            )
            if len(list(possible_niftis)) > 0:
                return possibility.name[possibility.name.rfind("-") + 1:]


def interpret_filename(filename, raw_data_dir):
    """ From the filename, determine task, subject, etc.
    """

    pattern_1 = re.compile(r"Conte(.+)[-_](U*\d+)\s*[-_].*.txt")
    pattern_2 = re.compile(r"ConteREAPPlog_sub([0-9]+)\.csv")

    subject, session, task, original = None, None, None, None
    if "Training" in filename:
        print(f"{filename:<60}  skipping training file")
    else:
        print(f"{filename:<60}: ...working...")

        match_1 = pattern_1.match(filename)
        if match_1:
            original = True
            task = match_1.group(1)
            if "mem" in task.lower():
                task = "mem"
            elif "reapp" in task.lower():
                task = "image"
            subject = match_1.group(2)
            if not subject.startswith("U"):
                subject = "U" + subject

        match_2 = pattern_2.match(filename)
        if match_2:
            original = False
            task = "image"
            subject = f"U{match_2.group(1)}"

        if match_1 or match_2:
            session = find_session(subject, raw_data_dir)

    if (subject is None) or (session is None) or (task is None):
        print(f"Cannot interpret '{filename}':")
        print(f"  subject is '{subject}'")
        print(f"  session is '{session}'")
        print(f"  task is '{task}'")

    return subject, session, task, original


def process_file(filename, raw_data_dir, force=False, verbose=False):
    """ Given a known OK ePrime txt output file, parse it

        This is the top-of-hierarchy file for each file

        :param pathlib.Path filename:
            The file to parse
        :param pathlib.Path raw_data_dir:
            The path to rawdata containing BIDS-compliant subjects
        :param bool force:
            Do not overwrite existing files unless this is set to True
        :param bool verbose:
            If this is set to True, print information about processing
    """

    tsv_template = "sub-{}_ses-{}_task-{}_run-{:02d}_events.tsv"
    subject, session, task, original = interpret_filename(
        filename.name, raw_data_dir
    )
    if subject and session and task:
        if original:
            frames = parse_file(filename)
            print(f"  {subject:<12}  {task:<12}  {len(frames):,} items")
            md, df = frames_to_dataframe(task, frames)
        else:
            md, df = translate_csv(filename)
        df = finalize_dataframe(df, md)

        # Pre-build and save some filters on the dataframe
        image_filter = df['stimulus'].str.endswith('.bmp')
        mem_filter = df['trial_type'] == 'memory'
        pos_trials = df['trial_type'] == 'ReappPos'
        neg_trials = df['trial_type'] == 'ReappNeg'

        func_dir = raw_data_dir / f"sub-{subject}" / f"ses-{session}" / "func"
        try:
            func_dir.mkdir(parents=True, exist_ok=True)
            runs = sorted(df['run'].unique())
            if verbose:
                print(f"  found {len(runs)} blocks/runs in timing file")
            for run in runs:
                run_filter = df['run'] == run
                events_name = tsv_template.format(subject, session, task, run, )
                events_file = func_dir / events_name
                if verbose:
                    if task == "mem":
                        # Build a boolean filter of memory trials.
                        num_mems = len(df[run_filter & mem_filter])
                        print(f"      events.tsv contains {num_mems} memories")
                    if task == "image":
                        # Build a boolean filter of picture trials.
                        num_images = len(df[run_filter & image_filter])
                        num_reappraise_pos = len(
                            df[run_filter & image_filter & pos_trials]
                        )
                        num_reappraise_neg = len(
                            df[run_filter & image_filter & neg_trials]
                        )
                        print(f"      events.tsv contains {num_images} images "
                              f"({num_reappraise_pos} ReappPos, "
                              f"{num_reappraise_neg} ReappNeg)")
                if events_file.exists():
                    if force:
                        print(f"  {events_file.name} already exists, "
                              "replacing it.")
                    else:
                        print(f"  {events_file.name} already exists, "
                              "not writing over it - check base directory.")
                if force or not events_file.exists():
                    df[run_filter].sort_values('_onset')[bids_columns].to_csv(
                        events_file, sep="\t", index=False
                    )
        except KeyError as key_error:
            print(f"  ERROR: {key_error}")
            print("  ERROR: Nothing to write to file")


def main(args):
    """ If a file is provided, process it.
        If a directory is provided, process every Conte*.txt file in it.
    """
    
    if Path(args.eprime_text).is_file():
        process_file(
            Path(args.eprime_text), Path(args.rawdata),
            force=args.force, verbose=args.verbose
        )

    elif Path(args.eprime_text).is_dir():
        potential_sources = list(
            Path(args.eprime_text).glob("Conte*.txt")
        ) + list(
            Path(args.eprime_text).glob("Conte*.csv")
        )
        for timing_file in potential_sources:
            process_file(
                timing_file, Path(args.rawdata),
                force=args.force, verbose=args.verbose
            )


if __name__ == "__main__":
    main(get_arguments())
