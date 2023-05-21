#!/usr/bin/env python3

# aggregate_scores.py

import argparse
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Extract all scores for each run and save to one file.",
    )
    parser.add_argument(
        "-i", "--input-dir", default=".",
        help="The path containing all subjects",
    )
    parser.add_argument(
        "-o", "--output-file", default="./scores.NEW.csv",
        help="The output file to write all results into",
    )
    parser.add_argument(
        "--fmriprep-path",
        default="/data/BI/human/derivatives/new_conte/fmriprep",
        help="The path to fMRIPrep data, containing subject directories"
    )
    parser.add_argument(
        "--steady-state-outliers", default=0, type=int,
        help="How many volumes to crop from the beginning of confounds"
    )
    parser.add_argument(
        "--motion-threshold", default=2.0, type=float,
        help="Threshold for motion (FD in mm) to consider volume a spike"
    )
    parser.add_argument(
        "--zero-point", default="instruct", type=str,
        help="Is the stimulus of interest 'instruct' or 'memory'?"
    )
    parser.add_argument(
        "--tr-dim", default=0.9, type=float,
        help="How many seconds in the 4th dimension from one volume to the next"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    # Report on fatal errors, and exit if we have a problem.
    need_to_bail = False
    if not Path(args.input_dir).exists():
        print(f"Input path '{args.input_dir}' does not exist.")
        need_to_bail = True
    if not Path(args.fmriprep_path).exists():
        print(f"fMRIPrep path '{args.fmriprep_path}' does not exist.")
        need_to_bail = True
    if not Path(args.output_file).parent.exists():
        print(f"{str(Path(args.output_file).parent)} doesn't exist.")
        need_to_bail = True
    if need_to_bail:
        sys.exit(1)

    # Save paths as Path objects rather than strings.
    setattr(args, "input_dir", Path(args.input_dir).absolute())
    setattr(args, "output_file", Path(args.output_file).absolute())
    setattr(args, "fmriprep_path", Path(args.fmriprep_path).absolute())

    return args


def get_val_from_key(filename, key):
    """ Extract and return a value from "..._key-value_..." string.
    """

    match = re.search(rf"{key}-([0-9A-Za-z]+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_fd(subject, task, run, args):
    """ From the score filepath, find the confounds file and extract FD.
    """

    confound_files = list((args.fmriprep_path / f"sub-{subject}").glob(
        f"**/sub-{subject}*task-{task}_run-{run}_desc-confounds_timeseries.tsv"
    ))
    if len(confound_files) > 0:
        df = pd.read_csv(confound_files[0], index_col=None, sep='\t')
        fd = df['framewise_displacement'][args.steady_state_outliers:]
        return fd.max(), len(fd[fd > args.motion_threshold])
    else:
        print(f"Confound file for {subject}/{task}/{run} could not be found.")
        return None, None


def get_run_events(run_dir, args):
    """ From the score filepath, find the events file return its data.
    """

    time_shift = args.tr_dim * args.steady_state_outliers
    # There should be one and only one events file per run, so assume it's so
    for ev_file in run_dir.glob("regressors/sub-*_task-*_run-*_events.tsv"):
        if ev_file.exists():
            df = pd.read_csv(ev_file, index_col=None, header=0, sep="\t")
            df = df.sort_values('onset')
            df['onset'] = df['onset'] - time_shift
            return df
    return None


def get_valence(subject, task, run, args, instruct, period):
    """ From the score filepath, find the events file and extract valence.
    """

    how_badly, how_vivid = None, None

    run_dir = args.input_dir / f"sub-{subject}" / f"task-{task}" / f"run-{run}"
    df = get_run_events(run_dir, args)
    if df is not None:
        onsets = df[
            (df['trial_type'] == 'instruct') & (df['stimulus'] == instruct)
        ]['onset']
        block_start = onsets.iloc[int(period) - 1]
        questions = df[
            (df['onset'] > block_start) &
            (df['onset'] < (block_start + 30)) &
            (df['trial_type'] == 'question')
        ]
        how_badly = questions[
            questions['stimulus'] == "How badly do you feel?"
        ]['response'].iloc[0]
        how_vivid = questions[
            questions['stimulus'] == "How vivid was the memory?"
        ]['response'].iloc[0]
    else:
        print(f"Could not find events file")

    return (
        "nan" if np.isnan(how_badly) else str(int(how_badly)),
        "nan" if np.isnan(how_vivid) else str(int(how_vivid)),
    )


def get_block_metadata(data, start_time, end_time):
    """ Get characteristics of this block from events data.
    """

    df = data[(data['onset'] > start_time) & (data['onset'] < end_time)]
    memory = df[df['trial_type'] == 'memory']['stimulus'].iloc[0]
    instruction = df[df['trial_type'] == 'instruct']['stimulus'].iloc[0]
    feel_bad = df[df['stimulus'] == 'How badly do you feel?']['response'].iloc[0]
    vividness = df[df['stimulus'] == 'How vivid was the memory?']['response'].iloc[0]

    # Retrieve the actual timing bookends for this memory+instruct block
    block_start = float(df[df['trial_type'] == 'memory']['onset'])
    instruct_onset = float(df[df['trial_type'] == 'instruct']['onset'])
    instruct_duration = float(df[df['trial_type'] == 'instruct']['duration'])
    block_end = instruct_onset + instruct_duration

    return {
        "memory": memory,
        "instruct": instruction,
        "feel_bad": feel_bad,
        "vividness": vividness,
        "block_start": block_start,
        "instruct_onset": instruct_onset,
        "block_end": block_end
    }


def get_decoder_name(filename):
    """ From the file name, get a usable column name for the decoder.
    """

    score_file_parts = filename.split("_")
    if score_file_parts[2] in ["negative", "negaff", ]:
        decoder = "negaff"
    elif score_file_parts[2] in ["reappraise", "emoreg", ]:
        decoder = "emoreg"
    else:
        decoder = score_file_parts[2]
        # print(f"Error! Score file {filename} has no decoder name.")
    if "_ones_" in filename:
        weighted = "0"
    elif "_weights_" in filename:
        weighted = "1"
    else:
        weighted = "-1"  # would only ever happen if something failed
        print(f"Error! Score file {filename} has no weighting info.")

    return decoder, weighted


def main(args):
    """ Entry point """

    if args.verbose:
        print(f"Searching {str(args.input_dir)} for score results.")

    # Load one demographic table for all lookups
    demographics = pd.read_csv(args.input_dir / "demographics.csv", index_col=0)

    # Scrape data from all files, categorizing it in memory.
    blocks = {}
    scores = {}
    for subject_dir in args.input_dir.glob("sub-U*"):
        subject = get_val_from_key(subject_dir.name, "sub")
        subject_demographics = demographics.loc[subject]
        task = "mem"
        for run_dir in subject_dir.glob(f"task-{task}/run-*"):
            run = get_val_from_key(run_dir.name, "run")
            print(f"Mining sub-{subject} run-{run} ... ", end='')
            max_fd, num_fd_outliers = get_fd(subject, task, run, args)
            timing_data = get_run_events(run_dir, args)
            print(f"max FD {max_fd:0.2f} with {num_fd_outliers:0d} outlier TRs")
            for score_file in run_dir.glob("decoding/all_trs_*_scores.tsv"):

                # Store metadata only once per subject/run/instruction/period.
                data = pd.read_csv(score_file, index_col=None, header=None)
                memories = timing_data[timing_data['trial_type'] == 'memory']
                memories = memories.reset_index()
                for idx, memory in memories.iterrows():
                    # Extract only timepoints in this block (ignoring other 3)
                    block_metadata = get_block_metadata(
                        timing_data, memory.onset - 0.01, memory.onset + 40.0
                    )
                    # We use a 6 TR (5.4s) shift to account for HRF.
                    # We used 4 to start but just two at the end, based on the
                    # prior matlab decoder, but it was written for 2s TRs, not
                    # 0.9s.
                    # Start TRs get 1 subtracted to include the 0-indexed TR
                    start_idx = int(6 + np.floor(
                        block_metadata['block_start'] / args.tr_dim
                    )) - 1
                    instruct_idx = int(6 + np.floor(
                        block_metadata['instruct_onset'] / args.tr_dim
                    )) - 1
                    end_idx = int(6 + np.ceil(
                        block_metadata['block_end'] / args.tr_dim
                    ))
                    if args.verbose:
                        print(
                            f"Start @ {start_idx:>3} ({block_metadata['block_start']:6.2f}), "
                            f"instruct @ {instruct_idx:>3} ({block_metadata['instruct_onset']:6.2f}), "
                            f"end @ {end_idx:>3}  ({block_metadata['block_end']:6.2f})."
                        )

                    block_id = (subject, run, idx)
                    if block_id not in blocks.keys():
                        # Store block metadata, scores coming separately
                        blocks[block_id] = {
                            "subject": subject,
                            "age": subject_demographics['age'],
                            "sex": subject_demographics['sex'],
                            "suicidality": subject_demographics['suicidality'],
                            "race_n": subject_demographics['race_n'],
                            "race_dich": subject_demographics['race_dich'],
                            "ethnicity": subject_demographics['ethnicity'],
                            "task": task,
                            "run": run,
                            "instruct": block_metadata['instruct'],
                            "period": idx,
                            "orig_start_tr": start_idx,
                            "max_fd": max_fd,
                            "fd_outliers": num_fd_outliers,
                            "feel_bad": block_metadata['feel_bad'],
                            "vividness": block_metadata['vividness'],
                        }
                        scores[block_id] = {}

                    # Each decoder gets its own score data per TR.
                    dec_name, dec_weighted = get_decoder_name(score_file.name)
                    if dec_name not in scores[block_id].keys():
                        scores[block_id][dec_name] = {}
                    scores[block_id][dec_name][dec_weighted] = {}
                    block_scores = data.iloc[start_idx:end_idx, :].values

                    # For instruct studies (Sarah), TR 0 is the beginning of instruct,
                    # and memory period is negative TR.
                    # For memory studies (Christina), the memory cue is TR 0.
                    # For anyone aligning these scores with other timeseries,
                    # orig_tr matches the original uncropped BOLD file (1-based, no zero)
                    if args.zero_point == "instruct":
                        tr_delta = instruct_idx - start_idx
                    else:
                        tr_delta = 0
                    for tr, score in enumerate(block_scores.ravel()):
                        scores[block_id][dec_name][dec_weighted][tr - tr_delta] = score
                    print(
                        f"  retrieved {len(block_scores.ravel())} "
                        f"{dec_name}-{dec_weighted} scores - "
                        f"{subject}.{run}.{idx} ({block_metadata['instruct']})"
                    )

    # Now that all decoder scores are in memory, lay them out properly.
    results = []
    for per_id, rec in blocks.items():
        if per_id not in scores:
            print(f"Error: id {per_id} in blocks, but not scores.")
            continue
        # The triple list comprehension finds all TR values within this period
        tr_values = set([
            tr for dec_key in scores[per_id].keys()
            for wt_key in scores[per_id][dec_key].keys()
            for tr in list(scores[per_id][dec_key][wt_key].keys())
        ])
        for tr_idx, tr in enumerate(sorted(tr_values)):
            for dec_name in scores[per_id].keys():
                # Start with a copy of the block's metadata, then add scores
                result = rec.copy()
                result['tr'] = tr
                result['orig_tr'] = (
                    tr_idx
                    + result['orig_start_tr']
                    + args.steady_state_outliers
                )
                del result['orig_start_tr']
                result['decoder'] = dec_name
                for dec_wt in scores[per_id][dec_name].keys():
                    if dec_wt == "0":
                        score_str = "average_score"
                    elif dec_wt == "1":
                        score_str = "weighted_score"
                    else:
                        score_str = "error"
                    if tr in scores[per_id][dec_name][dec_wt].keys():
                        result[score_str] = scores[per_id][dec_name][dec_wt][tr]
                results.append(result)

    final_results = pd.DataFrame(results).sort_values(
        ["subject", "run", "period", "decoder", ]
    )
    final_results.to_csv(args.output_file, index=False)
    if args.verbose:
        print(f"Wrote scores to {str(args.output_file)}.")


if __name__ == "__main__":
    main(get_arguments())
