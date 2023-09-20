#!/usr/bin/env python3

# aggregate_decoder_scores.py

import argparse
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


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
        "--decoder-names", nargs='+',
        help="Explicitly list decoders to aggregate, rather than all of them"
    )
    parser.add_argument(
        "--tr-dim", default=0.9, type=float,
        help="How many seconds in the 4th dimension from one volume to the next"
    )
    parser.add_argument(
        "--hrf-shift", default=5.4, type=float,
        help="How many seconds to shift from event to HRF response"
    )
    parser.add_argument(
        "--demographics-file", default="./demographics.csv",
        help="File containing diagnoses, ages, etc"
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

    # We are only using this on mem tasks, and we can update this if needed.
    setattr(args, "task", "mem")

    return args


def get_val_from_key(filename, key):
    """ Extract and return a value from "..._key-value_..." string.
    """

    match = re.search(rf"{key}-([0-9A-Za-z]+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_fd(subject, run, args):
    """ Find the uncropped confounds file and extract cropped FD.
    """

    confound_files = list((args.fmriprep_path / f"sub-{subject}").glob(
        f"**/sub-{subject}*task-{args.task}_run-{run}_"
        "desc-confounds_timeseries.tsv"
    ))
    if len(confound_files) > 0:
        df = pd.read_csv(confound_files[0], index_col=None, sep='\t')
        fd = df['framewise_displacement'][args.steady_state_outliers:]
        return fd.max(), len(fd[fd > args.motion_threshold])
    else:
        print(f"Confounds for {subject}/{args.task}/{run} could not be found.")
        return None, None


def get_run_events(run_dir, args):
    """ Find the uncropped events file and return its cropped/shifted data.
    """

    time_shift = args.tr_dim * args.steady_state_outliers
    # There should be one and only one events file per run, so assume it's so
    for ev_file in run_dir.glob("regressors/sub-*_task-*_run-*_events.tsv"):
        if ev_file.exists():
            print(f"Reading {ev_file.name}")
            df = pd.read_csv(ev_file, index_col=None, header=0, sep="\t")
            df = df.sort_values('onset')
            df['onset'] = df['onset'] - time_shift
            return df
    return None


def get_block_metadata(data, start_time, end_time):
    """ Get characteristics of this block from events data.
    """

    df = data[(data['onset'] > start_time) & (data['onset'] < end_time)]
    memory = df[
        df['trial_type'] == 'memory'
    ]['stimulus'].iloc[0]
    instruction = df[
        df['trial_type'] == 'instruct'
    ]['stimulus'].iloc[0]
    feel_bad = df[
        df['stimulus'] == 'How badly do you feel?'
    ]['response'].iloc[0]
    vividness = df[
        df['stimulus'] == 'How vivid was the memory?'
    ]['response'].iloc[0]

    # Retrieve the actual timing bookends for this memory+instruct block
    memory_onset = float(
        df[df['trial_type'] == 'memory']['onset'].iloc[0]
    )
    memory_duration = float(
        df[df['trial_type'] == 'memory']['duration'].iloc[0]
    )
    instruct_onset = float(
        df[df['trial_type'] == 'instruct']['onset'].iloc[0]
    )
    instruct_duration = float(
        df[df['trial_type'] == 'instruct']['duration'].iloc[0]
    )
    block_end = instruct_onset + instruct_duration

    return {
        "memory": memory,
        "instruct": instruction,
        "feel_bad": feel_bad,
        "vividness": vividness,
        "memory_onset": memory_onset,
        "memory_duration": memory_duration,
        "instruct_onset": instruct_onset,
        "block_end": block_end
    }


def get_decoder_name(filename):
    """ From the file name, get a usable column name for the decoder.
    """

    updated_names = {
        "negative": "negaff",
        "reappraise": "emoreg",
    }
    pattern = re.compile(r"all_trs_(\w*)_([A-Za-z]*)_scores\.tsv")
    match = pattern.search(filename)
    if match:
        # If there's an alternate name, translate it
        decoder = updated_names.get(match.group(1), match.group(1))
        weighted = match.group(2)
        return decoder, weighted
    else:
        print(f"ERROR: could not interpret score filename '{filename}'.")
        return "unknown", "unknown"


def get_subject_data(subject_id, demographics):
    """ Return a dict with subject-level data.
    """

    return {
        "subject": subject_id,
        "age": demographics.get('age', "-1"),
        "sex": demographics.get('sex', "-1"),
        "suicidality": demographics.get('suicidality', "-1"),
        "race_n": demographics.get('race_n', "-1"),
        "race_dich": demographics.get('race_dich', "-1"),
        "ethnicity": demographics.get('ethnicity', "-1"),
    }


def get_blocks_from_run(subject, run_dir, args):
    """ Read blocks from one run, and organize them alongside other information.
    """

    run = get_val_from_key(run_dir.name, "run")
    print(f"Mining sub-{subject} run-{run} ... ", end='')

    run_blocks = {}

    # Get steady-state cropped events times
    timing_data = get_run_events(run_dir, args)

    # Get steady-state cropped movement confounds
    max_fd, num_fd_outliers = get_fd(subject, run, args)
    if args.verbose:
        print(f"max FD {max_fd:0.2f} with {num_fd_outliers:0d} outlier TRs")

    # And drop all but the four 'memory' events
    memories = timing_data[timing_data['trial_type'] == 'memory']
    memories = memories.reset_index()
    for idx, memory in memories.iterrows():
        # Extract only timepoints in this block (ignoring other 3)
        # the 40s gets past the memory/instruct without hitting the next block.
        block_metadata = get_block_metadata(
            timing_data, memory.onset - 0.01, memory.onset + 40.0
        )
        # We use a 6 TR (5.4s) shift to account for HRF (unless overridden).
        # We previously used 4 to start but just 2 at the end, based on the
        # prior matlab decoder, but it was written for 2s TRs, not
        # 0.9s.
        #
        # Example (cropping done when file was loaded, not here):
        #   for a block with events       memory @ 100.0s and instruct @ 112.0s:
        #   cropping 7 0.9s TRs shifts to memory @  93.7s and instruct @ 105.7s
        #   hrf-shifting by 6s            memory @  99.1s and instruct @ 111.1s
        #   'floor'ing to the current TR  memory @ TR#110 and instruct @ TR#127
        #                                           99.0s                110.7s
        #
        memory_idx = int(np.floor(
            (block_metadata['memory_onset'] + args.hrf_shift) / args.tr_dim
        ))
        instruct_idx = int(np.floor(
            (block_metadata['instruct_onset'] + args.hrf_shift) / args.tr_dim
        ))
        end_idx = int(np.ceil(
            (block_metadata['block_end'] + args.hrf_shift) / args.tr_dim
        ))
        if args.verbose:
            print(
                f"Start @ TR {memory_idx:>3} "
                f"(floor(({block_metadata['memory_onset']:6.2f}s "
                f"+ {args.hrf_shift}s) / {args.tr_dim:0.1f}s/tr)), "
                f"instruct @ TR {instruct_idx:>3} "
                f"(floor(({block_metadata['instruct_onset']:6.2f} "
                f"+ {args.hrf_shift}s) / {args.tr_dim:0.1f})), "
                f"end @ TR {end_idx:>3}  "
                f"(ceil(({block_metadata['block_end']:6.2f} "
                f"+ {args.hrf_shift}s) / {args.tr_dim:0.1f}))."
            )

        block_id = (subject, run, idx)
        if block_id in run_blocks.keys():
            print(f"Duplicate block {block_id}!")
        else:
            # Store block metadata, scores coming separately
            run_blocks[block_id] = {
                "task": args.task,
                "run": run,
                "period": idx,
                "orig_start_tr": memory_idx + args.steady_state_outliers,
                "memory_tr": memory_idx,  # Final, in ss-cropped reference
                "instruct_tr": instruct_idx,  # Final, in ss-cropped reference
                "end_tr": end_idx,  # Final, in ss-cropped reference
                "max_fd": max_fd,
                "fd_outliers": num_fd_outliers,
            }
            run_blocks[block_id].update(block_metadata)

    return run_blocks


def get_scores_from_block(block, score_vec, dec_name, dec_weighted):
    """ Read blocks from one run, and organize them alongside other information.
    """

    # Each decoder gets its own score data per TR.
    run_scores = {
        dec_name: {
            dec_weighted: {
            }
        }
    }
    block_scores = score_vec[block['memory_tr']:block['end_tr']]

    for tr, score in enumerate(block_scores.ravel()):
        # Save scores as (tr == 0 at memory cue), tr: score
        run_scores[dec_name][dec_weighted][tr] = score
    print(
        f"  retrieved {len(block_scores.ravel())} "
        f"{dec_name} - {dec_weighted} scores - "
        "{subject}.{run}.{period} ({instruct})".format(**block)
    )

    return run_scores


def main(args):
    """ Entry point """

    dt1 = datetime.now().strftime("%Y%m%d %H:%M:%S")
    if args.verbose:
        print(f"Searching {str(args.input_dir)} for score results ({dt1}).")

    # Load one demographic table for all lookups
    if Path(args.demographics_file).exists():
        print(f"Reading {Path(args.demographics_file).name}")
        demographics = pd.read_csv(Path(args.demographics_file), index_col=0)
    else:
        # An empty dataframe will allow us to continue, ignoring missing info
        demographics = pd.DataFrame()

    # Scrape data from all files, categorizing it in memory.
    blocks = {}
    scores = {}
    for subject_dir in sorted(args.input_dir.glob("sub-U*")):
        subject = get_val_from_key(subject_dir.name, "sub")
        if subject in demographics.index:
            subject_dict = get_subject_data(
                subject, demographics.loc[subject]
            )
        else:
            subject_dict = get_subject_data(
                subject, pd.Series(dtype=str)
            )
        for run_dir in sorted(subject_dir.glob(f"task-{args.task}/run-*")):
            some_blocks = get_blocks_from_run(subject, run_dir, args)
            for b_key, block in some_blocks.items():
                # Add subject demographic data to the block
                block.update(subject_dict)
                # And save it to the global block storage
                if b_key not in blocks.keys():
                    blocks[b_key] = block
                else:
                    blocks[b_key].update(block)
                if b_key not in scores.keys():
                    scores[b_key] = {}

            for score_file in run_dir.glob("decoding/all_trs_*_scores.tsv"):
                # See if we are interested in this particular decoder's output.
                dec_name, dec_weighted = get_decoder_name(score_file.name)
                if args.decoder_names is not None:
                    if dec_name not in args.decoder_names:
                        continue
                # else go ahead and collect all decoders...

                # Read the scores and start sorting them out.
                print(f"Reading {score_file.name}")
                score_df = pd.read_csv(score_file, index_col=None, header=None)
                score_vec = score_df.values
                for b_key, block in some_blocks.items():
                    some_scores = get_scores_from_block(
                        block, score_vec, dec_name, dec_weighted
                    )
                    for d_key in some_scores.keys():
                        if d_key not in scores[b_key].keys():
                            scores[b_key][d_key] = some_scores[d_key]
                        else:
                            for w_key in some_scores[d_key].keys():
                                if w_key not in scores[b_key][d_key].keys():
                                    scores[b_key][d_key][w_key] = \
                                        some_scores[d_key][w_key]

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
            # tr_idx and tr should be the same thing; we started at zero
            if tr != tr_idx:
                print(f"ERROR: TRs OFF: {tr} vs {tr_idx}")
            for dec_name in scores[per_id].keys():
                # Start with a copy of the block's metadata, then add scores
                result = rec.copy()

                # For memory studies (as the earliest cue, and for Christina),
                # the memory cue is TR 0, and the whole block is positive.
                # For instruct studies (original, and for Sarah),
                # TR 0 is the beginning of instruct,
                # and the preceding memory period has negative TRs.
                # For anyone aligning these scores with other timeseries,
                # orig_tr matches the original un-cropped BOLD file
                tr_delta = result['instruct_tr'] - result['memory_tr']
                result['tr_from_scan_start'] = tr + result['orig_start_tr']
                result['tr_from_memory'] = tr
                result['tr_from_instruct'] = tr - tr_delta
                # Avoid later ambiguity and confusion from too many options
                del result['orig_start_tr']
                del result['memory_tr']
                del result['instruct_tr']

                result['decoder'] = dec_name
                for dec_wt in scores[per_id][dec_name].keys():
                    if dec_wt == "ones":
                        score_str = "average_score"
                    elif dec_wt == "weights":
                        score_str = "weighted_score"
                    else:
                        score_str = "error"
                    if tr in scores[per_id][dec_name][dec_wt].keys():
                        result[score_str] = scores[per_id][dec_name][dec_wt][tr]
                results.append(result)

    # Sort and order results, without changing any data
    final_results = pd.DataFrame(results).sort_values(
        ["subject", "run", "period", "decoder", ]
    )[[
        'subject', 'age', 'sex', 'suicidality', 'race_n', 'race_dich',
        'ethnicity', 'task', 'run', 'instruct', 'period',
        'max_fd', 'fd_outliers', 'feel_bad', 'vividness',
        'tr_from_scan_start', 'tr_from_memory', 'tr_from_instruct',
        'decoder', 'weighted_score', 'average_score',
    ]]
    final_results.to_csv(args.output_file, index=False)
    dt2 = datetime.now().strftime("%Y%m%d %H:%M:%S")
    if args.verbose:
        print(f"Wrote scores to {str(args.output_file)} ({dt2}).")


if __name__ == "__main__":
    main(get_arguments())
