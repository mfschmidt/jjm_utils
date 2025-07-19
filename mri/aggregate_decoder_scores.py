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
        "--preproc-path",
        default="/data/BI/human/derivatives/new_conte/fmriprep",
        help="The path to fMRIPrep or feat data, containing subject "
             "directories with pre-processing support files"
    )
    parser.add_argument(
        "--rawdata-path",
        default="/data/BI/human/derivatives/new_conte/rawdata",
        help="The path to raw BIDS data, containing subject directories"
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
    if not Path(args.preproc_path).exists():
        print(f"preproc path '{args.preproc_path}' does not exist.")
        need_to_bail = True
    if not Path(args.rawdata_path).exists():
        print(f"Raw data path '{args.rawdata_path}' does not exist.")
        need_to_bail = True
    if not Path(args.output_file).parent.exists():
        print(f"{str(Path(args.output_file).parent)} doesn't exist.")
        need_to_bail = True
    if need_to_bail:
        sys.exit(1)

    # Save paths as Path objects rather than strings.
    setattr(args, "input_dir", Path(args.input_dir).absolute())
    setattr(args, "output_file", Path(args.output_file).absolute())
    setattr(args, "preproc_path", Path(args.preproc_path).absolute())
    setattr(args, "rawdata_path", Path(args.rawdata_path).absolute())

    # We are only using this on mem tasks, and we can update this if needed.
    setattr(args, "task", "mem")

    return args


def get_val_from_key(filename, key):
    """ Extract and return a value from "..._key-value_..." string.
    """

    match = re.search(rf"{key}-([0-9A-Za-z]+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    # Override for old-style subject-named directories without "sub-" prefix
    if key == "sub" and filename.startswith("U"):
        return filename
    return None


def get_fd(subject, session, run, args):
    """ Find the uncropped confounds file and extract cropped FD.
    """

    # We assume the confounds will be in an fMRIPrep or feat directory,
    # but will also support the feat-based rms file if that's the only choice.
    # If fMRIPrep was used:
    fmriprep_glob_pattern = list(args.preproc_path.glob(
        f"**/sub-{subject}*ses-{session}_*task-{args.task}_run-*{run}_"
        "desc-confounds_timeseries.tsv"
    ))
    feat_glob_pattern_1 = list(args.preproc_path.glob(
        f"{subject}/mem/run{run}/preproc.feat/mc/"
        "prefiltered_func_data_mcf_rel.rms"
    ))
    feat_glob_pattern_2 = list(args.preproc_path.glob(
        f"{subject}/mems/run{run}/preproc.feat/mc/"
        "prefiltered_func_data_mcf_rel.rms"
    ))
    feat_glob_pattern_3 = list(args.preproc_path.glob(
        f"sub-{subject}_ses-{session}/task-{args.task}_run-{int(run):02d}/"
        "trial-01_lev-1.feat/mc/prefiltered_func_data_mcf_rel.rms"
    ))
    confound_files = (
        fmriprep_glob_pattern + feat_glob_pattern_1 +
        feat_glob_pattern_2 + feat_glob_pattern_3
    )
    if args.verbose:
        print(f"  found {len(confound_files)} motion file possibilities:")
    for confound_file in confound_files:
        if confound_file.exists():
            if confound_file.name.endswith(".tsv"):
                # An fMRIPrep tsv file
                try:
                    df = pd.read_csv(
                        confound_files[0], index_col=None, header=0, sep='\t'
                    )
                    fd = df['framewise_displacement'][args.steady_state_outliers:].astype(float)
                    return fd.max(), fd.mean(), len(fd[fd > args.motion_threshold])
                except KeyError:
                    print(f"Could not find 'framewise_displacement' in {confound_file.name}")
                    # continue on to the next file, no harm done
            if confound_file.name.endswith(".rms"):
                # An fMRIPrep tsv file
                df = pd.read_csv(
                    confound_files[0], index_col=None, header=None, sep='\t'
                )
                fd = df[args.steady_state_outliers - 1:].values.astype(float)
                return fd.max(), fd.mean(), len(fd[fd > args.motion_threshold])
    # If the searching all failed,
    print(f"Confounds for {subject}/{session}/{args.task}/run-{run} "
          "could not be found.")
    return 0.0, 0.0, 0


def calc_fd(row):
    """ From a row of six displacement parameters, summarize them.

        This is based on the same Power, 2012 calculation used by fmriprep.
        But it doesn't match fMRIPrep, so shouldn't yet be used.
    """
    return np.sum(np.abs(np.array([
        row['trans_x'],
        row['trans_y'],
        row['trans_z'],
        50.0 * np.sin(row["rot_x"] * 180.0 / np.pi),
        50.0 * np.sin(row["rot_y"] * 180.0 / np.pi),
        50.0 * np.sin(row["rot_z"] * 180.0 / np.pi),
    ])))


def get_run_events(raw_func_dir, task, run, tr_dim, steady_state_outliers):
    """ Find the uncropped events file and return its cropped/shifted data.
    """

    time_shift = tr_dim * steady_state_outliers
    # There should be one and only one events file per run, so assume it's so
    for ev_file in raw_func_dir.glob(f"sub-*_task-{task}_run-?{run}_events.tsv"):
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
    try:
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
    except IndexError:
        return None

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
        "other_id": demographics.get('other_id', "na"),
        "age": demographics.get('Age', demographics.get('age', "na")),
        "sex": demographics.get('sex', "na"),
        "suicidality": demographics.get('suicidality', "na"),
        "race_n": demographics.get('race_n', "na"),
        "race_dich": demographics.get('race_dich', "na"),
        "ethnicity": demographics.get('ethnicity', "na"),
        "bdi_base":  demographics.get('BDI_Base', "na"),
        "dbt1_ssri0":  demographics.get('dbt1_ssri0', "na"),
        "bdi_6month":  demographics.get('T2_BDI_6MO', "na"),
        "ders_base":  demographics.get('DERS_Base', "na"),
        "ders_6month":  demographics.get('T2_DERS_6MO', "na"),
        "zan_base":  demographics.get('ZAN_BaseTOT', "na"),
        "zan_6month":  demographics.get('T2_ZAN_6MO', "na"),
        "ssi_base":  demographics.get('SSICURBaseline', "na"),
        "ssi_6month":  demographics.get('T2_SSI_6MO', "na"),
        "als_base":  demographics.get('ALS_Base', "na"),
        "als_6month":  demographics.get('T2_ALS_6MO', "na"),
        "ham_base":  demographics.get('HAM17_bl', "na"),
        "ham_6month":  demographics.get('T2_HAM17_6', "na"),
        "bis_base":  demographics.get('BISTOTNnew_B', "na"),
        "bis_6month":  demographics.get('T2_BISTOTNnew_6MO', "na"),
        "busdk_base":  demographics.get('BUSDK_Base', "na"),
        "busdk_6month":  demographics.get('T2_BUSDK_6MO', "na"),
        "nssi_bl":  demographics.get('nssi_bl', "na"),
        "treatment": demographics.get('treatment', "na"),
        "baseline_DERS": demographics.get('baseline_DERS', "na"),
        "post_DERS": demographics.get('post_DERS', "na"),
        "baseline_HAM": demographics.get('baseline_HAM', "na"),
        "post_HAM": demographics.get('post_HAM', "na"),
        "baseline_BDI": demographics.get('baseline_BDI', "na"),
        "post_BDI": demographics.get('post_BDI', "na"),
        "baseline_SSI": demographics.get('baseline_SSI', "na"),
        "baseline_ZANTOT": demographics.get('baseline_ZANTOT', "na"),
        "baseline_BUSS": demographics.get('baseline_BUSS', "na"),
        "baseline_BIS": demographics.get('baseline_BIS', "na"),
        "attempts": demographics.get('attempts', "na"),
        "num_sui": demographics.get('num_sui', "na"),
        "abuse_phys": demographics.get('abuse_phys', "na"),
        "abuse_sex": demographics.get('abuse_sex', "na"),
        "neg_emot": demographics.get('neg_emot', "na"),
        "neg_phys": demographics.get('neg_phys', "na"),
    }


def get_blocks_from_run(subject, session, run_dir, raw_func_dir, args):
    """ Read blocks from one run, and organize them alongside other information.
    """

    task = get_val_from_key(str(run_dir), "task")
    run = get_val_from_key(run_dir.name, "run")
    print(f"Mining sub-{subject} ses-{session} task-{task} run-{run} ... ", end='')

    run_blocks = {}

    # Get steady-state cropped events times
    timing_data = get_run_events(
        raw_func_dir, task, int(run), args.tr_dim, args.steady_state_outliers
    )

    # Get steady-state cropped movement confounds
    max_fd, mean_fd, num_fd_outliers = get_fd(subject, session, run, args)
    if (
            (max_fd is not None) and
            (mean_fd is not None) and
            (num_fd_outliers is not None) and
            args.verbose
    ):
        print(f"FD:(sub={subject},ses={session},task={task},run={run},"
              f"maxfd={max_fd:0.4f},meanfd={mean_fd:0.4f},"
              f"outliertrs={num_fd_outliers:0d})")

    if (timing_data is None) or (max_fd is None) or (num_fd_outliers is None):
        return {}

    # And drop all but the four 'memory' events
    memories = timing_data[timing_data['trial_type'] == 'memory']
    memories = memories.reset_index()
    for idx, memory in memories.iterrows():
        # Extract only timepoints in this block (ignoring other 3)
        # the 52s gets past the memory/instruct without hitting the next block.
        # This was 40 in Conte, is 52 in BPD, not sure if there's a universal.
        block_metadata = get_block_metadata(
            timing_data, memory.onset - 0.01, memory.onset + 52.0
        )
        if block_metadata is None:
            continue
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

        block_id = (subject, session, run, idx)
        if block_id in run_blocks.keys():
            print(f"Duplicate block {block_id}!")
        else:
            # Store block metadata, scores coming separately
            run_blocks[block_id] = {
                "task": args.task,
                "session": session,
                "run": run,
                "period": idx,
                "orig_start_tr": memory_idx + args.steady_state_outliers,
                "memory_tr": memory_idx,  # Final, in ss-cropped reference
                "instruct_tr": instruct_idx,  # Final, in ss-cropped reference
                "end_tr": end_idx,  # Final, in ss-cropped reference
                "max_fd": 0 if max_fd is None else max_fd,
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
        "{subject}.{session}.{run}.{period} ({instruct})".format(**block)
    )

    return run_scores


def main():
    """ Entry point """

    dt1 = datetime.now().strftime("%Y%m%d %H:%M:%S")
    args = get_arguments()
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
    subject_dirs = sorted(
        list(args.input_dir.glob("U*")) +
        list(args.input_dir.glob("ERBPD[0-9][0-9][0-9]")) +
        list(args.input_dir.glob("sub-*"))
    )
    if args.verbose:
        print(f"Found {len(subject_dirs)} subject directories.")
    for subject_dir in subject_dirs:
        # Use sub-ID or ID as equivalent
        if subject_dir.name.startswith("sub-"):
            subject_id = subject_dir.name[4:]
        else:
            subject_id = subject_dir.name

        # Find this subject in the demographics table.
        if subject_id in demographics.index:
            subject_dict = get_subject_data(
                subject_id, demographics.loc[subject_id]
            )
        else:
            subject_dict = get_subject_data(
                subject_id, pd.Series(dtype=str)
            )

        # Some outputs have session directories, some don't.
        session_subdirs = sorted(
            list(subject_dir.glob("ses*")) +
            list(subject_dir.glob("session*"))
        )
        if len(session_subdirs) == 0:
            # If there are no sessions, fine, just use the subject dir
            session_subdirs = [subject_dir, ]

        for session_dir in session_subdirs:
            if session_dir == subject_dir:
                session_id = "na"
            else:
                session_id = get_val_from_key(session_subdirs[0].name, "ses")
                if session_id is None:
                    session_id = get_val_from_key(session_subdirs[0].name, "session")

            run_dirs = sorted(
                list(session_dir.glob(f"task-{args.task}/run-*")) +
                list(session_dir.glob(f"task-{args.task}_run-*")) +
                list(session_dir.glob(f"{args.task}s/run[0-9]"))
            )
            if "task" in session_dir.name:
                run_dirs += list(session_dir.glob("run*"))
            for run_dir in run_dirs:
                raw_func_dir_candidates = list(
                    (args.rawdata_path / f"sub-{subject_id}").glob(f"ses-*/func")
                )
                if len(raw_func_dir_candidates) > 0:
                    raw_func_dir = raw_func_dir_candidates[0]
                else:
                    print("|*")
                    print(f"|* WARNING: no raw func dir for {subject_id}")
                    print("|*")
                    continue

                # Using all available run information, find relevant events
                some_blocks = get_blocks_from_run(
                    subject_id, session_id, run_dir, raw_func_dir, args
                )
                if len(some_blocks) == 0:
                    print(f"MISSING: {str(raw_func_dir)}")
                    continue
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

                for score_file in sorted(
                    list(run_dir.glob("all_trs_*_scores.tsv")) +
                    list(run_dir.glob("decoding/all_trs_*_scores.tsv"))
                ):
                    # See if we are interested in this particular decoder's output.
                    dec_name, dec_weighted = get_decoder_name(score_file.name)
                    # If no --decoder-names are provided, use them all. But if some are listed,
                    # restrict our aggregation to included decoders.
                    if args.decoder_names is not None:
                        if dec_name not in args.decoder_names:
                            continue

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

    # Put all data into a dataframe, and finalize the data
    final_results = pd.DataFrame(results)
    # Create a unique identifier for each period.
    final_results["pid"] = final_results.apply(
        lambda r: f"{r['subject']}_{r['session']}_{r['task']}_{r['run']}_{r['period']}_{r['instruct'][0]}",
        axis=1,
    )
    # Then change the period to indicate task-dependent rather than absolute
    final_results['a_period'] = final_results['period'].copy()
    final_results['period'] = 0
    sri_idx = final_results.sort_values(
        ["subject", "run", "instruct"]
    ).groupby(
        ["subject", "run", "instruct"]
    )["pid"].count().index
    for subject, run, instruct in sri_idx:
        sri_mask = (
            (final_results["subject"] == subject)
            & (final_results["run"] == run)
            & (final_results["instruct"] == instruct)
        )
        periods = sorted(final_results[sri_mask]["a_period"].unique())
        for i, period in enumerate(periods):
            this_period_mask = sri_mask & (final_results["a_period"] == period)
            final_results.loc[this_period_mask, "period"] = i + 1

    # Sort and order results, without changing any data
    fields_to_keep = [
        'subject', 'other_id', 'age', 'sex', 'treatment', 'suicidality',
        'race_n', 'race_dich','ethnicity', 'dbt1_ssri0', 'bdi_base',
        'bdi_6month', 'ders_base', 'ders_6month', 'zan_base', 'zan_6month',
        'ssi_base', 'ssi_6month', 'als_base', 'als_6month', 'ham_base',
        'ham_6month', 'bis_base', 'bis_6month', 'busdk_base', 'busdk_6month',
        'nssi_bl', 'baseline_DERS', 'baseline_HAM', 'baseline_BDI',
        'baseline_SSI', 'baseline_ZANTOT', 'baseline_BUSS', 'baseline_BIS',
        'post_BDI', 'post_DERS', 'post_HAM', 'attempts', 'num_sui',
        'abuse_phys', 'abuse_sex', 'neg_emot', 'neg_phys',
        'task', 'session', 'run', 'instruct', 'period', 'pid',
        'max_fd', 'fd_outliers', 'feel_bad', 'vividness',
        'tr_from_scan_start', 'tr_from_memory', 'tr_from_instruct',
        'decoder', 'weighted_score', 'average_score',
    ]
    final_results = final_results.sort_values(
        ["pid", "decoder", ]
    )[[f for f in fields_to_keep if f in final_results.columns]]
    final_results.to_csv(args.output_file, index=False)
    dt2 = datetime.now().strftime("%Y%m%d %H:%M:%S")
    if args.verbose:
        print(f"Final table contains {final_results.shape[0]} observations "
              f"with {final_results.shape[1]} features each.")
        print(f"Wrote scores to {str(args.output_file)} ({dt2}).")


if __name__ == "__main__":
    main()
