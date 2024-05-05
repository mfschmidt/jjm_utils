#!/usr/bin/env python3

"""
extract_conte_one_memory_difference_ratings.py

A typical invocation:

    To loop over all subjects in the standard rawdata space,
    and extract all of their ratings, combining them into a
    tsv file with difference ratings at
    /home/mike/Desktop/difference_ratings.csv

    extract_conte_one_image_difference_ratings.py \
    /home/mike/data/BI/human/rawdata/old_conte \
    /home/mike/Desktop \
    --verbose
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import re


# Some globals


def get_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Extract ratings of memories in the Conte1 data.",
    )
    parser.add_argument(
        "input_path",
        help="the path containing BIDS-compliant rawdata",
    )
    parser.add_argument(
        "output_path",
        help="The output path for writing a tsv file",
    )
    parser.add_argument(
        "--participant",
        default="all",
        help="The subject/participant to work with, leave blank for all",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Subjects to avoid looking through",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    return args


def get_env(args):
    """Integrate environment variables into our args."""

    return args


def validate_args(args):
    """Ensure the environment will support the requested workflows."""

    errors = []

    if args.participant.startswith("sub-"):
        setattr(args, "participant", args.participant[4:])

    setattr(args, "input_path", Path(args.input_path).resolve())
    if not args.input_path.exists():
        errors.append(f"ERR: The input path '{str(args.input_path)}' "
                      "doesn't exist.")

    setattr(args, "output_path", Path(args.output_path).resolve())
    if not args.output_path.exists():
        errors.append(f"ERR: The output path '{str(args.output_path)}' "
                      "doesn't exist.")

    if args.exclude is None:
        setattr(args, "exclude", [])

    if len(errors) > 0:
        for error in errors:
            print(error)
            sys.exit(1)

    return args


def extract_from_subject(subject_path):
    """Do the work on a participant."""

    ratings = []
    pattern = re.compile(
        r"sub-([A-Z][0-9]+)_ses-([0-9]+)_task-(mem)_run-([0-9]+)_events.tsv"
    )
    for events_file in sorted(
        subject_path.glob("ses-*/func/sub-*_ses-*_task-mem*events.tsv")
    ):
        subject_id, session_id, task, run = "NA", "NA", "NA", "NA"
        match = pattern.match(events_file.name)
        if match:
            subject_id = match.group(1)
            session_id = match.group(2)
            task = match.group(3)
            run = match.group(4)
        else:
            print(f"ERR: No RE match for '{events_file.name}'")
        print(
            f" {subject_path.name}/{subject_id} {session_id:>6}: "
            f"{task:<6} {run:>3} - {str(events_file)}"
        )
        run_df = pd.read_csv(events_file, sep="\t", header=0, index_col=None)

        # Group events into trials, and store them
        memory_prompt, instruction = None, None
        bad_rating, bad_rt = None, None
        vivid_rating, vivid_rt = None, None
        period = 0
        events_in_trial = 0
        for idx, row in run_df.sort_values("onset").iterrows():
            events_in_trial += 1
            if row["trial_type"] == "memory":
                memory_prompt = row["stimulus"]
            elif row["trial_type"] == "instruct":
                instruction = row["stimulus"]
            elif row["trial_type"] == "question":
                if row["stimulus"] == "How badly do you feel?":
                    bad_rating = row["response"]
                    bad_rt = row["response_time"]
                elif row["stimulus"] == "How vivid was the memory?":
                    vivid_rating = row["response"]
                    vivid_rt = row["response_time"]
            if not (
                None
                in [
                    memory_prompt,
                    instruction,
                    bad_rating,
                    bad_rt,
                    vivid_rating,
                    vivid_rt,
                ]
            ):  # if the trial record is complete, with all parts filled in,
                period += 1
                ratings.append(
                    {
                        "subject": subject_id,
                        "session": session_id,
                        "task": task,
                        "run": run,
                        "period": period,
                        "memory": memory_prompt,
                        "trial_type": instruction,
                        "rating_type": "bad",
                        "rating": bad_rating,
                        "rt": bad_rt,
                    }
                )
                ratings.append(
                    {
                        "subject": subject_id,
                        "session": session_id,
                        "task": task,
                        "run": run,
                        "period": period,
                        "memory": memory_prompt,
                        "trial_type": instruction,
                        "rating_type": "vivid",
                        "rating": vivid_rating,
                        "rt": vivid_rt,
                    }
                )
                memory_prompt, instruction = None, None
                bad_rating, bad_rt = None, None
                vivid_rating, vivid_rt = None, None
                events_in_trial = 0

    return ratings


def main(args):
    """Entry point"""

    # Discover which participants to run
    if args.participant == "all":
        subject_dirs = [
            d
            for d in args.input_path.glob("sub-[A-Z][0-9]*")
            if re.match(r"sub-[A-Z][0-9]+", d.name)
        ]
    else:
        subject_dirs = [args.input_path / f"sub-{args.participant}"]

    # Run them
    ratings = []
    subject_count = 0
    for subject_dir in subject_dirs:
        exclude = False
        for exclusion in args.exclude:
            if exclusion in subject_dir.name:
                print(f"excluding '{exclusion}'=='{subject_dir.name}'")
                exclude = True
        if not exclude:
            # ACTUAL EXTRACTION WORK gets done in extract_from_subject()
            print(f"SUBJECT {subject_dir.name}:")
            ratings = ratings + extract_from_subject(subject_dir)
            subject_count += 1

    # Save out results
    print(
        f"Combining {len(ratings)} ratings from {subject_count} "
        f"participants and writing to "
        f"{str(args.output_path / 'conte_one_mem_ratings_by_trial.csv')}"
    )
    # Create a dataframe, and keep columns consistently ordered.
    data = pd.DataFrame(ratings)[
        [
            "subject",
            "session",
            "task",
            "run",
            "period",
            "memory",
            "trial_type",
            "rating_type",
            "rating",
            "rt",
        ]
    ]
    for col in [
        "rating",
        "rt",
    ]:
        data[col] = data[col].astype(float)
    data["memory"] = data["memory"].astype(str)

    # In case two subjects called their memories the same thing,
    # create a unique subject-specific memory id.
    data["mem_id"] = data.apply(
        lambda row: f"{row['subject']} {row['memory']}", axis=1
    )

    # As of 11/5/2023, with ? subjects, data have ? rows.

    # Break down by memory
    # Calculate the mean immerse rating for each memory
    # For each distance trial, calculate 'trial rating' - 'mean immerse rating'
    # (With ? subjects, ~? immerses and ~? distances for ? memories)
    immerse_filter = data["trial_type"] == "immerse"
    distance_filter = data["trial_type"] == "distance"
    for mem_id in data["mem_id"].unique():
        mem_filter = data["mem_id"] == mem_id
        print(f" *add* {np.sum(mem_filter)}/4 ratings: "
              f"{np.sum(immerse_filter & mem_filter)}/2 immerse, "
              f"{np.sum(distance_filter & mem_filter)}/2 distance, "
              f"'{mem_id}'")
        if np.sum(mem_filter) != 4:
            print(
                f"ERROR: The memory, '{mem_id}' has "
                f"{np.sum(data['mem_id'] == mem_id)} occurrence(s). "
                "Skipping it"
            )
            continue
        elif (
                (np.sum(immerse_filter & mem_filter) != 2) |
                (np.sum(distance_filter & mem_filter) != 2)
        ):
            print(
                f"ERROR: '{mem_id}' has "
                f"{np.sum(immerse_filter & mem_filter)} immerse and "
                f"{np.sum(distance_filter & mem_filter)} distance "
                "events. Skipping this one."
            )
            continue

        # We are confident there are exactly 2 events with this memory,
        # and one is a distance trial, and one is an immerse trial.
        i_filter = immerse_filter & mem_filter
        d_filter = distance_filter & mem_filter
        for rating_type in [
            "vivid",
            "bad",
        ]:
            rt_filter = data['rating_type'] == rating_type
            i_score = data[i_filter & rt_filter].iloc[0]['rating']
            d_score = data[d_filter & rt_filter].iloc[0]['rating']
            data.loc[mem_filter & rt_filter, "immerse_rating"] = i_score
            data.loc[mem_filter & rt_filter, "distance_rating"] = d_score
            data.loc[mem_filter & rt_filter, "delta_rating"] = d_score - i_score
            data.loc[mem_filter & rt_filter, "success"] = np.max(
                [0.0, i_score - d_score, ]
            )

    # Save out difference scores
    data.sort_values(
        [
            "subject",
            "session",
            "run",
            "period",
        ]
    ).to_csv(args.output_path / f"conte_one_ratings_by_trial.csv", index=False)

    # Save out a smaller summary of ratings by memory.
    # First, catalog the problematic memories we should just skip.
    mem_scores = [
        "distance_rating", "immerse_rating", "success",
    ]
    unique_scores = data.groupby(["subject", "memory", "rating_type", ])[
        mem_scores
    ].nunique()
    problems = unique_scores[
        (unique_scores['distance_rating'] != 1) |
        (unique_scores['immerse_rating'] != 1)
    ].reset_index(drop=False)
    problem_list = [
        f"{row['subject']} - {row['memory']}"
        for idx, row in problems.iterrows()
    ]
    problems.to_csv(
        args.output_path / f"conte_one_problematic_memories.csv", index=False
    )

    mem_dfs = []
    for mem_id in data["mem_id"].unique():
        if mem_id not in problem_list:
            mem_df = (
                data[data["mem_id"] == mem_id]
                .groupby(["subject", "memory", "rating_type"])[
                    ["subject", "memory", "rating_type"] + mem_scores
                ]
                .mean(numeric_only=True)
                .reset_index(drop=False)
            )
            mem_dfs.append(
                mem_df[["subject", "memory", "rating_type"] + mem_scores]
            )
        else:
            print(f"Problem with '{mem_id}', so it won't be included.")
    mem_data = pd.concat(mem_dfs).sort_values(
        ["subject", "memory", "rating_type"]
    )
    mem_data = mem_data.reset_index(drop=True)

    mem_data.to_csv(
        args.output_path / f"conte_one_ratings_by_memory.csv", index=False
    )

    # Save out a smaller summary of ratings by subject.
    subject_cols = ["subject", "rating_type"] + mem_scores
    subject_data = mem_data.groupby(
        ["subject", "rating_type", ]
    )[subject_cols].mean(
        numeric_only=True
    )
    subject_data = subject_data[mem_scores]
    subject_data.sort_values(["subject", "rating_type", ]).to_csv(
        args.output_path / f"conte_one_mem_ratings_means_by_subject.csv",
        index=True
    )


if __name__ == "__main__":
    main(validate_args(get_env(get_arguments())))
