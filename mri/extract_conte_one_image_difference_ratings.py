#!/usr/bin/env python3

# template.py

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import re


# Some globals


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Describe the point of this script.",
    )
    parser.add_argument(
        "input_path",
        help="a required path or a file or whatever",
    )
    parser.add_argument(
        "output_path",
        help="The output path for writing",
    )
    parser.add_argument(
        "--participant", default="all",
        help="The subject/participant to work with",
    )
    parser.add_argument(
        "--exclude", nargs="+",
        help="Subjects to avoid looking through",
    )
    parser.add_argument(
        "-o", "--option",
        help="change some other optional option",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run all tasks, even if it means overwriting existing data",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run no tasks, just report on what would be run without this flag",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    return args


def get_env(args):
    """ Integrate environment variables into our args. """

    return args


def validate_args(args):
    """ Ensure the environment will support the requested workflows. """

    errors = []

    if args.participant.startswith("sub-"):
        setattr(args, "participant", args.participant[4:])

    setattr(args, "input_path", Path(args.input_path).resolve())
    if not args.input_path.exists():
        errors.append(
            f"ERR: The input path '{str(args.input_path)}' doesn't exist."
        )

    setattr(args, "output_path", Path(args.output_path).resolve())
    if not args.output_path.exists():
        errors.append(
            f"ERR: The output path '{str(args.output_path)}' doesn't exist."
        )

    if args.exclude is None:
        setattr(args, "exclude", [])

    if len(errors) > 0:
        for error in errors:
            print(error)
            sys.exit(1)

    return args


def combo_score(df, subject_id, image_id, affect):
    """ Return a success score for this subject's reappraisal of this image.
    """

    subject_filter = df['subject_id'] == subject_id
    image_filter = df['image'] == image_id
    reap_filter = df['mode'] == f"Reapp{affect}"

    rating = f"{affect.lower()}_rating"

    # See if this subject reappraised this image
    reappraise_trial = df.loc[subject_filter & image_filter & reap_filter]
    print(f"{len(reappraise_trial)} {affect} reappraisals "
          f"of image {image_id} for {subject_id}")
    if len(reappraise_trial) != 0:
        return None

    # Exactly one trial with correct reappraisal, calculate success score
    look_filter = df['mode'] == f"Look{affect}"
    look_trials = df.loc[look_filter, rating]
    grand_look_mean = look_trials.mean()
    grand_look_std = look_trials.std()
    image_look_mean = df.loc[look_filter & image_filter, rating].mean()
    subject_look_mean = df.loc[look_filter & subject_filter, rating].mean()
    image_adj = (grand_look_mean - image_look_mean) / grand_look_std
    subject_adj = (grand_look_mean - subject_look_mean) / grand_look_std
    predicted_look_rating = grand_look_mean + image_adj + subject_adj
    success_score = reappraise_trial.iloc[0][rating] - predicted_look_rating
    print(f"Sub {subject_id} {affect}-reappraised {image_id} "
          f"as a {reappraise_trial.iloc[0][rating]}; compared to "
          f"grand mean of {grand_look_mean:0.2f} +- {grand_look_std:0.2f}, "
          f"image mean of {image_look_mean:0.2f}, "
          f"subject mean of {subject_look_mean:0.2f}, "
          f"and a predicted look score of {predicted_look_rating:0.2f}. "
          f"We score reappraisal success as {success_score}")
    return max(0.00, success_score)


def build_combo_scores(df):
    """ Fill in success scores for all reappraisals.
    """

    for affect in ["Pos", "Neg"]:
        rating = f"{affect.lower()}_rating"
        reap_filter = df['mode'] == f"Reapp{affect}"
        look_filter = df['mode'] == f"Look{affect}"
        look_trials = df.loc[look_filter, rating]
        grand_look_mean = look_trials.mean()
        grand_look_std = look_trials.std()

        for image_id in sorted(df['image'].unique()):
            image_filter = df['image'] == image_id
            image_look_mean = df.loc[look_filter & image_filter,
                                     rating].mean()
            image_adj = (image_look_mean - grand_look_mean) / grand_look_std

            for subject_id in sorted(df['subject_id'].unique()):
                subject_filter = df['subject_id'] == subject_id
                subject_look_mean = df.loc[look_filter & subject_filter,
                                           rating].mean()
                subject_adj = ((subject_look_mean - grand_look_mean)
                               / grand_look_std)

                # See if this subject reappraised this image
                reappraise_trial = df.loc[
                    subject_filter & image_filter & reap_filter
                ]
                # print(f"{len(reappraise_trial)} {affect} reappraisals "
                #       f"of image {image_id} for {subject_id}")
                if len(reappraise_trial) != 1:
                    continue

                # For one trial with reappraisal, calculate success score
                # ReappPos should decrease neg_rating and increase pos_rating
                # ReappNeg should decrease neg_rating and increase pos_rating
                pred_look_rating = grand_look_mean + image_adj + subject_adj
                actual_reapp_rating = reappraise_trial.iloc[0][rating]
                if affect == "Pos":
                    success_score = actual_reapp_rating - pred_look_rating
                elif affect == "Neg":
                    success_score = pred_look_rating - actual_reapp_rating
                else:
                    # There is no else, this should never happen
                    continue
                # print(
                #     f"Sub {subject_id} {affect}-reappraised {image_id} as a "
                #     f"{reappraise_trial.iloc[0][rating]}; compared to grand "
                #     f"mean of {grand_look_mean:0.2f} +- {grand_look_std:0.2f}"
                #     f",  image mean of {image_look_mean:0.2f}, "
                #     f"subject mean of {subject_look_mean:0.2f}, and a "
                #     f"predicted look score of {pred_look_rating:0.2f}. "
                #     f"We score reappraisal success as {success_score}"
                # )

                df.loc[
                    subject_filter & image_filter & reap_filter,
                    'reapp_success_score'
                ] = max(0.00, success_score)


def extract_from_subject(subject_path):
    """ Do the work on a participant.
    """

    trials = []
    instructions = [
        'ReappPos', 'ReappNeg', 'LookPos', 'LookNeu', 'LookNeg',
    ]
    pattern = re.compile(
        r"sub-([A-Z][0-9]+)_ses-([0-9]+)_task-(image)_run-([0-9]+)_events.tsv"
    )
    for events_file in sorted(subject_path.glob(
            "ses-*/func/sub-*_ses-*_task-image*events.tsv"
    )):
        subject_id, session_id, task, run = "NA", "NA", "NA", "NA"
        match = pattern.match(events_file.name)
        if match:
            subject_id = match.group(1)
            session_id = match.group(2)
            task = match.group(3)
            run = match.group(4)
        else:
            print(f"ERR: No RE match for '{events_file.name}'")
        print(f" {subject_path.name}/{subject_id} {session_id:>6}: "
              f"{task:<6} {run:>3} - {str(events_file)}")
        run_df = pd.read_csv(events_file, sep='\t', header=0, index_col=None)

        # Group events into trials, and store them
        mode, image, pos_rating, neg_rating = None, None, None, None
        pos_rt, neg_rt = None, None
        events_in_trial = 0
        for idx, row in run_df.sort_values('onset').iterrows():
            events_in_trial += 1
            if row['trial_type'] in instructions:
                if row['stimulus'] in ['INCREASE POSITIVE', 'LOOK', ]:
                    mode = row['trial_type']
                elif row['stimulus'].endswith(".bmp"):
                    image = row['stimulus']
            elif row['trial_type'] == 'rating':
                if row['stimulus'] == 'rp':
                    pos_rating = row['response']
                    pos_rt = row['response_time']
                elif row['stimulus'] == 'rn':
                    neg_rating = row['response']
                    neg_rt = row['response_time']
            elif row['trial_type'] == 'iti':
                trials.append({
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "task": task,
                    "run": run,
                    "mode": mode,
                    "image": image,
                    "pos_rating": pos_rating,
                    "pos_rt": pos_rt,
                    "neg_rating": neg_rating,
                    "neg_rt": neg_rt,
                    "events_in_trial": events_in_trial,
                })
                mode, image, pos_rating, neg_rating = None, None, None, None
                pos_rt, neg_rt = None, None
                events_in_trial = 0

    return trials


def any_pos_rating(row):
    for rating in [('pos_rating', 'LookPos'),
                   ('pos_rating', 'LookNeg'),
                   ('pos_rating', 'LookNeu')]:
        if np.isfinite(row[rating]):
            return row[rating]
    return np.nan


def any_neg_rating(row):
    for rating in [('neg_rating', 'LookPos'),
                   ('neg_rating', 'LookNeg'),
                   ('neg_rating', 'LookNeu')]:
        if np.isfinite(row[rating]):
            return row[rating]
    return np.nan


def main(args):
    """ Entry point """

    # Discover which participants to run
    if args.participant == "all":
        subject_dirs = [
            d for d in args.input_path.glob("sub-[A-Z][0-9]*")
            if re.match(r'sub-[A-Z][0-9]+', d.name)
        ]
    else:
        subject_dirs = [
            args.input_path / f"sub-{args.participant}"
        ]

    # Run them
    trials = []
    subject_count = 0
    for subject_dir in subject_dirs:
        exclude = False
        for exclusion in args.exclude:
            if exclusion in subject_dir.name:
                print(f"excluding '{exclusion}'=='{subject_dir.name}'")
                exclude = True
        if not exclude:
            trials = trials + extract_from_subject(subject_dir)
            subject_count += 1

    # Save out results
    print(
        f"Combining {len(trials)} records from {subject_count} "
        f"participants and writing to "
        f"{str(args.output_path / 'conte_one_ratings_by_trial.csv')}"
    )
    data = pd.DataFrame(trials)

    # As of 5/19/2023, with 58 subjects, data have 5220 trials/rows

    # Break down by image
    # Calculate the mean Look rating for each image
    # For each Reapp trial, calculate 'trial rating' - 'image mean look rating'
    # (With 58 subjects, ~30 Looks and ~30 Reapps for each of 30 neg/pos images)
    look_filter = data['mode'].str.startswith('Look')
    for image in data['image'].unique():
        look_image_filter = look_filter & (data['image'] == image)

        for rating in ['pos_rating', 'neg_rating', ]:
            # We only need to filter on image because each image is pos or neg
            look_mean = data[look_image_filter][rating].mean()
            data.loc[
                data['image'] == image, f"mean_{rating}_by_image"
            ] = look_mean
            data.loc[
                data['image'] == image, f"delta_{rating}_vs_image_mean"
            ] = data.loc[data['image'] == image, rating] - look_mean

    for sid in data['subject_id'].unique():
        subject_filter = data['subject_id'] == sid
        for rating in ['pos_rating', 'neg_rating']:
            # Subjects look at pos/neg/neu images, so we must filter multiple
            for trial_type in data['mode'].unique():
                final_filter = subject_filter & (data['mode'] == trial_type)
                mean_rating = data[final_filter][rating].mean()
                data.loc[final_filter,
                         f"mean_{trial_type.lower()}_by_subject"] = mean_rating

    # Calculate the scores we'll use to train the new ER decoder
    data['neg_reapp_img_delta'] = data['delta_neg_rating_vs_image_mean'].apply(
        lambda x: max(0.0, -1.0 * x)
    )
    data['pos_reapp_img_delta'] = data['delta_pos_rating_vs_image_mean'].apply(
        lambda x: max(0.0, x)
    )
    build_combo_scores(data)

    # Save out difference scores
    data.to_csv(
        args.output_path / f"conte_one_ratings_by_trial.csv",
        index=False
    )

    # Save out a smaller summary of ratings by image.
    image_dataframes = []
    image_scores = ["image", "mode", "pos_rating", "neg_rating"]
    for mode in data['mode'].unique():
        img_df = data[data['mode'] == mode].groupby("image")[image_scores].mean(
            numeric_only=True
        )
        pos_label = f"{mode[:-3].lower()}_pos_rating"
        neg_label = f"{mode[:-3].lower()}_neg_rating"
        # img_df['instruct'] = mode[:-3].lower()
        img_df['affect'] = mode[-3:].lower()
        img_df[pos_label] = img_df["pos_rating"]
        img_df[neg_label] = img_df["neg_rating"]
        image_dataframes.append(img_df[['affect', pos_label, neg_label]])
    image_data = pd.concat(image_dataframes).sort_values(['image', 'affect'])
    image_data = image_data.groupby(['image', 'affect']).mean(numeric_only=True)
    # image_data['affect'] = [idx[1] for idx in image_data.index]
    # image_data['image'] = [idx[0] for idx in image_data.index]
    image_data = image_data.reset_index().sort_values(['affect', 'image'])

    image_data.to_csv(
        args.output_path / f"conte_one_ratings_means_by_image.csv",
        index=False
    )

    # Save out a smaller summary of ratings by subject.
    subject_scores = [col for col in data.columns if col.endswith("_subject")]
    subject_cols = ["subject_id", ] + subject_scores
    subject_data = data.groupby("subject_id")[subject_cols].mean(
        numeric_only=True
    )

    subject_data.to_csv(
        args.output_path / f"conte_one_ratings_means_by_subject.csv",
        index=True
    )


if __name__ == "__main__":
    main(validate_args(get_env(get_arguments())))
