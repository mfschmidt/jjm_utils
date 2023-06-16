#!/usr/bin/env python3

# correlate_gordon_networks_from_xcpd.py

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# Some globals

# These codes allow for turning color on and off
GREEN_ON = "\033[1;32m"
RED_ON = "\033[0;31m"
COLOR_OFF = "\033[0m"


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Build network-specific correlation matrices from xcp_d.",
    )
    parser.add_argument(
        "participant",
        help="The subject/participant to work with (with or without sub-) "
             "If this is 'aggregate', this script will combine all existing "
             "data rather than redo the correlations for a subject",
    )
    parser.add_argument(
        "input_path",
        help="The path where xcp_d saved all of its subject directories",
    )
    parser.add_argument(
        "output_path",
        help="The path to save all of your subject directories",
    )
    parser.add_argument(
        "--space", default="MNI152NLin2009cAsym",
        help="Select a space, must be available in xcp_d outputs",
    )
    parser.add_argument(
        "--atlas", default="Gordon",
        help="Select an atlas, must be available in xcp_d outputs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run all tasks, even if it means overwriting existing data",
    )
    parser.add_argument(
        "--dry-run", "--dry_run", action="store_true",
        help="Run no tasks, just report on what would be run without this flag",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    args = parser.parse_args()

    args = get_env(args)
    args = validate_args(args)

    return args


def get_env(args):
    """ Integrate environment variables into our args. """

    return args


def validate_args(args):
    """ Ensure the environment will support the requested workflows. """

    setattr(args, "input_path", Path(args.input_path))
    setattr(args, "output_path", Path(args.output_path))

    if args.participant.startswith("sub-"):
        setattr(args, "participant", args.participant[4:])
    setattr(args, "participant_path",
            args.input_path / f"sub-{args.participant}")

    if args.verbose:
        if args.input_path.exists():
            print(f"Input path '{args.input_path}' exists.")
        else:
            print(f"Input path '{args.input_path}' does not exist.")
        if args.output_path.exists():
            print(f"Output path '{args.output_path}' exists.")
        else:
            print(f"Output path '{args.output_path}' does not exist.")

    return args


def get_available_timeseries(args):
    """ Find appropriately named timeseries files in participant path """

    rs_ts_files = []
    for ses_dir in args.participant_path.glob("ses-*"):
        print(f"found session dir {ses_dir.name}")
        for rs_ts_file in (ses_dir / "func").glob(
            "sub-{}_{}_task-rest*space-{}_atlas-{}_timeseries.tsv".format(
                args.participant, ses_dir.name, args.space, args.atlas
            )
        ):
            rs_ts_files.append(rs_ts_file)

    return rs_ts_files


def prepare_rs_ts(files):
    """ Concatenate multiple files together, if necessary """

    dataframes = [
        pd.read_csv(f, sep='\t', index_col=False) for f in files
    ]

    if len(dataframes) == 0:
        return dataframes[0]
    else:
        return pd.concat(dataframes, axis=0)


def tabulate_matrix_by_network(matrix_df, participant):
    """ From a correlation matrix with labels, save long data by network.
    """

    # To preserve high correlations, that would cause arctanh to divide
    # by zero, we cap them at almost 1. Now they will all remain in the data.
    capped_df = matrix_df.copy()
    capped_df[capped_df > 0.99999] = 0.99999
    capped_df[capped_df < -0.99999] = -0.99999
    z_df = np.arctanh(capped_df)
    if participant.startswith("U"):
        site = "NYSPI"
    elif participant.startswith("P"):
        site = "UPMC"
    else:
        site = "N/A"
    stored_observations = set()
    observations = []
    for src in matrix_df.columns:
        src_hemi, src_network, src_num = src.split("_")
        for tgt in matrix_df.columns:
            tgt_hemi, tgt_network, tgt_num = tgt.split("_")
            observation = {
                'site': site,
                'participant': participant,
                # 'src': src,
                'src_hemi': src_hemi,
                'src_network': src_network,
                'src_num': src_num,
                # 'tgt': tgt,
                'tgt_hemi': tgt_hemi,
                'tgt_network': tgt_network,
                'tgt_num': tgt_num,
                'intra_network': int(src_network == tgt_network),
                'intra_hemi': int(src_hemi == tgt_hemi),
                'r': f"{matrix_df.loc[src, tgt]:0.5f}",
                'z': f"{z_df.loc[src, tgt]:0.5f}",
            }
            # We don't need the diagonal self-correlations
            # We only need one r value per pair, not both
            # And nan values do us no good at all, ignore them.
            if (
                    (src != tgt)
                    and ((src, tgt) not in stored_observations)
                    and np.isfinite(z_df.loc[src, tgt])
            ):
                # Remember this src/tgt pair in both directions
                stored_observations.add((src, tgt))
                stored_observations.add((tgt, src))
                observations.append(observation)

    return pd.DataFrame(observations)


def build_subject_correlations(args):
    """ For one subject, calculate correlations and save them
    """
    if args.verbose:
        print("Looking in {} for {} atlas timeseries in {} space.".format(
            args.participant_path, args.atlas, args.space
        ))

    # Find available files
    rs_ts_files = get_available_timeseries(args)
    if args.verbose:
        print(f"Found {len(rs_ts_files)} timeseries:")
        for file in sorted(rs_ts_files):
            print(f"  {file.name}")
    if len(rs_ts_files) == 0:
        # No files, there's nothing more to do.
        return

    # Concatenate files if necessary
    ts_data = prepare_rs_ts(rs_ts_files)
    if args.verbose:
        print(f"Complete timeseries data shaped {ts_data.shape}.")

    # Write out final timeseries
    (args.output_path / f"sub-{args.participant}").mkdir(
        parents=True, exist_ok=True,
    )
    ts_file_name = "sub-{}_task-rest_space-{}_atlas-{}_timeseries.tsv".format(
        args.participant, args.space, args.atlas
    )
    ts_data.to_csv(
        args.output_path / f"sub-{args.participant}" / ts_file_name,
        sep='\t',
    )

    # Correlate each column with each other
    corr_matrix = np.corrcoef(ts_data.values, rowvar=False)
    if args.verbose:
        print(f"Complete correlation matrix shaped {corr_matrix.shape}.")

    # Write out final correlation matrix
    corr_file_name = "sub-{}_task-rest_space-{}_atlas-{}_conmat.tsv".format(
        args.participant, args.space, args.atlas
    )
    corr_df = pd.DataFrame(
        data=corr_matrix, columns=ts_data.columns, index=ts_data.columns
    )
    corr_df.to_csv(
        args.output_path / f"sub-{args.participant}" / corr_file_name,
        sep='\t'
    )

    # Reformat the correlations to account for networks and inter/intra
    final_df = tabulate_matrix_by_network(corr_df, args.participant)
    if args.verbose:
        print(f"Final results shaped {final_df.shape}.")

    # Write out final correlation matrix
    final_file_name = "sub-{}_task-rest_space-{}_atlas-{}_results.tsv".format(
        args.participant, args.space, args.atlas
    )
    final_df.to_csv(
        args.output_path / f"sub-{args.participant}" / final_file_name,
        sep='\t', index=False
    )


def aggregate_data(args):
    """ Loop over all participants, concatenating all data
    """

    dataframes = []
    for p in args.output_path.glob("sub-*"):
        f = "{}_task-rest_space-{}_atlas-{}_results.tsv".format(
            p.name, args.space, args.atlas
        )
        if (p / f).exists():
            dataframes.append(pd.read_csv(p / f, sep='\t', index_col=False))
        else:
            print(f"  WARN: cannot find {f} in {str(p)}")

    combo_data = pd.concat(dataframes)
    if args.verbose:
        print("Combined {} files into one with {} rows.".format(
            len(dataframes), len(combo_data)
        ))
    combo_file = "combined_task-rest_space-{}_atlas-{}_results.tsv".format(
        args.space, args.atlas
    )
    combo_data.to_csv(
        args.output_path / combo_file, sep='\t', index=False
    )


def main(args):
    """ Entry point """

    if args.participant == 'aggregate':
        aggregate_data(args)
    else:
        build_subject_correlations(args)


if __name__ == "__main__":
    main(get_arguments())
