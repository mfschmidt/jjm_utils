#!/usr/bin/env python3

# sub_pipeline.py

import sys
import os
import platform
import pwd
import pathlib
import argparse
import json
import subprocess

from datetime import datetime


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Submit a Singularity pipeline request to SLURM.",
    )
    parser.add_argument(
        "pipeline",
        help="The name of a pipeline to execute",
    )
    parser.add_argument(
        "subject",
        help="The name of a subject to be processed",
    )
    parser.add_argument(
        "--home",
        help="Submitting user's home path",
    )
    parser.add_argument(
        "--userid",
        help="Submitting user's numeric user id",
    )
    parser.add_argument(
        "--username",
        help="Submitting user's user name",
    )
    parser.add_argument(
        "--rawdata",
        help="The path to get bids-formatted raw data",
    )
    parser.add_argument(
        "--derivatives",
        help="The path to write pipeline derivative output",
    )
    parser.add_argument(
        "--log-file",
        help="The path to a file for logging details",
    )
    parser.add_argument(
        "--work-directory",
        help="The path to read and write temporary files",
    )
    parser.add_argument(
        "--pipeline-version",
        help="The version of the docker/singularity pipeline to run",
    )
    parser.add_argument(
        "--slurm-partition",
        help="The slurm partition for this job",
    )
    parser.add_argument(
        "--freesurfer-license",
        help="The location of a valid FreeSurfer license file",
    )
    parser.add_argument(
        "--output-resolution", default="2.0",
        help="The isotropic resolution in mm for qsiprep output",
    )
    parser.add_argument(
        "--output-space", default="T1w",
        help="Anatomical space for qsiprep output, should stay with T1w.",
    )
    parser.add_argument(
        "--template", default="MNI152NLin2009cAsym",
        help="The intended template for qsiprep",
    )
    parser.add_argument(
        "--recon-spec", default=None,
        help="The pipeline for diffusion, after pre-processing is complete",
    )
    parser.add_argument(
        "--hmc-model", default="eddy",
        help="The diffusion correction algorithm, 'eddy' or '3dSHORE'",
    )
    parser.add_argument(
        "--num-cpus", default="4",
        help="The number of processors to allocate for the pipeline.",
    )
    parser.add_argument(
        "--memory-mb", default="24576",
        help="The amount of memory to allocate for the pipeline.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    # Determine expected values for variables not provided or overridden
    if "--userid" not in sys.argv:
        sys.argv.append("--userid")
        sys.argv.append(str(os.getuid()))
    if "--username" not in sys.argv:
        sys.argv.append("--username")
        sys.argv.append(pwd.getpwuid(os.getuid())[0])
    if "--home" not in sys.argv:
        sys.argv.append("--home")
        sys.argv.append(os.path.expanduser("~"))

    args = parser.parse_args()
    setattr(args, "timestamp", datetime.now().strftime("%Y%m%d%H%M%S"))

    # Fill in missing arguments with user settings in home directory
    user_settings_file = os.path.join(args.home, ".pipelines.json")
    if os.path.isfile(user_settings_file):
        if args.verbose:
            print("Found user settings at {}".format(user_settings_file))
        user_settings = json.load(open(user_settings_file, "r"))
        if args.pipeline in user_settings:
            if args.verbose:
                print("  {} in settings".format(args.pipeline))
            for key in user_settings[args.pipeline]:
                if getattr(args, key.replace("-", "_")) is None:
                    setattr(args, key, user_settings[args.pipeline][key])
                    if args.verbose:
                        print("  using user settings: {} := {}".format(
                            key, user_settings[args.pipeline][key]
                        ))
                else:
                    if args.verbose:
                        print("  args override user settings: {} := {}".format(
                            key, user_settings[args.pipeline][key]
                        ))

    # Generate final defaults, if they aren't provided.
    if getattr(args, "log_file") is None:
        setattr(args, "log_file", os.path.join(
            args.home, "_".join([
                "sub-" + args.subject,
                "pipeline-" + args.pipeline,
                "ts-" + args.timestamp,
            ]) + ".log"
        ))
    if getattr(args, "work_directory") is None:
        setattr(args, "work_directory", os.path.join(
            "/tmp", "_".join([args.pipeline, args.timestamp, "working", ])
        ))
    setattr(args, "staging_directory", os.path.join(
        "/tmp", "_".join([args.pipeline, args.timestamp, "staging", ])
    ))
    setattr(args, "templateflow_directory", os.path.join(
        "/tmp", "templateflow"
    ))
    if getattr(args, "slurm_partition") is None:
        setattr(args, "slurm_partition", args.pipeline)

    return args


def get_singularity_image(args):
    """ Return the path to a singularity file, if it exists, else None """

    sif_path = "/var/lib/singularity/images"
    sif_file = None
    possibilities = []

    if args.pipeline_version is None:
        if args.verbose:
            print("No version specified for {} pipeline.".format(
                args.pipeline
            ))
        possibilities = list(pathlib.Path(sif_path).glob(
            "{}-*.sif".format(args.pipeline)
        ))
        if len(possibilities) < 0:
            print("No image exists for {} pipeline on {}.".format(
                args.pipeline, platform.node()
            ))
        elif len(possibilities) == 1:
            sif_file = possibilities[0]
            if args.verbose:
                print("The only {} pipeline is version {}".format(
                    args.pipeline, sif_file[sif_file.rfind("-") + 1:-4]
                ))
        else:
            sif_file = sorted(possibilities, reverse=True)[0]
            print("Using latest {} pipeline, version {}".format(
                args.pipeline, sif_file[sif_file.rfind("-") + 1:-4]
            ))
    else:
        sif_file = pathlib.Path(os.path.join(
            sif_path, "{}-{}.sif".format(args.pipeline, args.pipeline_version)
        ))
        if not os.path.isfile(sif_file):
            print("No image exists at {}.".format(sif_file))
            sif_file = None

    return sif_file


def context_is_valid(args):
    """ Ensure paths are OK and writable before bothering slurm queues. """

    # Pipeline is available.
    setattr(args, "sif_file", get_singularity_image(args))
    sif_available = args.sif_file is not None

    # Subject is available in rawdata
    sub_available = os.path.exists(
        os.path.join(args.rawdata, "sub-" + args.subject)
    )
    if not sub_available:
        print("Cannot find a file at {}".format(
            os.path.join(args.rawdata, "sub-" + args.subject)
        ))

    # Work directory is writable
    os.makedirs(args.work_directory, exist_ok=True)
    test_w_file = os.path.join(args.work_directory, "test.file")
    with open(test_w_file, "w") as test_file:
        test_file.write("Just a test, this file can be deleted.")
    work_writable = os.path.exists(test_w_file)
    os.remove(test_w_file)

    # Staging directory is writable
    os.makedirs(args.staging_directory, exist_ok=True)
    test_s_file = os.path.join(args.staging_directory, "test.file")
    with open(test_s_file, "w") as test_file:
        test_file.write("Just a test, this file can be deleted.")
    stage_writable = os.path.exists(test_s_file)
    os.remove(test_s_file)

    retval = (sif_available and sub_available and
              work_writable and stage_writable)
    print("'context_is_valid' returning {}".format(retval))
    return retval


def write_batch_script(args):
    """ Create the batch script to batch the pipeline execution. """

    os.makedirs(os.path.join(args.home, "bin", "slurm"), exist_ok=True)
    script_name = "_".join([
        "batch", "sub-" + args.subject, args.pipeline, args.timestamp,
    ]) + ".sbatch"
    batch_file_path = pathlib.Path(args.home, "bin", "slurm", script_name)
    log_file_path = str(batch_file_path).replace(".sbatch", ".out")
    with open(batch_file_path, "w") as batch_file:
        batch_file.write("#!/bin/bash\n")
        batch_file.write("#SBATCH --job-name={}\n".format(args.subject))
        batch_file.write("#SBATCH --partition={}\n".format(args.slurm_partition))
        batch_file.write("#SBATCH --output={}\n".format(log_file_path))
        batch_file.write("#SBATCH --error={}\n".format(log_file_path))
        batch_file.write("#SBATCH --time=0\n")
        batch_file.write("#SBATCH --ntasks-per-node={}\n".format(args.num_cpus))
        batch_file.write("#SBATCH --mem={}\n".format(args.memory_mb))
        batch_file.write("#SBATCH --chdir=/tmp/\n")
        batch_file.write("#SBATCH --export={}\n".format(
            "SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow"
        ))
        batch_file.write("\n")
        batch_file.write("mkdir -p {}\n".format(args.work_directory))
        batch_file.write("mkdir -p {}\n".format(args.templateflow_directory))
        batch_file.write("mkdir -p {}\n".format(args.staging_directory))
        batch_file.write("\n")
        batch_file.write("singularity run --cleanenv \\\n")
        batch_file.write("  -B {}:/data:ro \\\n".format(args.rawdata))
        batch_file.write("  -B {}:/out \\\n".format(args.staging_directory))
        batch_file.write("  -B {}:/work \\\n".format(args.work_directory))
        batch_file.write("  -B {}:/opt/templateflow \\\n".format(
            args.templateflow_directory
        ))
        batch_file.write("  -B {}:/opt/freesurfer/license.txt:ro \\\n".format(
            args.freesurfer_license
        ))
        batch_file.write("  {} \\\n".format(args.sif_file))
        batch_file.write("  /data /out participant \\\n")
        batch_file.write("  -w /work \\\n")
        batch_file.write("  --fs-license-file /opt/freesurfer/license.txt \\\n")
        batch_file.write("  --participant-label {} \\\n".format(args.subject))
        if args.pipeline == "qsiprep":
            batch_file.write("  --output-resolution {} \\\n".format(
                args.output_resolution
            ))
            batch_file.write("  --hmc-model {} \\\n".format(args.hmc_model))
            batch_file.write("  --output-space {} \\\n".format(args.output_space))
            batch_file.write("  --template {} \\\n".format(args.template))
            if (args.recon_spec is not None) and (args.recon_spec != "none"):
                batch_file.write("  --recon-spec {} \\\n".format(args.recon_spec))
        batch_file.write("  --nthreads {} --mem-mb {}\n".format(
            int(args.num_cpus) * 2, args.memory_mb
        ))
        batch_file.write("\n")
        batch_file.write("if [[ \"$?\" != \"0\" ]]; then\n")
        batch_file.write("  echo \"Container run failed. Avoiding clean-up.\"\n")
        batch_file.write("  echo \"$(hostname):{}\"\n".format(args.staging_directory))
        batch_file.write("  exit $?\n")
        batch_file.write("fi\n")
        batch_file.write("\n")
        batch_file.write("# Format and copy output to final destination.\n")
        batch_file.write("chown {}:{} --recursive {}\n".format(
            args.username, args.userid, args.staging_directory
        ))
        batch_file.write("rsync -ah {}/{}/sub-{}* {}/{}/\n".format(
            args.staging_directory, args.pipeline, args.subject,
            args.derivatives, args.pipeline
        ))
        if args.pipeline == "fmriprep":
            batch_file.write("if [[ ! -e {}/freesurfer/sub-{} ]]; then\n".format(
                args.derivatives, args.subject,
            ))
            batch_file.write("  rsync -ah {}/freesurfer/sub-{}* {}/freesurfer/\n".format(
                args.staging_directory, args.subject, args.derivatives,
            ))
            batch_file.write("else\n")
            batch_file.write("  echo \"Freesurfer output at {} was abandoned\"\n".format(
                args.staging_directory,
            ))
            batch_file.write("  echo \"to avoid over-writing {}/freesurfer/sub-{}\"\n".format(
                args.derivatives, args.subject,
            ))
            batch_file.write("fi\n")
        elif args.pipeline == "qsiprep":
            batch_file.write("rsync -ah {}/qsirecon/sub-{}* {}/qsirecon/\n".format(
                args.staging_directory, args.subject, args.derivatives,
            ))
        batch_file.write("rm -rf {}\n".format(args.work_directory))
        batch_file.write("# intentionally leaving templateflow dir\n")
        batch_file.write("\n")
        batch_file.write("echo \"Job complete at $(date)\"\n")
        batch_file.write("\n")
        batch_file.write("rsync -ah {}/*.{{err,out}} {}/logs/\n".format(
            args.staging_directory, args.derivatives,
        ))
        batch_file.write("# rm -rf {}\n".format(args.staging_directory))

    return batch_file_path


def submit_batch_script(sbatch_file, verbose=False):
    """ Create the batch script to batch the pipeline execution. """
    if verbose:
        print("|==========---------- Batch file follows ----------==========|")
        subprocess.run(["cat", sbatch_file, ], check=True)
        print("|==========----------        end         ----------==========|")
    print("Batch file was saved to {}".format(sbatch_file))
    print("Submitting job at {}".format(
        datetime.now().strftime("%Y%m%d %H:%M:%S")
    ))
    sbatch_submit_process = subprocess.run(
        ["sbatch", sbatch_file, ], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(sbatch_submit_process.stdout.decode("utf-8"))


def main(args):
    """ Entry point """

    if args.verbose:
        print("Asked to process data from {} with options:".format(
            args.subject
        ))
        for arg in vars(args):
            print("  {}: {}".format(arg, getattr(args, arg)))

    if context_is_valid(args):
        batch_script = write_batch_script(args)
        submit_batch_script(batch_script, args.verbose)


if __name__ == "__main__":
    main(get_arguments())
