#!/usr/bin/env python3

# sub_pipeline.py

import sys
import os
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
        help="The space for qsiprep output, should stay with T1w to match anat.",
    )
    parser.add_argument(
        "--template", default="MNI152NLin2009cAsym",
        help="The intended template for qsiprep",
    )
    parser.add_argument(
        "--recon-spec", default="dsi_studio_gqi",
        help="The processing pipeline for diffusion, after pre-processing is complete",
    )
    parser.add_argument(
        "--hmc-model", default="eddy",
        help="The diffusion correction algorithm, 'eddy' or '3dSHORE'",
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
        "/tmp", "_".join([args.pipeline, args.timestamp, "templateflow", ])
    ))
    if getattr(args, "slurm_partition") is None:
        setattr(args, "slurm_partition", args.pipeline)

    return args


def get_singularity_image(args):
    """ Return the path to a singularity file, if it exists, else None """

    sif_path = "/var/lib/singularity/images"
    sif_file = None

    if args.pipeline_version is None:
        if args.verbose:
            print("No version specified for {} pipeline.".format(args.pipeline))
        possibilities = pathlib.Path(sif_path).glob("{}-*.sif".format(args.pipeline))
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
    with open(test_w_file, "w") as f:
        f.write("Just a test, this file can be deleted.")
    work_writable = os.path.exists(test_w_file)
    os.remove(test_w_file)

    # Staging directory is writable
    os.makedirs(args.staging_directory, exist_ok=True)
    test_s_file = os.path.join(args.staging_directory, "test.file")
    with open(test_s_file, "w") as f:
        f.write("Just a test, this file can be deleted.")
    stage_writable = os.path.exists(test_s_file)
    os.remove(test_s_file)

    print( "'context_is_valid' returning",
        sif_available and
        sub_available and
        work_writable and
        stage_writable
    )
    return sif_available and sub_available and work_writable and stage_writable


def write_batch_script(args):
    """ Create the batch script to batch the pipeline execution. """

    os.makedirs(os.path.join(args.home, "bin", "slurm"), exist_ok=True)
    script_name = "_".join([
        "batch", "sub-" + args.subject, args.pipeline, args.timestamp,
    ]) + ".slurm"
    batch_file = pathlib.Path(args.home, "bin", "slurm", script_name)
    setattr(args, "job_name", "_".join([args.username, args.pipeline, args.subject, ]))
    setattr(args, "log_name", "_".join([
        "sub-" + args.subject,
        "pipeline-" + args.pipeline,
        "ts-" + args.timestamp,
    ]))
    with open(batch_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name={}\n".format(args.job_name))
        f.write("#SBATCH --partition={}\n".format(args.slurm_partition))
        f.write("#SBATCH --output={}\n".format(
            os.path.join(args.staging_directory, args.log_name + ".out")
        ))
        f.write("#SBATCH --error={}\n".format(
            os.path.join(args.staging_directory, args.log_name + ".err")
        ))
        f.write("#SBATCH --time=0\n")
        f.write("#SBATCH --mem=32768\n")
        f.write("#SBATCH --chdir=/tmp/\n")
        f.write("#SBATCH --export=TEMPLATEFLOW_HOME=/opt/templateflow\n")
        # TODO: Ensure a unique timestamped work and staging directory.
        # TODO: And remove anything in the way.
        f.write("mkdir -p {}\n".format(args.work_directory))
        f.write("mkdir -p {}\n".format(args.templateflow_directory))
        f.write("mkdir -p {}\n".format(args.staging_directory))
        f.write("singularity run --cleanenv \\\n")
        f.write("  -B {}:/data:ro \\\n".format(args.rawdata))
        f.write("  -B {}:/out \\\n".format(args.staging_directory))
        f.write("  -B {}:/work \\\n".format(args.work_directory))
        f.write("  -B {}:/opt/templateflow \\\n".format(args.templateflow_directory))
        f.write("  -B {}:/opt/freesurfer/license.txt:ro \\\n".format(
            args.freesurfer_license
        ))
        f.write("  {} \\\n".format(args.sif_file))
        f.write("  /data /out participant \\\n")
        f.write("  -w /work \\\n")
        f.write("  --fs-license-file /opt/freesurfer/license.txt \\\n")
        f.write("  --participant-label {} \\\n".format(args.subject))
        if args.pipeline == "qsiprep":
            f.write("  --output-resolution {} \\\n".format(args.output_resolution))
            f.write("  --hmc-model {} \\\n".format(args.hmc_model))
            f.write("  --output-space T1w \\\n".format(args.output_space))
            f.write("  --template {} \\\n".format(args.template))
            f.write("  --recon-spec {} \\\n".format(args.recon_spec))
        f.write("  --nthreads 16 --mem-mb 32768\n")
        f.write("\n")
        f.write("chown {}:{} --recursive {}\n".format(
            args.username, args.userid, args.staging_directory
        ))
        f.write("rsync -ah {}/{}/sub-{}* {}/{}/\n".format(
            args.staging_directory, args.pipeline, args.subject,
            args.derivatives, args.pipeline
        ))
        if args.pipeline == "fmriprep":
            f.write("if [[ ! -e {}/freesurfer/sub-{} ]]; then\n".format(
                args.derivatives, args.subject,
            ))
            f.write("  rsync -ah {}/freesurfer/sub-{}* {}/freesurfer/\n".format(
                args.staging_directory, args.subject, args.derivatives,
            ))
            f.write("else\n")
            f.write("  echo \"Freesurfer output at {} was abandoned to avoid over-writing {}/freesurfer/sub-{}".format(
                args.staging_directory, args.derivatives, args.subject,
            ))
            f.write("fi\n")
        elif args.pipeline == "qsiprep":
            f.write("rsync -ah {}/qsirecon/sub-{}* {}/qsirecon/\n".format(
                args.staging_directory, args.subject, args.derivatives,
            ))
        f.write("echo \"Job complete at $(date)\"")

    return batch_file


def submit_batch_script(sbatch_file):
    """ Create the batch script to batch the pipeline execution. """
    print("|==========---------- Batch file follows ----------==========|")
    subprocess.run(["cat", sbatch_file, ])
    print("|==========----------        end         ----------==========|")
    print("Batch file was saved to {}".format(sbatch_file))
    print("Submitting job at {}".format(datetime.now().strftime("%Y%m%d %H:%M:%S")))
    subprocess.run(["sbatch", sbatch_file, ])



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
        submit_batch_script(batch_script)


if __name__ == "__main__":
    """ Parse arguments and pass them to main. """
    main(get_arguments())
