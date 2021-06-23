# sub_pipeline.py

## Quick start

Executing this without arguments will give you brief usage instructions. Executing it with the -h flag will give you more explicit instructions.

It will take arguments first from a json file in your home directory, ~/.pipelines.json, and then overwrite them with arguments supplied at the command line.

For one example, your .pipelines.json file could look like this:

    {
      "fmriprep": {
        "freesurfer_license": "/data/export/home/mikes/freesurfer/license.txt"
      }
    }

If you submitted a pipeline like this:

    sub_pipeline.py fmriprep U00001

It would use the freesurfer license from your .pipelines.json file. But if you either didn't have that specified in the file or if you wanted to override it, you could submit like this:

    sub_pipeline.py fmriprep U00001 --freesurfer-license /opt/freesurfer/license.txt

So, if you plan to submit a bunch of jobs with the same settings, save them to your ~/.pipelines.json file and let sub_pipeline.py read them from there. If you need to change them up, overwrite them on the command line. Set --verbose to get a full description of everything it finds and what it assumes along the way.

## What it does

sub_pipeline.py collects the two required arguments, pipeline and subject, along with optional arguments, ~/.pipeline.json fields, and what it can figure out from the environment, and writes a unique detailed batch file for this particular pipeline execution. It then submits that batch file to SLURM. This allows for two layers of execution when SLURM gets to it. The first layer can set up the appropriate context on whichever node is selected for execution. The second layer actually executes the container with the pipeline, which runs fastest on the local disk as root. And the first layer can then clean up after it, change file permissions, and move the output to the desired location. All of this was written to the batch file that is saved with the logs for future reference.

All of this is completely transparent. Read the python code at /usr/local/bin/sub_pipeline to see what it's doing. Read each individual batch file it creates in ~/bin/slurm/. And share any problems with Mike.

