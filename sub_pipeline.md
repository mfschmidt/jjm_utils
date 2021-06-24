# sub_pipeline

## Quick start

This has been installed on every jjm node, so it should work there after doing
the following setup.

To install it to your own machine, just put `sub_pipeline` anywhere in your path.
Also, python3 must be installed and your singularity images should be available
in `/var/lib/singularity/images/`.

### Setup

Download a local config file template and customize it. Following this command,
edit as you see fit.

    wget https://github.com/mfschmidt/jjm_utils/raw/main/.pipelines.json \
    -o ~/.pipelines.json \
    && sed -i 's@HOME@'"${HOME}"'@g' ~/.pipelines.json

### Execution

Executing `sub_pipeline` without arguments will output brief usage instructions.
Executing it with the -h flag will give you more explicit instructions.

It will process arguments in a cascading order:

1. Each command-line argument has a default value.
2. Your local settings file `~/.pipelines.json` overrides the defaults.
3. Explicit command line arguments override everything else.

For one example, your .pipelines.json file could look like this:

    {
      "fmriprep": {
        "freesurfer_license": "/data/export/home/mikes/freesurfer/license.txt"
      }
    }

If you submitted a pipeline like this:

    sub_pipeline fmriprep U00001

It would use the freesurfer license from your .pipelines.json file.
But if you wanted to use a different license, you could either edit your
local config file or you could submit like this:

    sub_pipeline fmriprep U00001 --freesurfer-license ~/license.txt

So, if you plan to submit a bunch of jobs with the same settings,
save them to your ~/.pipelines.json file and let sub_pipeline.py read them from
there. If you need to change them frequently, specify them on the command line.
Set --verbose to see a full description of everything it finds and what it
assumes along the way.

## What it does

`sub_pipeline` collects the two required arguments, pipeline and subject,
along with optional arguments, `~/.pipeline.json` fields, and what it can
figure out from the environment, and writes a unique detailed and
time-stamped batch file for this particular pipeline execution
(written to `~/bin/slurm/`). It then submits that batch file to SLURM.
This allows for two layers of execution when SLURM gets to it.
The first layer can set up the appropriate context on whichever node is
selected for execution. The second layer actually executes the container
with the pipeline, which runs fastest on the local disk, avoiding network I/O.
When the pipeline is complete, layer two returns control to the first layer
which then cleans up after it. It changes file permissions and moves the
node-local output to the desired location anywhere across the network.
Instructions for all of this were written to the batch file that is saved
with the logs for future reference.

All of this is completely transparent. Read the python code at
`/usr/local/bin/sub_pipeline` to see what it's doing.
Read each individual batch file it creates in ~/bin/slurm/.
And share any problems with Mike.
