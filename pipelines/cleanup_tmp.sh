#!/bin/bash

JOBS=$(squeue | awk '{print $1}')
for JOB in $JOBS; do
  if [[ "$JOB" != "JOBID" ]]; then
    CMD=$(scontrol show job ${JOB} | grep Command)
    CMD=${CMD##*=}
    HOST=$(scontrol show job ${JOB} | grep BatchHost)
    HOST=${HOST##*=}
    if [[ "$HOST" == "$(hostname)" ]]; then
    	echo "Job $JOB ran $CMD on host $HOST"
    fi
  fi
done
