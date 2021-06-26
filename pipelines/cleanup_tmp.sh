#!/bin/bash

JOBS=$(squeue | awk '{print $1}')
for JOB in $JOBS; do
  if [[ "$JOB" != "JOBID" ]]; then
    HOST=$(scontrol show job ${JOB} | grep BatchHost)
    HOST=${HOST##*=}
    CMD=$(scontrol show job ${JOB} | grep Command)
    CMD=${CMD##*=}
    TS=${CMD%%.*}
    TS=${TS##*_}    
    if [[ "$HOST" == "$(hostname)" ]]; then
    	echo "Job $JOB ran $CMD on host $HOST with timestamp $TS"
    fi
  fi
done
