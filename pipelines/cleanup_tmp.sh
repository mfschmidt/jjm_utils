#!/bin/bash


declare -a ACTIVE_JOBS
N_ACTIVE=0

JOBS=$(squeue | awk '{print $1}')
for JOB in $JOBS; do
  if [[ "$JOB" != "JOBID" ]]; then
    HOST=$(scontrol show job ${JOB} | grep BatchHost)
    HOST=${HOST##*=}
    CMD=$(scontrol show job ${JOB} | grep Command)
    CMD=${CMD##*=}
    TS=${CMD%%.*}
    TS=${TS##*_}
    STAGING=$(ls -1d /tmp/*prep_${TS}_staging)
    WORKING=$(ls -1d /tmp/*prep_${TS}_working)    
    if [[ "$HOST" == "$(hostname)" ]]; then
    	echo "Host $HOST, Job $JOB, timestamp $TS:"
    	echo "  C: $CMD"
    	echo "  S: $STAGING"
    	echo "  W: $WORKING"
    	ACTIVE_JOBS[$TS]=$TS
    	N_ACTIVE=$(( N_ACTIVE + 1 ))
    fi
  fi
done

echo "Found $N_ACTIVE active jobs on host $(hostname)."

ALL_TMP=$(ls -1d /tmp/*prep_*_*ing)
REGEX="_([0-9*])_"
for TMP in ALL_TMP; do
  if [[ $TMP =~ $REGEX ]]; then
    TMP_TS="${BASH_REMATCH[1]}"
    if [[ "${ARR[$TMP_TS]}" == "" ]]; then
      echo "  OK to delete $TMP, inactive."
    else
      echo "  SAVE $TMP"
    fi
  fi
done
