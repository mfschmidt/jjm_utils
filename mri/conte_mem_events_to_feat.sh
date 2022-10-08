#!/usr/bin/env bash

# Check input file
if [[ -z "$1" ]]; then
  echo "No events.tsv file provided"
  exit 1
else
  if [[ ! -f "$1" ]]; then
    echo "The file '$1' does not exist."
    exit 1
  fi
fi
INPUT_PATH="${1}"

# Check output path
if [[ -z "$2" ]]; then
  echo "No output path provided"
  exit 1
else
  if [[ ! -d "$2" ]]; then
    mkdir -p "$2"
  fi
fi
OUTPUT_PATH="${2}"

# Do we have other arguments? "--shift 0.0" or "--verbose"?
shift  # get rid of INPUT_PATH
shift  # get rid of OUTPUT_PATH
# whatever remains will be passed along to bids_events_to_feat.py as "$@"

# Extract information from BIDS-compatible input file name
INFILE=${INPUT_PATH##*/}
if [[ $INFILE =~ sub-([A-Z][0-9]+).*_run-([0-9]+).*_events.tsv ]]; then
  SUB=${BASH_REMATCH[1]}
  RUN=${BASH_REMATCH[2]}
  echo "${INFILE}  ->  $SUB $RUN"
else
  echo "${INFILE}  ->  no match"
fi

# Use Input file, output path, and extracted info to generate feat timing files
# The test events.tsv has ~93 events: ~91 in 4 blocks, and 2 end-blocks
#   each block containing:
#   1 memory, 1 instruct, 2 question, 1 directions, 17 or 18 arrows.

# Extract all trials with '1' as the value, simple default
#   generates 7 txt files, one for each trial_type in events.tsv
bids_events_to_feat.py "${INPUT_PATH}" "${OUTPUT_PATH}" "$@"

# Extract all trials with '1' as the value, separating distance/immerse
#   generates 4 new txt files, two each for question and instruct stimuli
#   leaving 11 total files in the directory.
bids_events_to_feat.py "${INPUT_PATH}" "${OUTPUT_PATH}" \
  --trial-types memory instruct --split-on-stimulus instruct "$@"

# Extract at least 12 more files, each with a different trial/question/instruction combination
#   'memory' and 'instruct' trials each get 2 unsplit, 2 distance, and 2 immerse
#   All 'nan' answers go in additional failure files, adding to the total number.
for TT in memory instruct; do
  for STIM in "How badly do you feel?" "How vivid was the memory?"; do
      # for both memory and instruct trials, both questions
      bids_events_to_feat.py "${INPUT_PATH}" "${OUTPUT_PATH}" \
        --trial-types ${TT} \
        --use-response-from "${STIM}" --use-response-to ${TT} "$@"
      bids_events_to_feat.py "${INPUT_PATH}" "${OUTPUT_PATH}" \
        --split-on-stimulus instruct --trial-types ${TT} \
        --use-response-from "${STIM}" --use-response-to ${TT} "$@"
  done
done

# Extract all permutations of specified trial_types and stimuli for PPI analyses
#   Creates 8 new txt files
bids_events_to_feat.py "${INPUT_PATH}" "${OUTPUT_PATH}" \
  --ppi-trial-types memory instruct --ppi-stimuli-from instruct "$@"
