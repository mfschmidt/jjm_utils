#!/bin/bash

# This script will read in the file specified as --input as a tab-separated table
# It will filter the columns by the --subset name
# And it will write out a new tab-separated file with only the columns needed.
# Specify -v or --verbose for more descriptive output.

# Handle command-line arguments
FORCE=0
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  -i|--input)
    INPUT_FILE="$2"
    shift # past argument
    shift # past value
    ;;
  -o|--output)
    OUTPUT_FILE="$2"
    shift # past argument
    shift # past value
    ;;
  -s|--subset)
    SUBSET="$2"
    shift # past argument
    shift # past value
    ;;
  -f|--force)
    FORCE=1
    shift # past argument
    shift # past value
    ;;
  -v|--verbose)
    VERBOSE="True"
    shift # past argument only
    ;;
  *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument only
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Fill in any missing arguments with defaults
if [[ -z $INPUT_FILE ]]; then
  echo "An input file must be specified."
  exit 1
fi
if [[ -z $SUBSET ]]; then
  SUBSET="motion6"
fi
if [[ -z $OUTPUT_FILE ]]; then
  OUTPUT_FILE=${INPUT_FILE:: -4}
  OUTPUT_FILE=${OUTPUT_FILE}_filter-${SUBSET}.tsv
fi
if [[ -f $OUTPUT_FILE ]]; then
  if [[ "$FORCE" == "0" ]]; then
    echo "$OUTPUT_FILE exists, and I don't want to overwrite it."
    echo "Please delete it if you'd like to replace it."
    echo "Quitting"
    exit 1
  else
    echo "overwriting $OUTPUT_FILE"
    rm -f $OUTPUT_FILE
  fi
fi

# Read columns available in input confounds file
HEAD=$(head -1 $INPUT_FILE)

# Define our criteria for columns to include/exclude
declare -a T_COMP_CORS_ARRAY
declare -a A_COMP_CORS_ARRAY
BASIC_ARRAY=("framewise_displacement")
MOTION_6_ARRAY=("trans_x" "trans_y" "trans_z" "rot_x" "rot_y" "rot_z")
CURIOUS_ARRAY=("global_signal" "csf" "white_matter")
for COL in $HEAD; do
  if [[ "$COL" == "t_comp_cor_"* ]]; then
    T_COMP_CORS_ARRAY+=("$COL")
  fi
  if [[ "$COL" == "a_comp_cor_"* ]]; then
    if [[ "10#${COL:11}" -lt "5" ]]; then
      A_COMP_CORS_ARRAY+=("$COL")
    fi
  fi
done

# Select columns from the confounds input file, based on the SUBSET
declare -a COLS
declare -a COL_IDS
I=0
for COL in $HEAD; do
  (( I++ ))
  if [[ " ${BASIC_ARRAY[@]} " =~ " ${COL} " ]]; then
    if [[ "$SUBSET" == "basic" || "$SUBSET" == "curious" ]]; then
      COLS+=("$COL")
      COL_IDS+=("$I")
    fi
  fi
  if [[ " ${MOTION_6_ARRAY[@]} " =~ " ${COL} " ]]; then
    if [[ "$SUBSET" == "basic" || "$SUBSET" == "curious" || "$SUBSET" == "motion6" ]]; then
      COLS+=("$COL")
      COL_IDS+=("$I")
    fi
  fi
  if [[ " ${T_COMP_CORS_ARRAY[@]} " =~ " ${COL} " ]]; then
    if [[ "$SUBSET" == "basic" || "$SUBSET" == "curious" ]]; then
      COLS+=("$COL")
      COL_IDS+=("$I")
    fi
  fi
  if [[ " ${A_COMP_CORS_ARRAY[@]} " =~ " ${COL} " ]]; then
    if [[ "$SUBSET" == "basic" || "$SUBSET" == "curious" ]]; then
      COLS+=("$COL")
      COL_IDS+=("$I")
    fi
  fi
  if [[ " ${CURIOUS_ARRAY[@]} " =~ " ${COL} " ]]; then
    if [[ "$SUBSET" == "curious" ]]; then
      COLS+=("$COL")
      COL_IDS+=("$I")
    fi
  fi
done

# Do the actual filtering
if [[ "${SUBSET}" == "all" ]]; then
  cp $INPUT_FILE $OUTPUT_FILE
else
  AWK_COLS=""
  for COL_ID in ${COL_IDS[@]}; do
    if [[ "${#AWK_COLS}" -eq "0" ]]; then
      AWK_COLS="\$${COL_ID}"
    else
      AWK_COLS="${AWK_COLS},\$${COL_ID}"
    fi
  done
  awk -v FS='\t' -v OFS='\t' "{print $AWK_COLS}" $INPUT_FILE >> $OUTPUT_FILE
fi

# Alert the user to our interpretation
if [[ "$VERBOSE" == "True" ]]; then
  echo "INPUT FILE   = ${INPUT_FILE}"
  echo "OUTPUT FILE  = ${OUTPUT_FILE}"
  echo "SUBSET       = ${SUBSET}"
  echo "     includes [${COLS[@]}]"
  echo "Confounds ${SUBSET}-filtered and written to ${OUTPUT_FILE}"
fi

