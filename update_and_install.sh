#!/bin/bash

# This script can be used to 'install' jjm_utils scripts and wrappers to each node.

#  for NODE in a b c; do ./update_and_install.sh ${NODE}; done

DEST="/usr/local/bin"
if [[ "$1" != "" ]]; then
    if [[ -e $1 ]]; then
        DEST=$1
    fi
fi
SRC_FILES=\
"pipelines/sub_{pipeline,fmriprep,qsiprep,mriqc,feat,freesurfer} \
 pipelines/cleanup_tmp.sh \
 pipelines/inventory_rawdata \
 mri/make_masks \
 confounds/filter_confounds.{py,sh}"
TGT_FILES=\
"${DEST}/sub_{pipeline,fmriprep,qsiprep,mriqc,feat,freesurfer} \
 ${DEST}/cleanup_tmp.sh \
 ${DEST}/inventory_rawdata \
 ${DEST}/make_masks \
 ${DEST}/filter_confounds.{py,sh}"

ssh ${1} "
cd ~/jjm_utils
git pull
sudo cp ${SRC_FILES} ${DEST}
sudo chown root:mriproc ${TGT_FILES}
sudo chmod 755 ${TGT_FILES}
ls -l ${TGT_FILES}
${DEST}/cleanup_tmp.sh
"
