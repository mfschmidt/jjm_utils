#!/bin/bash

# This script can be used to 'install' sub_pipeline and wrappers to each node.

#  for NODE in a b c; do ./update_sub_pipeline.sh ${NODE}; done


ssh ${1} "
cd ~/jjm_utils
git pull
sudo cp pipelines/sub_{pipeline,fmriprep,qsiprep,mriqc,feat} pipelines/cleanup_tmp.sh pipelines/inventory_rawdata /usr/local/bin/
sudo chown root:mriproc /usr/local/bin/sub_{pipeline,fmriprep,qsiprep,mriqc,feat} /usr/local/bin/cleanup_tmp.sh /usr/local/bin/inventory_rawdata
sudo chmod 755 /usr/local/bin/sub_{pipeline,fmriprep,qsiprep,mriqc,feat} /usr/local/bin/cleanup_tmp.sh /usr/local/bin/inventory_rawdata
ls -l /usr/local/bin/sub_{pipeline,fmriprep,qsiprep,mriqc,feat} /usr/local/bin/cleanup_tmp.sh /usr/local/bin/inventory_rawdata
/usr/local/bin/cleanup_tmp.sh
"
