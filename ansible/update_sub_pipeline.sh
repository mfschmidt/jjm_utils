#!/bin/bash

ssh ${1} "
cd ~/jjm_utils
git pull
sudo cp sub_pipeline /usr/local/bin/
sudo chown root:mriproc /usr/local/bin/sub_pipeline
sudo chmod 755 /usr/local/bin/sub_pipeline
ls -l /usr/local/bin/sub_pipeline
"

