#!/bin/bash

if [[ "$1" != "" ]]; then
  LIST="-l $1"
else
  LIST=""
fi

for SCRIPT in set_timezone initial_software install_docker install_slurm update_munge update_slurm; do
  ansible-playbook ${LIST} ~/ansible/${SCRIPT}.yml
done

