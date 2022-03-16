#!/bin/bash

# These scripts determine slurm controller and nodes from /etc/ansible/hosts
# [heads] and [nodes] groups, so do not use -l. Edit the hosts file to change
# which nodes are used.

if [[ " $(cat /etc/ansible/hosts) " =~ heads ]]; then
    echo "Updating and restarting slurm"
    ansible-playbook /home/aa/jjm_utils/ansible/stop_slurm.yml
    ansible-playbook /home/aa/jjm_utils/ansible/update_slurm.yml
    ansible-playbook /home/aa/jjm_utils/ansible/start_slurm.yml
else
    echo "Update /etc/ansible/hosts with 'heads' node before updating"
fi

