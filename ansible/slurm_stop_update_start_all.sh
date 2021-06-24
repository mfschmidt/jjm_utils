#!/bin/bash

ansible-playbook /home/aa/ansible/stop_slurm.yml
ansible-playbook /home/aa/ansible/update_slurm.yml
ansible-playbook /home/aa/ansible/start_slurm.yml

