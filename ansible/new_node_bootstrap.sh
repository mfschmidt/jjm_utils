#!/bin/bash


##  The Bootstrapper
##  
##  This file sets up the pre-requisites on a new cluster node before we can
##  even begin using ansible to install software and configure slurm, etc.
##
##  It is assumed the aa user was set up as user 1000 during setup,
##  that aa is the only user, and that it is in the sudo group.
##  This script must be run as root (or sudo)

# Update all software and install prerequisites.

apt update
apt upgrade -y
apt autoremove -y
apt install zip unzip git gnupg vim sshpass python2 python3 ansible nfs-kernel-server nfs-common -y


# Check on fstab and update it if necessary.

mkdir -p /data
mkdir -p /mnt/rawdata
mkdir -p /mnt/derivatives
mkdir -p /mnt/hcp

if grep -q "10.20.193.33" /etc/fstab; then
    echo "MIND /data already mapped"
else
    echo "" >> /etc/fstab
    echo "# MIND data directory" >> /etc/fstab
    echo "10.20.193.33:/volume/data /data nfs tcp,rsize=8192,wsize=8192,retrans=64,hard,noacl 0 0" >> /etc/fstab
    echo "10.20.193.33:/volume/data/BI/human/rawdata /mnt/rawdata nfs tcp,rsize=8192,wsize=8192,retrans=64,hard,noacl 0 0" >> /etc/fstab
    echo "10.20.193.33:/volume/data/BI/human/derivatives /mnt/derivatives nfs tcp,rsize=8192,wsize=8192,retrans=64,hard,noacl 0 0" >> /etc/fstab
    echo "" >> /etc/fstab
    echo "# HCP data" >> /etc/fstab
    echo "//10.20.220.21:/MRI_HCP/HCP_1200 /mnt/hcp cifs credentials=/home/aa/.ssh/.hcp.credentials,vers=2.0 0 0" >> /etc/fstab
    mount -av
    echo "MIND data newly mapped from 10.20.193.33 to /data/"
fi


# Expand the HDD we're using to full size.

# sudo lvresize -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
# sudo resize2fs /dev/ubuntu-vg/ubuntu-lv


# Add user 'aa' to password-less sudoers so it can manage nodes later.
if [ -f /etc/sudoers.d/aa ]; then
    echo "User 'aa' already has sudoer access."
else
    echo "aa  ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/aa
    echo "User aa has now been given sudoer access."
fi


# Set up host with manual local DNS
cd /home/aa
git clone https://github.com/mfschmidt/jjm_utils.git
cp /home/aa/jjm_utils/data/hosts /etc/hosts
sed -i "s/{HOSTNAME}/$(hostname)/g" /etc/hosts
chown aa:aa /home/aa/jjm_utils --recursive

# Install FSL
mkdir -p /opt/fsl
cd /opt/fsl
wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
python2 /opt/fsl/fslinstaller.py -q

# Install Connectome Workbench
mkdir -p /opt/
cd /opt/
wget https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip
ln -s /opt/workbench/bin_linux64/wb_command /usr/local/bin/wb_command

