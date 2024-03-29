username=$1
userid=$2
password=$3
humanname=$4

sudo adduser --quiet --disabled-password --home /data/export/home/${username} --no-create-home --uid ${userid} --gecos "${humanname}" ${username}
if [[ "$(id -u ${username})" == "${userid}" ]]; then
  sudo pwconv
  # Group permissions
  sudo usermod -aG docker ${username}
  # Home directory
  if [[ ! -e /data/export/home/${username}/.bashrc ]]; then

    sudo mkdir /var/tmp/${username}
    sudo cp -r /etc/skel/. /var/tmp/${username}
    sudo mkdir /var/tmp/${username}/.ssh
    sudo chown ${username}:${username} /var/tmp/${username} --recursive
    sudo cp /var/tmp/${username}/. /data/export/home/${username}
    sudo rm /var/tmp/${username} -rf
  fi
  # Password
  # sudo echo ${username}:${password} | chpasswd
fi

