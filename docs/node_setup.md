Setting up a jjm node:

Image virtually booted from http://10.20.193.215/ubuntu-20.04.3-live-server-amd64.iso vi iLO

English, English (US), English (US)

Manually configure first NIC, enp3s0f0, eth as IPv4:
    Subnet: 10.20.193.0/24
    Address: 10.20.193.x where x is specified in spreadsheet
    Gateway: 10.20.193.1
    Name Servers: 1.1.1.1
    Search domains: nyspi-pet.cumc.columbia.edu

No proxy

Default mirror is fine

Update to the new installer. (If this isn't available, you may have misconfigured the network.

Disk configuration depends on hardware, but best to expand boot drive to its maximum rather than accepting the default 100GB.

Initial user:
    Ansible Assistant
    jjm*
    aa
    password
    password

Yes, install OpenSSH Server, but do not select anything else.

Wait for install and updates, then reboot. Dismount CD from iLO or console menu.