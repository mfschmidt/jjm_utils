Usage
=====

Installation
------------

Scripts are written to avoid use of python libraries and be easily executed
from any bash shell with access to python3. For that reason, the scripts
should simply be copied to a location in the path. In our jjm context,
the 'aa' user has a clone of this git repository in its home directory on
each jjm node. It can easily pull new changes, then copy them to
`/usr/local/bin/` so all users have access. This is automated by the included
script, `update_and_install.sh`. Pass it the node to install onto, and it
will do the rest (assuming keys are configured in ~/.ssh/config).

    update_and_install.sh jjm8

or, to do several at a time,

    for NODE in jjm5 jjm6; do update_and_install.sh ${NODE}; done

