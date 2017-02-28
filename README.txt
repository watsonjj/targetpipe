targetpipe

A python module to read in CHEC data through target_io.py into the event
containers defined in ctapipe (The CTA Pipeline python module). This allows
CHEC data to be analaysed using methods defined in ctapipe, and is the first
step to fully integrating the offline analysis into ctapipe.

INSTALLATION
------------
Installation of all software (ctapipe, its requirements, and TargetIO) must
be done in the same conda environment created in the ctapipe installation
instructions!

This script is installed alongside TargetIO when 'make install' is executed,
assuming the cmake argument -DPYTHON=ON has been used.

REQUIREMENTS
------------
- ctapipe -
https://cta-observatory.github.io/ctapipe/getting_started/index.html

- target_io -
targetpipe require the module target_io to be installed. This is achieved
by following the python section in the TargetIO README.

- tqdm -
'pip install tqdm'
Some scripts use this module.
It provides a well-developed progress meter.

