#!/bin/bash

rsync --archive --recursive --verbose -P --times --delete tct@tct-computer.physik.uzh.ch:/home/tct/measurements_data/* /home/alf/cernbox/projects/4D_sensors/AC-LGAD_FBK_RSD1/measurements_data
