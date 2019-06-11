#!/bin/bash

if [ -z "$1" ];then
    RUN_DIR=build_rhel8
else
    RUN_DIR=$1
fi


./$RUN_DIR/bin/raja-perf.exe -pk | \
    tail -n +4 | sed -e '/^$/d' -e '/DONE/d' -e '/^Reading/d'| \
    while read kernel;do
	perf record -o $kernel.perf $RUN_DIR/bin/raja-perf.exe -k $kernel
    done
