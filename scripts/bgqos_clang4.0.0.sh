#!/bin/bash

##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For additional details and restrictions, please see RAJA/LICENSE.txt
##

rm -rf build_bgqos-clang-4.0.0 2>/dev/null
mkdir build_bgqos-clang-4.0.0 && cd build_bgqos-clang-4.0.0
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.3

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/bgqos/clang_4_0_0.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_bgqos_clang-4.0.0 \
  "$@" \
  ${PERFSUITE_DIR}
