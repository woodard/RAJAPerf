###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738930
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For details about use and distribution, please read raja-perfsuite/LICENSE.
#
###############################################################################

include_directories(.)

add_subdirectory(common)
add_subdirectory(basic)

set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    common
    basic)

list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )


blt_add_executable(
  NAME raja-perf-nolibs.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  basic/INIT3.cpp
  common/DataUtils.cpp
  common/Executor.cpp
  common/KernelBase.cpp
  common/OutputUtils.cpp
  common/RAJAPerfSuite.cpp
  common/RPTypes.hpp
  common/RunParams.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
)


if (ENABLE_CUDA)
  set_target_properties(raja-perf-nolibs.exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()  

