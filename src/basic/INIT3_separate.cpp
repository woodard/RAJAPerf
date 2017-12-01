//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT3 kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;
/// }
///

#include "INIT3.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define INIT3_DATA \
  ResReal_ptr out1 = m_out1; \
  ResReal_ptr out2 = m_out2; \
  ResReal_ptr out3 = m_out3; \
  ResReal_ptr in1 = m_in1; \
  ResReal_ptr in2 = m_in2;

#define INIT3_BODY  \
  out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;

void init_3_cuda_baseline(KernelBase &kb,RepIndex_type run_reps, Index_type ibegin, Index_type iend, Real_ptr m_out1, Real_ptr m_out2, Real_ptr m_out3, 
Real_ptr m_in1, Real_ptr m_in2);


void init_3_cuda_raja(KernelBase &kb, RepIndex_type run_reps, Index_type ibegin, Index_type iend, Real_ptr m_out1, Real_ptr m_out2, Real_ptr m_out3, 
Real_ptr m_in1, Real_ptr m_in2);


INIT3::INIT3(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT3, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

INIT3::~INIT3() 
{
}


void INIT3::setUp(VariantID vid)
{
  allocAndInitData(m_out1, getRunSize(), vid);
  allocAndInitData(m_out2, getRunSize(), vid);
  allocAndInitData(m_out3, getRunSize(), vid);
  allocAndInitData(m_in1, getRunSize(), vid);
  allocAndInitData(m_in2, getRunSize(), vid);
}

void INIT3::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      INIT3_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT3_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INIT3_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT3_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      INIT3_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT3_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      INIT3_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT3_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#define NUMTEAMS 128

    case Base_OpenMPTarget : {

      INIT3_DATA;

      int n = getRunSize();
      #pragma omp target enter data map(to:in1[0:n],in2[0:n],out1[0:n],out2[0:n],out3[0:n])

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT3_BODY;
        }

      }
      stopTimer();

      #pragma omp target exit data map(delete:in1[0:n],in2[0:n]) map(from:out1[0:n],out2[0:n],out3[0:n])

      break;
    }

    case RAJA_OpenMPTarget : {

      INIT3_DATA;

      int n = getRunSize();
      #pragma omp target enter data map(to:in1[0:n],in2[0:n],out1[0:n],out2[0:n],out3[0:n])

      startTimer();
      #pragma omp target data use_device_ptr(in1,in2,out1,out2,out3)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT3_BODY;
        });

      }
      stopTimer();

      #pragma omp target exit data map(delete:in1[0:n],in2[0:n]) map(from:out1[0:n],out2[0:n],out3[0:n])

      break;
    }
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             

//#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {
      init_3_cuda_baseline(*this,run_reps,ibegin,iend,m_out1,m_out2,m_out3,m_in1,m_in2);
      break; 
    }

    case RAJA_CUDA : {
      init_3_cuda_raja(*this,run_reps,ibegin,iend,m_out1,m_out2,m_out3,m_in1,m_in2);
      break;
    }
//#endif

    default : {
      std::cout << "\n  INIT3 : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INIT3::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunSize());
  checksum[vid] += calcChecksum(m_out2, getRunSize());
  checksum[vid] += calcChecksum(m_out3, getRunSize());
}

void INIT3::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_out1);
  deallocData(m_out2);
  deallocData(m_out3);
  deallocData(m_in1);
  deallocData(m_in2);
}

} // end namespace basic
} // end namespace rajaperf
