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

#include <basic/INIT3.hpp>

#include <common/DataUtils.hpp>

#include <RAJA/RAJA.hpp>

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


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT3_DATA_SETUP_CUDA \
  Real_ptr out1; \
  Real_ptr out2; \
  Real_ptr out3; \
  Real_ptr in1; \
  Real_ptr in2; \
\
  allocAndInitCudaDeviceData(out1, m_out1, iend); \
  allocAndInitCudaDeviceData(out2, m_out2, iend); \
  allocAndInitCudaDeviceData(out3, m_out3, iend); \
  allocAndInitCudaDeviceData(in1, m_in1, iend); \
  allocAndInitCudaDeviceData(in2, m_in2, iend);

#define INIT3_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out1, out1, iend); \
  getCudaDeviceData(m_out2, out2, iend); \
  getCudaDeviceData(m_out3, out3, iend); \
  deallocCudaDeviceData(out1); \
  deallocCudaDeviceData(out2); \
  deallocCudaDeviceData(out3); \
  deallocCudaDeviceData(in1); \
  deallocCudaDeviceData(in2);

__global__ void init3(Real_ptr out1, Real_ptr out2, Real_ptr out3, 
                      Real_ptr in1, Real_ptr in2, 
                      Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INIT3_BODY; 
   }
}


void init_3_cuda_baseline(KernelBase &kb,RepIndex_type run_reps, Index_type ibegin, Index_type iend, Real_ptr m_out1, Real_ptr m_out2, Real_ptr m_out3, 
Real_ptr m_in1, Real_ptr m_in2)
{  
      INIT3_DATA_SETUP_CUDA;

      kb.startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         init3<<<grid_size, block_size>>>( out1, out2, out3, in1, in2, 
                                           iend ); 

      }
      kb.stopTimer();

      INIT3_DATA_TEARDOWN_CUDA;
}

void init_3_cuda_raja(KernelBase &kb, RepIndex_type run_reps, Index_type ibegin, Index_type iend, Real_ptr m_out1, Real_ptr m_out2, Real_ptr m_out3, 
Real_ptr m_in1, Real_ptr m_in2)
{  

      INIT3_DATA_SETUP_CUDA;
      kb.startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
           INIT3_BODY;
         });

      }
      kb.stopTimer();
      INIT3_DATA_TEARDOWN_CUDA;
    }

#endif

} // end namespace basic
} // end namespace rajaperf
