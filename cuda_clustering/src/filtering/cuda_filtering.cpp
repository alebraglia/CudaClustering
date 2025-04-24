#include "cuda_clustering/filtering/cuda_filtering.hpp"
#include <stdlib.h>
CudaFilter::CudaFilter()
{
  FilterType_t type = PASSTHROUGH;

  this->setP.type = type;
  this->setP.dim = 2;
  this->setP.upFilterLimits = 1.0;
  this->setP.downFilterLimits = 0.0;
  this->setP.limitsNegative = false;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr CudaFilter::filterPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  cudaStream_t stream = NULL;
  cudaStreamCreate ( &stream );

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);

  unsigned int nCount = cloudSrc->width * cloudSrc->height;
  float *inputData = (float *)cloudSrc->points.data();

  cloudDst->width  = nCount;
  cloudDst->height = 1;
  cloudDst->resize (cloudDst->width * cloudDst->height);

  float *outputData = (float *)cloudDst->points.data();

  memset(outputData,0,sizeof(float)*4*nCount);

  std::cout << "\n------------checking CUDA ---------------- "<< std::endl;
  std::cout << "CUDA Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  float *input = NULL;
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input );
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *output = NULL;
  cudaMallocManaged(&output, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, output );
  cudaStreamSynchronize(stream);

  cudaFilter filterTest(stream);

  unsigned int countLeft = 0;
  std::cout << "\n------------checking CUDA PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  filterTest.set(this->setP);

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  filterTest.filter(output, &countLeft, input, nCount);
  checkCudaErrors(cudaMemcpyAsync(outputData, output, sizeof(float) * 4 * countLeft, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA PassThrough by Time: " << time_span.count() << " ms." << std::endl;
  std::cout << "CUDA PassThrough before filtering: " << nCount << std::endl;
  std::cout << "CUDA PassThrough after filtering: " << countLeft << std::endl;

  //
  
  unsigned int dimension = sizeof(float) * 4 * nCount;
  unsigned int cut_dim = dimension - dimension/4;
  
  float *seg_o_ne = malloc(sizeof(float) * cut_dim);

  for (std::size_t i = 0; i < dimension; ++i)
  {
      seg_o_ne[i] = output[i*4];
      seg_o_ne[i+1] = output[i*4+1];
      seg_o_ne[i+2] = output[i*4+2];
  }
  return cloudNew;
  //
}