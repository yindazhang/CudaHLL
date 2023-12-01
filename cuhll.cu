#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include "cxxopts.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

// from https://github.com/jarro2783/cxxopts

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

#define BLOCK_SIZE 1024
#define MAX_MERGE 4096

#define MULTI 4

class DeviceMemory {
public:
    DeviceMemory(size_t size) {
        cudaMalloc(&ptr, size);
    }

    ~DeviceMemory() {
        cudaFree(ptr);
    }

    void* get() const {
        return ptr;
    }

private:
    void* ptr = nullptr;
};

template <typename T>
class HostMemory {
public:
    HostMemory(size_t size) : size(size), data(new T[size]) {}

    ~HostMemory() {
        delete[] data;
    }

    T* get() const {
        return data;
    }

    size_t getSize() const {
        return size;
    }

private:
    T* data;
    size_t size;
};


enum Algo
{
    cpu = 0,
    basic,
    filter,
    multi,
    merge,
    multi_merge,
    numAlgos
};

const char *algo2str(Algo a)
{
    switch (a)
    {
    case cpu:
        return "cpu";
    case basic:
        return "basic";
    case filter:
        return "filter";
    case multi:
        return "multi";
    case merge:
        return "merge";
    case multi_merge: 
        return "multi_merge";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
bool verify_hll(uint8_t *expected, uint8_t *actual, int M);
void runAlgo(Algo algo, int N, int M, uint32_t *A, uint8_t *C, uint32_t *dA, uint8_t *dC, cudaStream_t stream);
void runCpu(int N, int M, uint32_t *A, uint8_t *C);

//TODO: genreate duplicates based on skewness
void randomize_vector(uint32_t *mat, int N, double skewness);

const std::string errLogFile = "hllValidationFailure.txt";
std::mt19937 distribution;

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("hll.cu", "CUDA HLL kernels");
    options.add_options()("size", "dataset size (2^N)", cxxopts::value<uint8_t>()->default_value("26"))
        ("skew", "skewness of the dataset", cxxopts::value<double>()->default_value("0")) //TODO
        ("bconfig", "b in the configuration of HLL", cxxopts::value<uint8_t>()->default_value("10"))
        ("reps", "repeat HLL this many times", cxxopts::value<uint16_t>()->default_value("1"))
        ("algo", "HLL algorithm to use, a number in [0,4], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0"))
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))     
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    uint8_t inputSize = clFlags["size"].as<uint8_t>();
    if (inputSize >= 30) {
    std::cerr << "Error: --size must be smaller than 30" << std::endl;
    exit(EXIT_FAILURE);
    }
    uint32_t SIZE = (1 << inputSize);

    double skewness = clFlags["skew"].as<double>();

    uint8_t HLLB = clFlags["bconfig"].as<uint8_t>();
    if (HLLB < 4 || HLLB > 25)
    {
        std::cout << "--b in HLL must be in the range [4,25]" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint32_t HLLM = (1 << clFlags["bconfig"].as<uint8_t>());


    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    distribution.seed(SEED);
    printf("Using %s algorithm\n", algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint32_t* h_A;
    cudaMallocHost(&h_A, sizeof(uint32_t) * SIZE);
    randomize_vector(h_A, SIZE, skewness);

    uint8_t* h_C;
    cudaMallocHost(&h_C, sizeof(uint8_t) * HLLM);
    memset(h_C, 0, sizeof(uint8_t) * HLLM);

    uint8_t* h_C_ref;
    cudaMallocHost(&h_C_ref, sizeof(uint8_t) * HLLM);
    memset(h_C_ref, 0, sizeof(uint8_t) * HLLM);

    DeviceMemory dA(sizeof(uint32_t) * SIZE);
    DeviceMemory dC(sizeof(uint8_t) * HLLM);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(dA.get(), h_A, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice, stream);
    
    printf("size: %u, m in HLL: %u\n", SIZE, HLLM);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE || ALGO == cpu)
    {
        printf("disabled validation\n");
    }
    else
    {
        runCpu(SIZE, HLLM, h_A, h_C_ref);


        // run user's algorithm, filling in dC
        runAlgo(ALGO, SIZE, HLLM, h_A, h_C, static_cast<uint32_t*>(dA.get()), static_cast<uint8_t*>(dC.get()), stream);

        cudaMemcpyAsync(h_C, dC.get(), sizeof(uint8_t) * HLLM, cudaMemcpyDeviceToHost, stream);

        // copy both results back to host
        cudaStreamSynchronize(stream);


        if (verify_hll(h_C_ref, h_C, HLLM))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++) {
        runAlgo(ALGO, SIZE, HLLM, h_A, h_C, static_cast<uint32_t*>(dA.get()), static_cast<uint8_t*>(dC.get()), stream);
        cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = SIZE;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GIPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        SIZE);

    // free CPU and GPU memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_ref);
    cudaStreamDestroy(stream);
    return 0;
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to random values. */
void randomize_vector(uint32_t *mat, int N, double skewness)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution();
    }
}

bool verify_hll(uint8_t *expected, uint8_t *actual, int M)
{
    for (int i = 0; i < M; i++)
    {
        if (expected[i] != actual[i])
        {
            printf("Divergence! Should be %u, is %u at [%d]\n",
                    expected[i], actual[i], i);
            return false;
        }
    }
    return true;
}

uint32_t cpuHash(uint32_t data, uint32_t seed){
    uint32_t ret = data * 0x114253d5;
    ret ^= (0x2745937f * seed);
    return ret;
}

__device__  uint32_t cudaHash(uint32_t data, uint32_t seed){
    uint32_t ret = data * 0x114253d5;
    ret ^= (0x2745937f * seed);
    return ret;
}

void runCpu(int N, int M, uint32_t *A, uint8_t *C){
    for (int i = 0; i < N; ++i)
    {
        uint32_t data = A[i];
        uint32_t position = cpuHash(data, 0) % M;
        uint8_t value = __builtin_clz(cpuHash(data, 1)) + 1;
        if(value > C[position])
            C[position] = value;
    }
}

// atomicMax in CUDA only supports uint32_t
// We implement atomicMax for uint8_t based on atomicCAS
__device__ void atomicMax8(uint8_t* address, uint8_t val){
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(size_t)address & 3];
    unsigned int old, assumed, max_, new_;

    old = *base_address;
    do {
        assumed = old;
        max_ = max(val, (char)__byte_perm(old, 0, ((size_t)address & 3)));
        new_ = __byte_perm(old, max_, sel);
        old = atomicCAS(base_address, assumed, new_);
    } while (assumed != old && val > *address);
    return;
}

__global__ void runBasic(int N, int M, uint32_t *A, uint8_t *C){
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        uint32_t data = A[index];
        uint32_t position = cudaHash(data, 0) % M;
        int value = __clz(cudaHash(data, 1)) + 1;
        atomicMax8(&C[position], value);
    }
}

__global__ void runFilter(int N, int M, uint32_t *A, uint8_t *C){
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        uint32_t data = A[index];
        uint32_t position = cudaHash(data, 0) % M;
        int value = __clz(cudaHash(data, 1)) + 1;
        if(value > C[position]) // Add a simple filter
            atomicMax8(&C[position], value);
    }
}

// Process MULTI items in one thread
__global__ void runMulti(int N, int M, uint32_t *A, uint8_t *C){
    uint32_t start = (blockIdx.x * blockDim.x + threadIdx.x) * MULTI;

    uint32_t position[MULTI];
    uint8_t value[MULTI];

    for(uint32_t i = 0;i < MULTI;++i){
        uint32_t index = start + i;
        if(index < N){
            position[i] = cudaHash(A[index], 0) % M;
            value[i] = __clz(cudaHash(A[index], 1)) + 1;
        }
    }

    for(uint32_t i = 0;i < MULTI;++i){
        uint32_t index = start + i;
        if(index < N && value[i] > C[position[i]]){
            atomicMax8(&C[position[i]], value[i]);
        }
    }
}

// Each block maintains a shared HyperLogLog
// Then merge multiple HyperLogLogs
__global__ void runMerge(int N, int M, int gap, uint32_t *A, uint8_t *C){
    __shared__ uint8_t merge[MAX_MERGE];

    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        uint32_t data = A[index];
        uint32_t position = cudaHash(data, 0) % M;
        int value = __clz(cudaHash(data, 1)) + 1;
        if(value > merge[position])
            atomicMax8(&merge[position], value);
    }

    __syncthreads();

    uint32_t start = threadIdx.x * gap;
    uint32_t end = (threadIdx.x + 1) * gap;

    for(uint32_t i = start;i < end;++i){
        if(i < M && merge[i] > C[i]){
            atomicMax8(&C[i], merge[i]);
        }
    }
}


// Merge + Process MULTI items in one thread
__global__ void runMultiMerge(int N, int M, int gap, uint32_t *A, uint8_t *C){
    __shared__ uint8_t merge[MAX_MERGE];
    
    uint32_t start = (blockIdx.x * blockDim.x + threadIdx.x) * MULTI;

    uint32_t position[MULTI];
    uint8_t value[MULTI];

    for(uint32_t i = 0;i < MULTI;++i){
        uint32_t index = start + i;
        if(index < N){
            position[i] = cudaHash(A[index], 0) % M;
            value[i] = __clz(cudaHash(A[index], 1)) + 1;
        }
    }

    for(uint32_t i = 0;i < MULTI;++i){
        uint32_t index = start + i;
        if(index < N){
            if(value[i] > merge[position[i]])
                atomicMax8(&merge[position[i]], value[i]);
        }
    }

    __syncthreads();

    uint32_t begin = threadIdx.x * gap;
    uint32_t end = (threadIdx.x + 1) * gap;

    for(uint32_t i = begin;i < end;++i){
        if(i < M && merge[i] > C[i]){
            atomicMax8(&C[i], merge[i]);
        }
    }
}


void runAlgo(Algo algo, int N, int M, uint32_t *A, uint8_t *C, uint32_t *dA, uint8_t *dC, cudaStream_t stream) {
    switch (algo) {
    case cpu:
        runCpu(N, M, A, C);
        break;
    case basic:
        runBasic<<<ROUND_UP_TO_NEAREST(N, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(N, M, dA, dC);
        break;
    case filter:
        runFilter<<<ROUND_UP_TO_NEAREST(N, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(N, M, dA, dC);
        break;
    case multi:
        runMulti<<<ROUND_UP_TO_NEAREST(N, BLOCK_SIZE * MULTI), BLOCK_SIZE, 0, stream>>>(N, M, dA, dC);
        break;
    case merge:
        assert(M <= MAX_MERGE);
        runMerge<<<ROUND_UP_TO_NEAREST(N, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(N, M, ROUND_UP_TO_NEAREST(M, BLOCK_SIZE), dA, dC);
        break;
    case multi_merge:
        assert(M <= MAX_MERGE);
        runMultiMerge<<<ROUND_UP_TO_NEAREST(N, BLOCK_SIZE * MULTI), BLOCK_SIZE, 0, stream>>>(N, M, ROUND_UP_TO_NEAREST(M, BLOCK_SIZE), dA, dC);
        break;
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    // Removed cudaDeviceSynchronize, replaced with cudaStreamSynchronize in main
    cudaCheck(cudaGetLastError()); // Check for errors from kernel run
}
