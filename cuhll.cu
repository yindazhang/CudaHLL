#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

#define BLOCK_SIZE

enum Algo
{
    cpu = 0,
    basic,
    opt,
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
    case opt:
        return "opt";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_vector(uint32_t *mat, int N, double skewness);
bool verify_hll(uint8_t *expected, uint8_t *actual, int M);
void runAlgo(Algo algo, int N, int M, uint32_t *A, uint8_t *C, uint32_t *dA, uint8_t *dC);
void runCpu(int N, int M, uint32_t *A, uint8_t *C);

const std::string errLogFile = "hllValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::mt19937 distribution;

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("hll.cu", "CUDA HLL kernels");
    options.add_options()("size", "dataset size (2^N)", cxxopts::value<uint8_t>()->default_value("20"))
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

    uint32_t SIZE = clFlags["size"].as<uint8_t>();
    if (SIZE > 25)
    {
        std::cout << "--size must be smaller than 25" << std::endl;
        exit(EXIT_FAILURE);
    }
    SIZE = (1 << SIZE);
    double skewness = clFlags["skew"].as<double>();

    uint8_t HLLB = clFlags["bconfig"].as<uint8_t>();
    if (HLLB < 4 || HLLB > 25)
    {
        std::cout << "--b in HLL must be in the range [4,25]" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint32_t HLLM = (1 << HLLB);


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
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint32_t *A = nullptr, *dA = nullptr;
    uint8_t *C = nullptr, *C_ref = nullptr, *dC = nullptr;

    A = (uint32_t *)malloc(sizeof(uint32_t) * SIZE);
    C = (uint8_t *)malloc(sizeof(uint8_t) * HLLM);
    C_ref = (uint8_t *)malloc(sizeof(uint8_t) * HLLM);

    randomize_vector(A, SIZE, skewness);
    memset(C, 0, sizeof(uint8_t) * HLLM);
    memset(C_ref, 0, sizeof(uint8_t) * HLLM);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(uint32_t) * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(uint8_t) * HLLM));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(uint8_t) * HLLM));

    cudaCheck(cudaMemcpy(dA, A, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(uint8_t) * HLLM, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(uint8_t) * HLLM, cudaMemcpyHostToDevice));

    printf("size: %u, m in HLL: %u\n", SIZE, HLLM);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE)
    {
        printf("disabled validation\n");
    }
    else
    {
        // run cublas to get correct answer in dC_ref
        runCpu(SIZE, HLLM, A, C_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, SIZE, HLLM, A, C, dA, dC);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(uint8_t) * HLLM, cudaMemcpyDeviceToHost);

        if (verify_hll(C_ref, C, HLLM))
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
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, SIZE, HLLM, A, C, dA, dC);
        cudaCheck(cudaDeviceSynchronize());
    }

    // TODO: measure timing without memory transfers?
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = SIZE;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        SIZE);

    // free CPU and GPU memory
    free(A);
    free(C);
    free(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dC));

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
        INT16_MIN value = countl_zero(cpuHash(data, 1)) + 1;
        C[position] = max(C[position], value);
    }
}

__global__ void runBasic(int N, int M, uint32_t *A, uint8_t *C){
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        uint32_t data = A[index];
        uint32_t position = cudaHash(data, 0) % M;
        int value = __clz(cudaHash(data, 1)) + 1;
        atomicMax(&C[position], value);
    }
}

void runAlgo(Algo algo, int N, int M, uint32_t *A, uint8_t *C, uint32_t *dA, uint8_t *dC)
{
    switch (algo)
    {
    case cpu:
        runCpu(N, M, A, C);
        break;
    case basic:
    {
        dim3 gridDim(ROUND_UP_TO_NEAREST(SIZE, BLOCK_SIZE));
        dim3 blockDim(BLOCK_SIZE);
        runBasic<<<gridDim, blockDim>>>(N, M, dA, dC);
        break;
    }
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}
