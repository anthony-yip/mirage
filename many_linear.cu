// #include "include/mirage/persistent_kernel/tasks/linear.cuh"
#define MEASURE 1
#include "include/mirage/persistent_kernel/tasks/linear_cutlass.cuh"
#include "include/mirage/persistent_kernel/tasks/linear_cutlass_split.cuh"
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

static constexpr int SINGLE_KERNEL_THREADS = 128;
static constexpr int MAX_SHARE_MEMORY_SIZE = 160 * 1024;
static constexpr size_t NUM_LAYERS = 30;
static constexpr size_t SM_COUNT = 96;
static constexpr size_t OUTPUT_SIZE = 64;
static constexpr size_t REDUCTION_SIZE = 1024;
static constexpr size_t BATCH_SIZE = 1;
static constexpr bool USE_PIPELINE = false;
static constexpr size_t NUM_TRIALS = 100;
static constexpr size_t NUM_WARMUP_TRIALS = 5;
static constexpr size_t K_TILE_SIZE = 256;
static constexpr size_t EXTRA_INFO_SIZE = 8;
static constexpr size_t FORLOOP_LENGTH = REDUCTION_SIZE / K_TILE_SIZE;
using bfloat16 = type::bfloat16_t;        // kernel::linear_prefetch<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(input_ptr_next, weight_ptr_next, smem_next);

__global__ void main_kernel(void *d_input, void *d_weight, void *d_output, size_t *clock_cycles_mem, size_t *clock_cycles_compute, size_t *clock_cycles_extra, bool write_measurements) {
    extern __shared__ char smem[];
  
    if constexpr (USE_PIPELINE) {
      size_t time_start_prefetch = clock64();
      kernel::linear_prefetch<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT, K_TILE_SIZE>(d_input, d_weight, smem, nullptr);
      size_t time_end_prefetch = clock64();

      for (size_t layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {
        char * shared_mem_start = smem + (MAX_SHARE_MEMORY_SIZE / 2) * (layer_num % 2);

        void * input_ptr = (bfloat16 *)d_input + (layer_num * BATCH_SIZE * REDUCTION_SIZE);
        void * weight_ptr = (bfloat16 *)d_weight + (layer_num * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);
        void * output_ptr = (bfloat16 *)d_output + (layer_num * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);

        char * smem_next = smem + (MAX_SHARE_MEMORY_SIZE / 2) * ((layer_num + 1) % 2);
        void * input_ptr_next = (bfloat16 *)d_input + ((layer_num + 1) * BATCH_SIZE * REDUCTION_SIZE);
        // void * weight_ptr_next = (bfloat16 *)d_weight + ((layer_num + 1) * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);
        void * weight_ptr_next = (char*) weight_ptr + (REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT);

        kernel::linear_main<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT, K_TILE_SIZE>(
        input_ptr,
        weight_ptr,
        nullptr,
        output_ptr,
        BATCH_SIZE,
        false,
        shared_mem_start,
        layer_num < NUM_LAYERS - 1,
        input_ptr_next,
        weight_ptr_next,
        smem_next,
        clock_cycles_mem + layer_num * (REDUCTION_SIZE / K_TILE_SIZE),
        clock_cycles_compute + layer_num * (REDUCTION_SIZE / K_TILE_SIZE)
        );

      }
      #if MEASURE
      clock_cycles_compute[NUM_LAYERS * REDUCTION_SIZE / K_TILE_SIZE - 4] = time_end_prefetch - time_start_prefetch;
      #endif
    }

    else {
      for (size_t layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {
        void * input_ptr = (bfloat16 *)d_input + (layer_num * BATCH_SIZE * REDUCTION_SIZE);
        void * weight_ptr = (bfloat16 *)d_weight + (layer_num * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);
        void * output_ptr = (bfloat16 *)d_output + (layer_num * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);

        size_t kernel_start = clock64();
        size_t *clock_cycles_mem_ptr = nullptr;
        size_t *clock_cycles_compute_ptr = nullptr;
        size_t *clock_cycles_extra_ptr = nullptr;
        if (write_measurements) {
          clock_cycles_mem_ptr = clock_cycles_mem + layer_num * (REDUCTION_SIZE / K_TILE_SIZE);
          clock_cycles_compute_ptr = clock_cycles_compute + layer_num * (REDUCTION_SIZE / K_TILE_SIZE);
          clock_cycles_extra_ptr = clock_cycles_extra + layer_num * (EXTRA_INFO_SIZE);
        }
        kernel::linear_kernel<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT, K_TILE_SIZE, EXTRA_INFO_SIZE>(
        input_ptr,
        weight_ptr,
        nullptr,
        output_ptr,
        BATCH_SIZE,
        false,
        clock_cycles_mem_ptr,
        clock_cycles_compute_ptr,
        clock_cycles_extra_ptr
        );
        size_t kernel_end = clock64();
        #if MEASURE
        if (write_measurements) {
          if (threadIdx.x == 10 && blockIdx.x == 4) {
          clock_cycles_extra[layer_num * (EXTRA_INFO_SIZE) + 0] = kernel_end - kernel_start;
          }
        }
        #endif
      }
    }

}


int main() {

  // Create synthetic inputs and weight tensors, cudaMemcpy to device memory
  printf("BATCH_SIZE = %zu, ", BATCH_SIZE);
  printf("SM_COUNT = %zu,", SM_COUNT);
  printf("REDUCTION_SIZE = %zu,", REDUCTION_SIZE);
  printf("OUTPUT_SIZE = %zu,", OUTPUT_SIZE);
  printf("|||");
  printf("NUM_LAYERS = %zu,", NUM_LAYERS);
  printf("NUM_TRIALS = %zu,", NUM_TRIALS);
  printf("NUM_WARMUP_TRIALS = %zu,", NUM_WARMUP_TRIALS);
  printf("K_TILE_SIZE = %zu", K_TILE_SIZE);
  printf("\n--------------------------------\n");

  // Launch the main kernel and start the timer
  cudaSetDevice(6);

  int device;
  cudaGetDevice(&device);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  printf("a single persistent kernel\n");

  // Allocate device memory for d_input, d_weight, d_output and fill with ones

  size_t input_size = NUM_LAYERS * BATCH_SIZE * REDUCTION_SIZE * sizeof(bfloat16);
  size_t weight_size = NUM_LAYERS * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT * sizeof(bfloat16);
  size_t output_size = NUM_LAYERS * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT * sizeof(bfloat16);

  bfloat16 *d_input = nullptr;
  bfloat16 *d_weight = nullptr;
  bfloat16 *d_output = nullptr;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_weight, weight_size);
  cudaMalloc(&d_output, output_size);

  // Fill with ones
  // Allocate host buffers
  bfloat16 *h_input = (bfloat16*)malloc(input_size);
  bfloat16 *h_weight = (bfloat16*)malloc(weight_size);
  bfloat16 *h_output = (bfloat16*)malloc(output_size);

  for (size_t i = 0; i < input_size / sizeof(bfloat16); ++i) {
      h_input[i] = bfloat16(1.0f);
  }
  for (size_t i = 0; i < weight_size / sizeof(bfloat16); ++i) {
      h_weight[i] = bfloat16(1.0f);
  }
  for (size_t i = 0; i < output_size / sizeof(bfloat16); ++i) {
      h_output[i] = bfloat16(1.0f);
  }

  cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, output_size, cudaMemcpyHostToDevice);

  free(h_input);
  free(h_weight);

  // Allocate device memory for clock_cycles_mem and clock_cycles_compute
  size_t *buffer = nullptr;
  cudaMalloc(&buffer, 1 << 18);
  size_t *d_clock_cycles_mem = nullptr;
  size_t *d_clock_cycles_compute = nullptr;
  size_t *d_clock_cycles_extra = nullptr;
  cudaMalloc(&d_clock_cycles_compute, NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t));
  cudaMalloc(&d_clock_cycles_mem, NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t));
  cudaMalloc(&d_clock_cycles_extra, NUM_LAYERS * (EXTRA_INFO_SIZE) * sizeof(size_t));
  // Launcher persistent kernel
  cudaFuncSetAttribute(main_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        MAX_SHARE_MEMORY_SIZE);

  for (size_t i = 0; i < NUM_WARMUP_TRIALS; ++i) {
    main_kernel<<<dim3(sm_count, 1, 1),
                      dim3(SINGLE_KERNEL_THREADS, 1, 1),
                      MAX_SHARE_MEMORY_SIZE /*smem*/>>>(d_input, d_weight, d_output, d_clock_cycles_mem, d_clock_cycles_compute, d_clock_cycles_extra, false);
  }
  std::array<float, NUM_TRIALS> all_elapsed_ms;
  for (size_t i = 0; i < NUM_TRIALS; ++i) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    main_kernel<<<dim3(sm_count, 1, 1),
                      dim3(SINGLE_KERNEL_THREADS, 1, 1),
                      MAX_SHARE_MEMORY_SIZE /*smem*/>>>(d_input, d_weight, d_output, d_clock_cycles_mem, d_clock_cycles_compute, d_clock_cycles_extra, i == (NUM_TRIALS / 2));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    all_elapsed_ms[i] = elapsed_ms;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Process the elapsed times
  float min_elapsed_ms = *std::min_element(all_elapsed_ms.begin(), all_elapsed_ms.end());
  float max_elapsed_ms = *std::max_element(all_elapsed_ms.begin(), all_elapsed_ms.end());
  float average_elapsed_ms = std::accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f) / NUM_TRIALS;
  float std_elapsed_ms = std::sqrt(std::accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f, [average_elapsed_ms](float acc, float x) { return acc + (x - average_elapsed_ms) * (x - average_elapsed_ms); }) / NUM_TRIALS);
  printf("Min elapsed time: %f ms\n", min_elapsed_ms);
  printf("Max elapsed time: %f ms\n", max_elapsed_ms);
  printf("Average elapsed time: %f ms\n", average_elapsed_ms);
  printf("Standard deviation: %f ms (%.2f%%)\n", std_elapsed_ms, std_elapsed_ms / average_elapsed_ms * 100);
  // print all the elapsed times
  for (size_t i = 0; i < NUM_TRIALS; ++i) {
    printf("Elapsed time %zu: %f ms, ", i, all_elapsed_ms[i]);
  }
  printf("\n");

  // Output the output tensors to a file for verification
  cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < output_size / sizeof(bfloat16); ++i) {
    if (h_output[i] != static_cast<bfloat16>(REDUCTION_SIZE)) {
      printf("Error: h_output[%zu] = %f\n", i, float(h_output[i]));
      return 1;
    }
  }

  // Write the clock_cycles_mem and clock_cycles_compute to a file
  size_t *h_clock_cycles_mem = (size_t*)malloc(NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t));
  size_t *h_clock_cycles_compute = (size_t*)malloc(NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t));
  size_t *h_clock_cycles_extra = (size_t*)malloc(NUM_LAYERS * EXTRA_INFO_SIZE * sizeof(size_t));
  cudaMemcpy(h_clock_cycles_mem, d_clock_cycles_mem, NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_clock_cycles_compute, d_clock_cycles_compute, NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE) * sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_clock_cycles_extra, d_clock_cycles_extra, NUM_LAYERS * EXTRA_INFO_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);

  size_t compute = 0;
  size_t launch = 0;
  size_t wait = 0;
  size_t warmup_launch = 0;
  size_t warmup_wait = 0;
  size_t whole_kernel_time = 0;
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t main_offset = i * (REDUCTION_SIZE / K_TILE_SIZE);
    size_t extra_offset = i * EXTRA_INFO_SIZE;
    whole_kernel_time += h_clock_cycles_extra[extra_offset];
    warmup_wait += h_clock_cycles_extra[extra_offset + 2];
    warmup_launch += h_clock_cycles_extra[extra_offset + 1];
    size_t acc_launch = 0;
    size_t acc_wait = 0;
    size_t acc_compute = 0;
    for (int j = 0; j < FORLOOP_LENGTH; ++j) {
      if (j % 2 == 0) {
        acc_wait += h_clock_cycles_mem[main_offset + j];
      } else {
        acc_launch += h_clock_cycles_mem[main_offset + j];
      }
      acc_compute += h_clock_cycles_compute[main_offset + j];
    }
    launch += (acc_launch / (FORLOOP_LENGTH / 2));
    wait += (acc_wait / (FORLOOP_LENGTH / 2));
    compute += (acc_compute / FORLOOP_LENGTH);
  }
  whole_kernel_time /= NUM_LAYERS;
  compute /= NUM_LAYERS;
  warmup_wait /= NUM_LAYERS;
  warmup_launch /= NUM_LAYERS;
  launch /= NUM_LAYERS;
  wait /= NUM_LAYERS;
  if constexpr (USE_PIPELINE) {
    printf("Using pipeline\n");
  } else {
    printf("Not using pipeline\n");
  }
  printf("Reporting average clock cycles for each layer:\n");
  printf("whole_kernel_time = %zu\n", whole_kernel_time);
  printf("compute = %zu\n", compute);
  printf("warmup_wait = %zu\n", warmup_wait);
  printf("warmup_launch = %zu\n", warmup_launch);
  printf("launch = %zu\n", launch);
  printf("wait = %zu\n", wait);

  for (size_t i = 0; i < NUM_LAYERS * (REDUCTION_SIZE / K_TILE_SIZE); ++i) {
    printf("clock_cycles_mem[%zu] = %zu\n", i, h_clock_cycles_mem[i]);
    printf("clock_cycles_compute[%zu] = %zu\n", i, h_clock_cycles_compute[i]);
  }
  free(h_clock_cycles_mem);
  free(h_clock_cycles_compute);
  free(h_clock_cycles_extra);
  cudaFree(buffer);
  cudaFree(d_clock_cycles_mem);
  cudaFree(d_clock_cycles_compute);
  cudaFree(d_clock_cycles_extra);
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}