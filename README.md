# SYCL Hashing Algorithms

This repository contains hashing algorithms implemented using [SYCL](https://www.khronos.org/sycl/) which is a heterogeneous programming model based on standard C++.

The following hashing methods are currently available:
* sha256
* sha1 (unsecure)
* md2 (unsecure)
* md5 (unsecure)
* keccak (128 224 256 288 384 512)
* sha3 (224 256 384 512)
* blake2b

## Performance

Some functions were ported from CUDA implementations. Here's how they perform (the values are in GB/s):

| Function | Native CUDA | SYCL running on DPC++ CUDA (nvptx64) | SYCL on ComputeCPP CPU (spir64) | SYCL on DPC++ CPU (spir64_x86_64) | 
|----------|-------------|------------------------------|-----------------------------|-------------|
| keccak   | 12.43       | 21.71                        | 3.99                        |  3.88       |
| md5      | 11.90       | 19.61                        | 5.32                        |  0.313      |
| blake2b  | 11.89       | 18.63                        | 8.10                        |  3.74       |
| sha1     | 10.85       | 14.84                        | 3.48                        |  3.37       |
| sha256   | 11.06       | 13.61                        | 2.11                        |  2.90       |
| md2      | 3.94        | 3.28                         | 0.20                        |  0.112      |

Benchmark configuration: 
* block_size: 512 kiB
* n_blocks: 4*1536
* n_outbit: 128
* GPU: GTX 1660 Ti
* OS: rhel8.4
* CPU: 2x E5-2670 v2

### Remark
These are not the "best" settings as the optimum changes with the algorithm. The benchmarks measure the time to run 40 iterations, without copying the memory between the device and the host. In a real application, you could be memory bound.

## How to build

```bash
git clone https://github.com/Michoumichmich/SYCL-Hashing-Algorithms.git ; cd SYCL-Hashing-Algorithms;
mkdir build; cd build
CXX=<sycl_compiler> cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
This will build the library, and a demo executable. Running it will perform a benchmark on your CPU and CUDA device (if available).

You do not necessarily need to pass the `<sycl_compiler>` to cmake, it depends on the implementation you're using and its toolchain. 

## How to use

Let's assume you used this [script](https://github.com/Michoumichmich/oneAPI-setup-script) to setup the toolchain with CUDA support.

Here's a minimal example:
```C++
#include <sycl/sycl.hpp> // SYCL headers
#include "sycl_hash.hpp" // The headers
#include "tools/sycl_queue_helpers.hpp" // To make sycl queue
using namespace hash;

int main(){
    auto cuda_q = try_get_queue(cuda_selector{}); // create a queue on a cuda device and attach an exception handler
    
    constexpr int hash_size = get_block_size<method::sha256>();
    constexpr int n_blocks = 20; // amount of hash to do in parallel
    constexpr int item_size = 1024;
    
    byte input[n_blocks * item_size]; // get an array of 20 same-sized data items to hash;
    byte output[n_blocks * hash_size]; // reserve space for the output
    
    compute<method::sha256>(cuda_q, input, item_size, output, n_blocks); // do the computing
    
    return 0;
}
```
And, for clang build with 
```
-fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-cuda-nvidia-sycldevice -I<include_dir> <build_dir>/libsycl_hash.a
```
And your hash will run on the GPU.


# Sources
You may find [here](https://github.com/Michoumichmich/cuda-hashing-algos-with-benchmark) the fork of the original CUDA implementations with the benchmarks added.

# Tested implementations
* [Intel's clang](https://github.com/intel/llvm) with OpenCL on CPU (using Intel's driver) and [Codeplay's CUDA backend](https://www.codeplay.com/solutions/oneapi/for-cuda/)
* [hipSYCL](https://github.com/illuhad/hipSYCL) on macOS with the OpenMP backend
* [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home) you can build with `cmake .. -DComputeCpp_DIR=/path_to_computecpp -DCOMPUTECPP_BITCODE=spir64 -DCMAKE_BUILD_TYPE=Release`, Tested on the host device, `spir64` and `spirv64`. See [ComputeCpp SDK](https://github.com/codeplaysoftware/computecpp-sdk)


# Contact
You can send a mail to `0_torte.say at icloud dot com`

# Acknowledgements

This repository contains code written by Matt Zweil & The Mochimo Core Contributor Team. Please see the [files](https://github.com/mochimodev/cuda-hashing-algos) for their respective licences.

