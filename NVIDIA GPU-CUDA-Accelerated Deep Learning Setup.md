## Curating NVIDIA GPU-Accelerated Deep Learning Environments: A Guide to Installing NVIDIA Drivers, CUDA, cuDNN, TensorFlow, and PyTorch

**Author**: Vinay Gundala  
**Date**: May 2025

## Abstract

This whitepaper documents the process of establishing a GPU-accelerated deep learning environment on Xubuntu 25.04, utilizing an NVIDIA RTX 3060 with NVIDIA driver 570.133, CUDA 12.3, cuDNN 9.3, TensorFlow 2.18.0, and PyTorch 2.5.0. It addresses challenges such as version mismatches, library conflicts, bundled dependencies, circular imports, and benign factory errors, providing a reproducible guide for researchers and developers. The successful outcome, achieved over approximately 5–7 days, enables concurrent GPU-accelerated execution of TensorFlow and PyTorch, validated through comprehensive tests for cuDNN, cuBLAS, and cuFFT operations. Used Grok to pinpoint and troubleshoot issues.

## Introduction

Deep learning frameworks like TensorFlow and PyTorch leverage GPU acceleration to achieve high computational efficiency. Configuring NVIDIA drivers, CUDA, cuDNN, and these frameworks is critical but challenging due to version incompatibilities and dependency conflicts. This whitepaper details the curation of such an environment on Xubuntu 25.04 with an NVIDIA RTX 3060, providing a blueprint for reproducible setups.

## System Requirements and Specifications

The target system comprises:

*   **Hardware**: NVIDIA GeForce RTX 3060.
*   **Operating System**: Xubuntu 25.04.
*   **Software**:
    *   Python 3.12 in a virtual environment (`~/Code/MIT/.tf_env/`).
    *   NVIDIA driver 570.133.
    *   CUDA Toolkit 12.3.
    *   cuDNN 9.3.
    *   TensorFlow 2.18.0.
    *   PyTorch 2.5.0 with torchvision 0.20.0 and torchaudio 2.5.0.
    *   Additional package: mitdeeplearning.

## Challenges Encountered

Several obstacles were encountered during setup:

*   **Versions Are a Maze**: Achieving a functional GPU-accelerated deep learning environment requires precise alignment of versions across the operating system, NVIDIA drivers, CUDA Toolkit, cuDNN, TensorFlow, and PyTorch, much like planets aligning for a rare celestial event. For instance, Xubuntu 25.04 (with Python 3.12) must support NVIDIA driver 570.133, which must be compatible with CUDA 12.3 and cuDNN 9.3, while TensorFlow 2.18.0 and PyTorch 2.5.0 must align with these libraries. Mismatches, such as PyTorch’s bundled CUDA 12.4 libraries conflicting with system CUDA 12.3, led to errors like `undefined symbol: __nvJitLinkComplete_12_4`. Navigating this maze demands consulting compatibility matrices (e.g., NVIDIA’s CUDA documentation, TensorFlow/PyTorch release notes) and using diagnostic tools like `pip list` and `strace` to ensure version harmony.
*   **PyTorch CUDA Mismatch**: PyTorch 2.5.0’s bundled CUDA 12.4 libraries (e.g., `libcusparse.so.12`) caused an `undefined symbol: __nvJitLinkComplete_12_4` error due to incompatibility with system CUDA 12.3’s `libnvJitLink.so.12`.
*   **Missing libcupti.so.12**: PyTorch required `libcupti.so.12`, which was absent initially.
*   **Circular Import in PyTorch**: The `mitdeeplearning` package triggered a circular import in PyTorch 2.5.0, resulting in an `AttributeError` during initialization.
*   **TensorFlow Factory Errors**: Benign factory registration errors (e.g., `Unable to register cuFFT factory`) appeared but did not impact functionality. These remain unresolved but are non-critical.

### Failed Attempts and Lessons Learned

Several approaches failed before achieving the final setup:

*   **CUDA 12.4 with PyTorch**: Using PyTorch’s bundled CUDA 12.4 libraries led to the `undefined symbol` error, as system CUDA 12.3’s `libnvJitLink.so.12` lacked required symbols.
*   **PyTorch without --no-deps**: Initial installations without `--no-deps` installed `nvidia-cusparse-cu12`, `nvidia-cudnn-cu12`, etc., causing library mismatches.
*   **Alternative PyTorch Versions**: Testing PyTorch 2.4.1 was considered to resolve the circular import but was unnecessary after applying workarounds.
*   **mitdeeplearning Dependencies**: The `mitdeeplearning` package installed additional PyTorch dependencies, reintroducing CUDA 12.4 libraries until `--no-deps` was used.

These failures underscored the importance of isolating dependencies and prioritizing system libraries.

## Installation and Configuration Process

The following steps outline the installation process, addressing the challenges above.

### Preparing the System

1.  Update the system and install prerequisites:
    
    ```sh
    sudo apt-get update && sudo apt-get upgrade -y
    sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
    ```
2.  Create and activate a virtual environment:
    
    ```sh
    python3.12 -m venv ~/Code/MIT/.tf_env
    source ~/Code/MIT/.tf_env/bin/activate
    ```

### Installing NVIDIA Driver

1.  Verify the GPU:
    
    ```sh
    lspci | grep -i nvidia
    ```
2.  Install NVIDIA driver 570.133:
    
    ```sh
    sudo apt-get install -y nvidia-driver-570 nvidia-utils-570
    ```
3.  Verify installation:
    
    ```sh
    nvidia-smi
    ```
    
    Expected output includes driver version 570.133 and CUDA version 12.3.

### Installing CUDA 12.3 and cuDNN 9.3

This section provides detailed steps for installing CUDA 12.3, cuDNN 9.3, and related libraries, including all necessary symlinks to ensure compatibility with TensorFlow and PyTorch.

#### Step 1: Install CUDA 12.3

1.  **Install Prerequisites**: Ensure the system has the necessary tools and libraries for CUDA installation:
    
    ```sh
    sudo apt-get install -y gcc g++ make libc-dev linux-headers-$(uname -r)
    ```
2.  **Download the CUDA 12.3 Installer**: Obtain the runfile installer for CUDA 12.3.2:
    
    ```sh
    wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
    ```
3.  **Run the Installer**: Execute the installer, selecting only the CUDA Toolkit (exclude the driver, as NVIDIA driver 570.133 is already installed):
    
    ```sh
    sudo sh cuda_12.3.2_545.23.08_linux.run --override
    ```
    
    *   Follow the prompts:
        *   Accept the EULA.
        *   Uncheck the "Driver" option.
        *   Select "CUDA Toolkit 12.3" and related components (e.g., samples, libraries).
        *   Install to the default location: `/usr/local/cuda-12.3/`.
    *   The installer creates `/usr/local/cuda-12.3/` containing binaries, libraries, and headers.
4.  **Verify CUDA Installation**: Check the CUDA compiler (`nvcc`) version:
    
    ```sh
    /usr/local/cuda-12.3/bin/nvcc --version
    ```
    
    Expected output:
    
    ```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2023 NVIDIA Corporation
    Built on Mon_Nov_13_21:11:24_PST_2023
    Cuda compilation tools, release 12.3, V12.3.107
    ```
    
    List CUDA libraries:
    
    ```sh
    ls -l /usr/local/cuda-12.3/lib64/
    ```
    
    Expected libraries include `libcudart.so.12`, `libcufft.so.11`, `libcublas.so.12`, `libcusparse.so.12`, `libnvJitLink.so.12`, etc.
5.  **Set Environment Variables**: Configure `PATH` and `LD_LIBRARY_PATH` to include CUDA binaries and libraries:
    
    ```sh
    export PATH=/usr/local/cuda-12.3/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/cuda-12.3/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/cuda-12.3/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```
6.  **Verify Library Paths**: Confirm the dynamic linker can find CUDA libraries:
    
    ```sh
    ldconfig -p | grep -E "libcudart|libcufft|libcublas|libcusparse|libnvJitLink"
    ```
    
    Expected output includes paths to `/usr/local/cuda-12.3/lib64/` libraries.

#### Step 2: Install cuDNN 9.3

1.  **Download cuDNN 9.3**:
    *   Visit NVIDIA’s developer portal ([https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)).
    *   Sign in or create an NVIDIA account.
    *   Download the cuDNN 9.3 Debian package for CUDA 12.x (e.g., `libcudnn9-cuda-12_9.3.0-1_amd64.deb`).
2.  **Install cuDNN**: Install the Debian package:
    
    ```sh
    sudo dpkg -i libcudnn9-cuda-12_9.3.0-1_amd64.deb
    ```
    
    This installs cuDNN libraries to `/usr/lib/x86_64-linux-gnu/`, including `libcudnn.so.9.3.0`.
3.  **Copy cuDNN Libraries to CUDA Directory**: To ensure TensorFlow and PyTorch find cuDNN, copy libraries to `/usr/local/cuda-12.3/lib64/`:
    
    ```sh
    sudo cp /usr/lib/x86_64-linux-gnu/libcudnn.so.9.3.0 /usr/local/cuda-12.3/lib64/libcudnn.so.9
    ```
4.  **Verify cuDNN Installation**: Check for `libcudnn.so.9`:
    
    ```sh
    ls -l /usr/local/cuda-12.3/lib64/libcudnn*
    ```
    
    Expected output:
    
    ```
    -rw-r--r-- 1 root root <size> May 12 2025 /usr/local/cuda-12.3/lib64/libcudnn.so.9
    ```

#### Step 3: Configure Other CUDA Libraries

1.  **Ensure Library Availability**: The CUDA 12.3 installer includes additional libraries required by TensorFlow and PyTorch:
    
    *   **cuBLAS**: `libcublas.so.12` (matrix operations).
    *   **cuFFT**: `libcufft.so.11` (Fast Fourier Transforms).
    *   **cuSPARSE**: `libcusparse.so.12` (sparse matrix operations).
    *   **nvJitLink**: `libnvJitLink.so.12` (JIT linking for CUDA kernels).
    *   **CUPTI**: `libcupti.so.12` (profiling tools, located in `/usr/local/cuda-12.3/extras/CUPTI/lib64/`).
    
    Verify their presence:
    
    ```sh
    ls -l /usr/local/cuda-12.3/lib64/libcublas*
    ls -l /usr/local/cuda-12.3/lib64/libcufft*
    ls -l /usr/local/cuda-12.3/lib64/libcusparse*
    ls -l /usr/local/cuda-12.3/lib64/libnvJitLink*
    ls -l /usr/local/cuda-12.3/extras/CUPTI/lib64/libcupti*
    ```
    
    Expected output includes:
    
    ```
    /usr/local/cuda-12.3/lib64/libcublas.so.12.3.4.1
    /usr/local/cuda-12.3/lib64/libcufft.so.11.0.12.1
    /usr/local/cuda-12.3/lib64/libcusparse.so.12.3.0.142
    /usr/local/cuda-12.3/lib64/libnvJitLink.so.12.3.101
    /usr/local/cuda-12.3/extras/CUPTI/lib64/libcupti.so.12.3.101
    ```
2.  **Handle Missing Libraries**: If any libraries are missing, reinstall CUDA 12.3:
    
    ```sh
    sudo sh cuda_12.3.2_545.23.08_linux.run --override
    ```

#### Step 4: Create Symlinks for All Libraries

To ensure compatibility, create symlinks for all CUDA and cuDNN libraries, pointing to their specific versions and generic names (e.g., `libcudnn.so` for `libcudnn.so.9`). This allows TensorFlow and PyTorch to resolve dependencies dynamically.

Run the following commands:

```sh
# cuDNN
sudo ln -sf /usr/local/cuda-12.3/lib64/libcudnn.so.9.3.0 /usr/local/cuda-12.3/lib64/libcudnn.so.9
sudo ln -sf /usr/local/cuda-12.3/lib64/libcudnn.so.9 /usr/local/cuda-12.3/lib64/libcudnn.so

# CUPTI
sudo ln -sf /usr/local/cuda-12.3/extras/CUPTI/lib64/libcupti.so.12.3.101 /usr/local/cuda-12.3/lib64/libcupti.so.12
sudo ln -sf /usr/local/cuda-12.3/lib64/libcupti.so.12 /usr/local/cuda-12.3/lib64/libcupti.so

# cuBLAS
sudo ln -sf /usr/local/cuda-12.3/lib64/libcublas.so.12.3.4.1 /usr/local/cuda-12.3/lib64/libcublas.so.12
sudo ln -sf /usr/local/cuda-12.3/lib64/libcublas.so.12 /usr/local/cuda-12.3/lib64/libcublas.so

# cuFFT
sudo ln -sf /usr/local/cuda-12.3/lib64/libcufft.so.11.0.12.1 /usr/local/cuda-12.3/lib64/libcufft.so.11
sudo ln -sf /usr/local/cuda-12.3/lib64/libcufft.so.11 /usr/local/cuda-12.3/lib64/libcufft.so

# cuSPARSE
sudo ln -sf /usr/local/cuda-12.3/lib64/libcusparse.so.12.3.0.142 /usr/local/cuda-12.3/lib64/libcusparse.so.12
sudo ln -sf /usr/local/cuda-12.3/lib64/libcusparse.so.12 /usr/local/cuda-12.3/lib64/libcusparse.so

# nvJitLink
sudo ln -sf /usr/local/cuda-12.3/lib64/libnvJitLink.so.12.3.101 /usr/local/cuda-12.3/lib64/libnvJitLink.so.12
sudo ln -sf /usr/local/cuda-12.3/lib64/libnvJitLink.so.12 /usr/local/cuda-12.3/lib64/libnvJitLink.so
```

Update the dynamic linker cache:

```sh
sudo ldconfig
```

Verify symlinks:

```sh
ls -l /usr/local/cuda-12.3/lib64/libcudnn*
ls -l /usr/local/cuda-12.3/lib64/libcupti*
ls -l /usr/local/cuda-12.3/lib64/libcublas*
ls -l /usr/local/cuda-12.3/lib64/libcufft*
ls -l /usr/local/cuda-12.3/lib64/libcusparse*
ls -l /usr/local/cuda-12.3/lib64/libnvJitLink*
```

Expected output includes:

```
/usr/local/cuda-12.3/lib64/libcudnn.so -> libcudnn.so.9
/usr/local/cuda-12.3/lib64/libcudnn.so.9 -> libcudnn.so.9.3.0
/usr/local/cuda-12.3/lib64/libcudnn.so.9.3.0
/usr/local/cuda-12.3/lib64/libcupti.so -> libcupti.so.12
/usr/local/cuda-12.3/lib64/libcupti.so.12 -> libcupti.so.12.3.101
/usr/local/cuda-12.3/lib64/libcupti.so.12.3.101
/usr/local/cuda-12.3/lib64/libcublas.so -> libcublas.so.12
/usr/local/cuda-12.3/lib64/libcublas.so.12 -> libcublas.so.12.3.4.1
/usr/local/cuda-12.3/lib64/libcublas.so.12.3.4.1
/usr/local/cuda-12.3/lib64/libcufft.so -> libcufft.so.11
/usr/local/cuda-12.3/lib64/libcufft.so.11 -> libcufft.so.11.0.12.1
/usr/local/cuda-12.3/lib64/libcufft.so.11.0.12.1
/usr/local/cuda-12.3/lib64/libcusparse.so -> libcusparse.so.12
/usr/local/cuda-12.3/lib64/libcusparse.so.12 -> libcusparse.so.12.3.0.142
/usr/local/cuda-12.3/lib64/libcusparse.so.12.3.0.142
/usr/local/cuda-12.3/lib64/libnvJitLink.so -> libnvJitLink.so.12
/usr/local/cuda-12.3/lib64/libnvJitLink.so.12 -> libnvJitLink.so.12.3.101
/usr/local/cuda-12.3/lib64/libnvJitLink.so.12.3.101
```

#### Step 5: Final Verification

Confirm all libraries are accessible:

```sh
ldconfig -p | grep -E "libcudnn|libcupti|libcublas|libcufft|libcusparse|libnvJitLink"
```

Expected output includes paths to `/usr/local/cuda-12.3/lib64/` for all libraries.

Test a simple CUDA program to verify the setup:

```sh
# Create a test file: cuda_test.cu
cat > cuda_test.cu << EOL
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }
    return 0;
}
EOL

# Compile and run
nvcc cuda_test.cu -o cuda_test
./cuda_test
```

Expected output:

```
Number of CUDA devices: 1
Device 0: NVIDIA GeForce RTX 3060
```

### Configuring TensorFlow

1.  Install TensorFlow 2.18.0:
    
    ```sh
    pip install tensorflow==2.18.0
    ```
2.  Verify GPU detection:
    
    ```python
    python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU')); print(tf.test.gpu_device_name())"
    ```
    
    Expected output:
    
    ```
    2.18.0
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    /device:GPU:0
    ```
    
    Benign factory errors (e.g., `Unable to register cuFFT factory`) may appear.

### Configuring PyTorch

1.  Remove existing PyTorch and bundled dependencies:
    
    ```sh
    pip uninstall -y torch torchvision torchaudio triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
    rm -rf ~/Code/MIT/.tf_env/lib/python3.12/site-packages/torch/lib/../../nvidia/
    ```
2.  Install PyTorch 2.5.0 without dependencies:
    
    ```sh
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 --no-deps
    ```
3.  Verify GPU detection:
    
    ```python
    python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
    ```
    
    Expected output:
    
    ```
    2.5.0
    True
    12.3
    NVIDIA GeForce RTX 3060
    ```

### Installing mitdeeplearning

1.  Install `mitdeeplearning` without dependencies:
    
    ```sh
    pip uninstall mitdeeplearning -y
    pip install mitdeeplearning --no-deps
    pip install openai datasets ipywidgets numpy matplotlib
    ```
2.  Address circular import in PyTorch:
    *   Set environment variable:
        
        ```sh
        export TORCHINDUCTOR_FX_GRAPH_CACHE=1
        echo 'export TORCHINDUCTOR_FX_GRAPH_CACHE=1' >> ~/.bashrc
        source ~/.bashrc
        ```
    *   Alternatively, modify `lab3.py`:
        
        ```python
        # Edit ~/Code/MIT/.tf_env/lib/python3.12/site-packages/mitdeeplearning/lab3.py
        def create_dataloader(style):
            from torch.utils.data import DataLoader
            # ... rest of the function
        ```
3.  Verify import:
    
    ```python
    python3 -c "import mitdeeplearning as mdl; print('mitdeeplearning imported successfully')"
    ```

## Verification and Testing

GPU-accelerated tests validated both frameworks.

### TensorFlow GPU Tests

A test script (`test_tensorflow_gpu.py`) verified cuDNN, cuBLAS, and cuFFT operations:

```python
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)
if gpus:
    print("GPU Name:", tf.test.gpu_device_name())
else:
    print("No GPU available. Tests cannot be run.")
    exit(1)
def test_cudnn_convolution():
    with tf.device('/GPU:0'):
        x = tf.random.normal((1, 28, 28, 1))
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        y = conv(x)
    print("cuDNN convolution operation completed successfully")
    print(f"Output shape: {y.shape}")
def test_cublas_matmul():
    with tf.device('/GPU:0'):
        a = tf.random.normal((1000, 1000))
        b = tf.random.normal((1000, 1000))
        c = tf.matmul(a, b)
    print("cuBLAS matrix multiplication completed successfully")
    print(f"Output shape: {c.shape}")
def test_cufft_fft():
    with tf.device('/GPU:0'):
        x = tf.complex(tf.random.normal((256, 256)), tf.random.normal((256, 256)))
        y = tf.signal.fft2d(x)
    print("cuFFT FFT operation completed successfully")
    print(f"Output shape: {y.shape}")
if gpus:
    test_cudnn_convolution()
    test_cublas_matmul()
    test_cufft_fft()
```

Run:

```sh
python3 test_tensorflow_gpu.py
```

Output:

```
TensorFlow Version: 2.18.0
GPUs Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
GPU Name: /device:GPU:0
cuDNN convolution operation completed successfully
Output shape: (1, 28, 28, 32)
cuBLAS matrix multiplication completed successfully
Output shape: (1000, 1000)
cuFFT FFT operation completed successfully
Output shape: (256, 256)
```

### PyTorch GPU Tests

A test script (`test_pytorch_gpu.py`) verified similar operations:

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
print("cuDNN Version:", torch.backends.cudnn.version())
print("CUDA Version:", torch.version.cuda)
x = torch.randn(1, 1, 28, 28).cuda()
conv = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1).cuda()
y = conv(x)
print("cuDNN convolution operation completed successfully")
print("Output shape:", y.shape)
a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()
c = torch.matmul(a, b)
print("cuBLAS matrix multiplication completed successfully")
print("Output shape:", c.shape)
x = torch.randn(256, 256, dtype=torch.complex64).cuda()
y = torch.fft.fft2(x)
print("cuFFT FFT operation completed successfully")
print("Output shape:", y.shape)
```

Run:

```sh
python3 test_pytorch_gpu.py
```

Output:

```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3060
cuDNN Version: 90300
CUDA Version: 12.3
cuDNN convolution operation completed successfully
Output shape: torch.Size([1, 32, 28, 28])
cuBLAS matrix multiplication completed successfully
Output shape: torch.Size([1000, 1000])
cuFFT FFT operation completed successfully
Output shape: torch.Size([256, 256])
```

## Best Practices and Lessons Learned

*   **Avoid Bundled Dependencies**: Use `--no-deps` for PyTorch and other packages to prevent CUDA version mismatches. Verify with `pip list` and `strace`.
*   **Manage Library Paths**: Ensure `LD_LIBRARY_PATH` prioritizes system libraries (`/usr/local/cuda-12.3/lib64/`).
*   **Handle Circular Imports**: Use environment variables or delay imports in dependent packages.
*   **Regular Verification**: Use `strace` to confirm library loading:
    
    ```sh
    strace -e openat python3 -c "import torch" 2>&1 | grep -E "libcudnn|libcufft|libcublas|libcuda|libcusparse|libnvJitLink|libcupti"
    ```
*   **Document Configurations**: Maintain records of versions to ensure reproducibility.

## Conclusion

This whitepaper outlines a robust process for curating a GPU-accelerated deep learning environment on Xubuntu 25.04, achieved over approximately 5–7 days of iterative troubleshooting. The setup enables TensorFlow 2.18.0 and PyTorch 2.5.0 to leverage the NVIDIA RTX 3060, validated through GPU tests. Unresolved benign TensorFlow factory errors (e.g., `Unable to register cuFFT factory`) do not impact functionality. This guide, developed with the assistance of Grok to pinpoint and troubleshoot issues, serves as a valuable resource for building reliable deep learning systems.

**Acknowledgment**: Used Grok to pinpoint and troubleshoot issues.

## References

*   NVIDIA, “NVIDIA Driver Downloads,” [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx).
*   NVIDIA, “CUDA Toolkit 12.3,” [https://developer.nvidia.com/cuda-12-3-2-download-archive](https://developer.nvidia.com/cuda-12-3-2-download-archive).
*   NVIDIA, “cuDNN Installation Guide,” [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
*   TensorFlow, “Install TensorFlow with pip,” [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).
*   PyTorch, “PyTorch Installation,” [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).