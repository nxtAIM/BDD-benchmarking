# BDD Dataset Benchmark

This repository contains the code used for benchmarking the BDD dataset. We evaluate dataset access performance using direct file system access with torchcodec, alongside h5py, PyArrow, and NVIDIA DALI.

## Libraries Used

- **File System Access with TorchCodec**  
  [torchcodec GitHub](https://github.com/pytorch/torchcodec)

- **HDF5 File Handling with h5py**  
  [h5py Documentation](https://docs.h5py.org/en/latest/quick.html)

- **Data Loading with NVIDIA DALI**  
  [DALI PyTorch Example](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-basic_example.html)

- **Columnar Data Format with PyArrow**  
  [PyArrow Documentation](https://arrow.apache.org/docs/python/index.html)

## Data Preprocessing Steps

1. **Weather and Time Filtering**:  
   We select only videos recorded during daytime and under clear weather conditions.

2. **Frame Extraction**:  
   From each selected video, every 5th frame is extracted.

3. **Data Shuffling**:  
   The extracted frames are shuffled randomly.

4. **Sliding Window Approach**:  
   A sliding window is used to return 4 frames per iteration, ensuring no frame repetition within a single epoch.

After the preprocessing steps, the dataset size is reduced from 100k videos to around 2k videos, resulting in a total of 396,839 frames.

## Benchmarking

The benchmarking is conducted using 4 GPUs for parallel processing.

## Results

You can find the benchmarking results in this [link](https://docs.google.com/presentation/d/1ac-wiRC_e3ALKewhmUmjJcYsixzFNvYad9mwggb19iQ/edit?usp=sharing)

Note that the results for PyArrow are not included, as loading the entire dataset took several hours.
