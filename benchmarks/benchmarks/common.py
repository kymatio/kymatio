import torch


# Make sure we have the same random numbers for each run.
torch.manual_seed(0)

# This constant corresponds to the total number of samples per batch. It is
# shared across all benchmarks of scattering: 1D, 2D, and 3D.
SCATTERING_BENCHMARK_SIZE = 2 ** 21
