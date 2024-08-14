# Common Issues

`I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero.`

Run `for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done` to suppress this warning.