#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:20 --metrics all --log-file query26_95_v1.prof ../stm_test -v1 -i ../dataset/input_data_26.txt -i ../dataset/26_uniform/query_95_8M.txt

