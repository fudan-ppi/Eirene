#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:20 --metrics all --log-file query23_90_v21.prof ../stm_test -v21 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

