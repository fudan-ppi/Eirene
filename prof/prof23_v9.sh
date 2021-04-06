#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:2 --metrics all --log-file query23_95_v9.prof ../stm_test -v9 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

