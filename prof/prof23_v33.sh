#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:30 --metrics all --log-file query23_95_v33.prof ../stm_test -v33 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

