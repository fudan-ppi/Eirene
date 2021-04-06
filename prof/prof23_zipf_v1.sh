#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:20 --metrics all --log-file query23_zipf_95_v1.prof ../stm_test -v1 -i ../dataset/input_data_23.txt -i ../dataset/23_zipf/query_95_50M.txt

