#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:20 --metrics all --log-file query23_100_v1_no_writeback.prof ../stm_test -v1 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_100_50M.txt

