#!/bin/bash

nvprof --devices 0 --kernels ::tx_insert_kernel:3 --metrics all --log-file query23_95_v27_insert.prof ../stm_test -v27 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

