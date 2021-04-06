#!/bin/bash

nvprof --devices 0 --kernels gpu_stm_nsp_35::insert_kernel:1 --metrics all --log-file query23_95_v35.prof ../stm_test -v35 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

