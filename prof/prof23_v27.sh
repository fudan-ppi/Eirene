#!/bin/bash

nvprof --devices 0 --kernels ::search_kernel:1 --metrics all --log-file query23_95_v27_query.prof ../stm_test -v27 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

