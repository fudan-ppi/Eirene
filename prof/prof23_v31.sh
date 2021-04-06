#!/bin/bash

nvprof --devices 0 --kernels ::query_kernel:1 --metrics all --log-file query23_95_v31_query.prof ../stm_test -v31 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

