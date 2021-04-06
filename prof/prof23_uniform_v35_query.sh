#!/bin/bash

nvprof --devices 0 --kernels ::query_kernel:2 --metrics all --log-file query23_95_uniform_v35_query.prof ../stm_test -v35 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

