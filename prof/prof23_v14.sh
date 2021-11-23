#!/bin/bash

nvprof --devices 0 --kernels ::search_kernel:1 --metrics all --log-file query23_95_v14.prof ../stm_test -v14 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

