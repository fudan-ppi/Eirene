#!/bin/bash

nvprof --devices 0 --kernels ::search_kernel:4 --metrics all --log-file query23_95_v13.prof ../stm_test -v13 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

