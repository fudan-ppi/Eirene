#!/bin/bash

nvprof --devices 0 --kernels ::query_kernel:1 --metrics all --log-file query23_95_v24.prof ../stm_test -v24 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

