#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:60 --metrics all --log-file query23_95_v2.prof ../stm_test -v2 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

