#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:10 --metrics all --log-file query23_95_v17.prof ../stm_test -v17 -i ../dataset/input_data_23.txt -i ../dataset/23_uniform/query_95_8M.txt

