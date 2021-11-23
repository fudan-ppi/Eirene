#!/bin/bash

nvprof --devices 0 --kernels ::tx_insert_kernel:5 --metrics all --log-file query23_uniform_95_v38_3.prof ../stm_test -v38 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

