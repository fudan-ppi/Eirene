#!/bin/bash

nvprof --devices 0 --kernels ::tx_insert_kernel:10 --metrics all --log-file query23_zipf_95_v38.prof ../stm_test -v38 -i ../dataset/input_data_23.txt -i ../dataset/23_zipf/query_95_50M.txt

