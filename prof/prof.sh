#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:20 --metrics all --log-file read95_v5 ../stm_test -v5 -i ../dataset/input_data_23.txt -i ../dataset/test_data_200M_all-read.txt 

