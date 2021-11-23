#!/bin/bash

nvprof --devices 0 --kernels ::tx_kernel:5 --events all --log-file query23_uniform_95_v1_largebatch_events.prof ../stm_test -v1 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

