#!/bin/bash

nvprof --devices 0 --kernels ::tx_insert_kernel:5 --analysis-metrics -o query23_uniform_95_v39_analysis.prof --log-file query23_uniform_95_v39_analysis-metrics.prof ../stm_test -v39 -i ../dataset/input_data_23.txt -i ../dataset/23_random/query_95_50M.txt

