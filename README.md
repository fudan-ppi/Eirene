# GPU-STM for BPlusTree

## How to build 
```c
1. run ./build.sh 
2. make -jn 
```
## How to run 
../stm_test -v{version} -i ../dataset/input_data_{23,24,25,26}.txt -i ../dataset/{test_data_set}

or use script in ./evalutaion/


## Versions
## version 39 : STM-based GPU B+tree 
## version 38 : STM-based GPU B+tree  + Sort based Merge
## version 27 : Sort based Merge + Optimistic Strategy
## version 35 : Sort based Merge + Optimistic Strategy + Warp Reorg

