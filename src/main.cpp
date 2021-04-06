#include <stdlib.h>
#include <stdio.h>
#include "tree.h"
#include "gpu-stm-tree-v27/bpt-gpustm.h"
#include "gpu-stm-tree-v35/bpt-gpustm.h"
#include "gpu-stm-tree-v38/bpt-gpustm.h"
#include "gpu-stm-tree-v39/bpt-gpustm.h"
#include <unistd.h>


int main(int argc, char *argv[]){
   
    GPU_BTree *tree;
 

    bool testFlag = false;

    int opt; 
    opterr = 0;
    while ((opt = getopt(argc, argv, "ts:i:r:v:"))!=-1) {
        switch(opt) {
            case 't':
                testFlag = ~testFlag;
                break;
            case 'i':
                tree->prepareInput(std::string(optarg));
                tree->gpu_work();
                if (testFlag) tree->test();
                break;
            case 'r':
                tree->prepareInput(std::string(optarg), (mission_t)RANGE);
                tree->gpu_work_range();
                if (testFlag) tree->test_range();
                break;
            case 'v':
                //std::cout <<optarg<<std::endl;
                switch (atoi(optarg)) {
                    case 27:
                        tree = new gpu_stm_nsp_27::GPU_STM_Tree();
                        tree->prepareGPU();
                        break;
                    case 35:
                        tree = new gpu_stm_nsp_35::GPU_STM_Tree();
                        tree->prepareGPU();
                        break;
                    case 38:
                        tree = new gpu_stm_nsp_38::GPU_STM_Tree();
                        tree->prepareGPU();
                        break;
                    case 39:
                        tree = new gpu_stm_nsp_39::GPU_STM_Tree();
                        tree->prepareGPU();
                        break;
                }
                break;
            case 's':
                tree->prepareInput(std::string(optarg));
                tree->gpu_work_sp();
                if (testFlag) tree->test();
                break;
                
        }
    }
    

/*
    for (int i=1; i<argc; i++) {
        tree.prepareInput(std::string(argv[i]));
        tree.gpu_work();
        if (testFlag) tree.test();
    }
*/
}



