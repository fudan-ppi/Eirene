#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> 
#include "../src/tree.h"
#include "../src/gpu-stm-tree-v14/bpt-gpustm.h"


// tree数据的范围    [A .. B]        
// leafnode个数      leaf_node_num       #来自
// 输入数据input     data (data[0], data[1], ... ) 
// 计算 平均每多少个query会产生一个true conflict 和 平均每多少个query会产生一个false conflict.
// 
// 



int main(int argc, char *argv[]) {

    gpu_stm_nsp_14::GPU_STM_Tree *tree = new gpu_stm_nsp_14::GPU_STM_Tree();

    int opt;
    opterr = 0;
    while ((opt = getopt(argc, argv, "i:s:"))!=-1){
        switch (opt) {
            case 'i':
                tree->prepareInput(std::string(optarg));
                tree->gpu_work();
                break;
            case 's':
                tree->prepareInput(std::string(optarg));
                tree->gpu_emulate_query_per_conflict();
                break;
            
        }
    }

    


}
