#include <cuda_runtime.h>
#include "../cuda_utils.h"
#include <iostream>
#include "bpt-gpustm.h"
#include "../input-helper.h"
#include "../time_measure.cuh"
#include "memory.h"

using namespace std;
using namespace gpu_stm_nsp_39;



void GPU_STM_Tree::prepareGPU(){

    cout<<"insert_queue_length: "<<insert_queue_length<<endl;
    cout<<"warp per block: "<<Block_Dim/WARPSIZE<<endl;
    //prepareGPU  这一部分是负责在GPU上申请空间。
    cudaSetDevice(gpu_start);
    prepareGPU_tree();
    prepareGPU_data(); 
    prepareGPU_tx(); 
    preprocess();   //preprocess on GPU




}

void GPU_STM_Tree::prepareInput(string fileName){
    //read data
    dc.clearMissions();
    if (h_ans!=NULL) {
        free(h_ans);
    }
    if (h_ans1!=NULL) {
        free(h_ans1);
    }
    int size = dc.readFile(fileName);
    h_ans = (ans_t *)malloc(sizeof(ans_t)*size);
    h_ans1 = (ans_t *)malloc(sizeof(ans_t)*size);
#ifdef ABORT_COUNT 
    h_roll_back_count = (count_t *)malloc(sizeof(count_t)*size);
    
    int batch_num = (size + batch_size -1) / batch_size ;
#endif 
}

void GPU_STM_Tree::prepareInput(string fileName, mission_t m){
    dc.clearMissions();
    if (h_ans!=NULL) {
        free(h_ans);
    }
    if (h_ans1!=NULL) {
        free(h_ans1);
    }
    int size = dc.readFile(fileName, m);
    h_ans = (ans_t *)malloc(sizeof(ans_t)*size);
    h_ans1 = (ans_t *)malloc(sizeof(ans_t)*size);
#ifdef ABORT_COUNT 
    h_roll_back_count = (count_t *)malloc(sizeof(count_t)*size);
    int batch_num = (size + batch_size -1) / batch_size ;
#endif 
}


void GPU_STM_Tree::gpu_work_main(bool special){
/*{{{*/
    int size = dc.keys.size();
    int batch_num = size / batch_size;
    int i = 0;
   

    cout<<"bucket size\t"<<batch_size<<endl;
    cout<<"total size\t"<<size<<endl;

    Time_Measure t;
    for (;i<batch_num;i++) {
        
        //cout<<"requests left: "<<size-i*batch_size<<endl;
        transferData(i*batch_size, batch_size);
        
        //============== 
        //t.gpuTimeStart();
        
        launchKernel( batch_size, special, t);
        
        //t.gpuTimeEnd();
               
        transferAns(i*batch_size, batch_size);
    }
    if (size%batch_size!=0){
        transferData(i*batch_size, size-i*batch_size);
        
               
        
        //t.gpuTimeStart();
        
        launchKernel(size-i*batch_size, special, t);
        
        //t.gpuTimeEnd();
        
               
        transferAns(i*batch_size, size-i*batch_size);
    }
    cout<<"total gpu exec time: "<<t.getGPUTime()<<endl;




#ifdef ABORT_COUNT 

    long long total_roll_back_times = 0;
    long long total_traversal_steps = 0;
    long long total_leaf_traversal_steps = 0;
    int total_working_num = size;
    for (int i=0;i<size;i++) {
            total_roll_back_times += h_roll_back_count[i].roll_back_times;
            total_traversal_steps += h_roll_back_count[i].traversal_steps;
            total_leaf_traversal_steps += h_roll_back_count[i].leaf_traversal_steps;
    }
    
    cout<<"total working num:                           "<<total_working_num<<endl;
    cout<<"total roll back times: (Only Insert Kernel)  "<<total_roll_back_times<<endl;
    cout<<"total traversal_steps: (All Kernel including all steps) "<<total_traversal_steps<<endl;
    cout<<"total leaf traversal_steps: (All Kernel including all steps) "<<total_leaf_traversal_steps<<endl;
    cout<<"average traversal_steps:  "<<(double)((double)total_traversal_steps / (double)total_working_num)<<endl;

#endif

/*}}}*/
}

void GPU_STM_Tree::gpu_work_sp(){
    gpu_work_main(true);
}
void GPU_STM_Tree::gpu_work(){
    gpu_work_main(false);
}

void GPU_STM_Tree::gpu_emulate_query_per_conflict(){

    




}

void GPU_STM_Tree::test(){

    dc.test(h_ans);

}



void GPU_STM_Tree::prepareGPU_tree() {
/*{{{*/
    
    offset_t *root = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&root, sizeof(offset_t)));
    offset_t *tree_size = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&tree_size, sizeof(offset_t)));

    //malloc key section and val section on GPU.
    keyArr_t *g_key_section = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&g_key_section, sizeof(keyArr_t)*MAX_NODE_NUM)); 
    valArr_t *g_val_section = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&g_val_section, sizeof(valArr_t)*MAX_NODE_NUM)); 

    g_tree = {root, tree_size, g_key_section, g_val_section};
/*}}}*/
}

void GPU_STM_Tree::prepareGPU_data() {
/*{{{*/
    
    //malloc missions on GPU
    key_t *g_key = NULL;
    offset_t *g_val = NULL;
    mission_t *g_mission = NULL;
    ans_t *g_ans = NULL;

#ifdef ABORT_COUNT 
    count_t *g_roll_back_count = NULL;
#endif

    CUDA_ERROR_HANDLER(cudaMalloc(&g_key, sizeof(key_t)*batch_size)); 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_val, sizeof(offset_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_mission, sizeof(mission_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_ans, sizeof(ans_t)*batch_size));
#ifdef ABORT_COUNT 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_roll_back_count, sizeof(count_t)*batch_size ));
#endif
    g_data.g_keys = g_key;
    g_data.g_vals = g_val;
    g_data.g_mission = g_mission;
    g_data.g_ans = g_ans;
#ifdef ABORT_COUNT 
    g_data.g_roll_back_count = g_roll_back_count;
#endif
 
   






        /*}}}*/
}

void GPU_STM_Tree::prepareGPU_tx() {
/*{{{*/
    
    //offset_t *g_wrtSet_location = NULL;
    //CUDA_ERROR_HANDLER(cudaMalloc(&g_wrtSet_location, sizeof(offset_t) * wrtSetSize * tx_num));
    
    //version_t *g_wrtSet_version = NULL;
    //CUDA_ERROR_HANDLER(cudaMalloc(&g_wrtSet_version, sizeof(version_t) * wrtSetSize * tx_num));

    //keyArr_t *g_wrtSet_content_key = NULL;
    //CUDA_ERROR_HANDLER(cudaMalloc(&g_wrtSet_content_key, sizeof(keyArr_t) * wrtSetSize * tx_num));

    //valArr_t *g_wrtSet_content_val = NULL;
    //CUDA_ERROR_HANDLER(cudaMalloc(&g_wrtSet_content_val, sizeof(valArr_t) * wrtSetSize * tx_num));

    version_t *g_lock_table = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&g_lock_table, sizeof(version_t) * MAX_NODE_NUM ));

    //g_tx = {g_wrtSet_location, g_wrtSet_version, g_wrtSet_content_key, g_wrtSet_content_val, g_lock_table}; 
    g_tx = {g_lock_table};
    /*}}}*/
}

void GPU_STM_Tree::transferData(int start, int size) {
/*{{{*/
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data.g_keys, dc.keys.data()+start, size*sizeof(key_t), cudaMemcpyHostToDevice)); 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data.g_vals, dc.vals.data()+start, size*sizeof(offset_t),cudaMemcpyHostToDevice)); 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data.g_mission, dc.mission.data()+start, size*sizeof(mission_t),cudaMemcpyHostToDevice)); 
/*}}}*/
}


void GPU_STM_Tree::transferAns(int start, int size) {
    /*{{{*/
    //cudaError_t error = cudaGetLastError();
    //printf("cuda error: %s\n",cudaGetErrorString(error));
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_ans+start, g_data.g_ans, size*sizeof(ans_t), cudaMemcpyDeviceToHost));
#ifdef ABORT_COUNT 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_roll_back_count+start, g_data.g_roll_back_count, size * sizeof(count_t), cudaMemcpyDeviceToHost));
#endif
/*}}}*/
}


