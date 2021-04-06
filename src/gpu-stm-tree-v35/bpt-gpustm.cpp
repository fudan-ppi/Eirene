#include <cuda_runtime.h>
#include "../cuda_utils.h"
#include <iostream>
#include "bpt-gpustm.h"
#include "../input-helper.h"
#include "../time_measure.cuh"
#include "memory.h"

using namespace std;
using namespace gpu_stm_nsp_35;



void GPU_STM_Tree::prepareGPU(){

    cout<<"tx size put: always 1"<<endl;
    cout<<"tx size get: "<<tx_size_get<<endl;
    cout<<"warp per block get: "<<Block_Dim_Get/WARPSIZE<<endl;
    cout<<"warp per block put: "<<Block_Dim_Put/WARPSIZE<<endl;
    //prepareGPU  这一部分是负责在GPU上申请空间。
    cudaSetDevice(gpu_start);
    prepareGPU_tree();
    prepareGPU_data(); 
    prepareGPU_tx(); 
    preprocess();   //preprocess on GPU
    //launchBuildIndexKernel();



}

void GPU_STM_Tree::prepareInput(string fileName){
    insert_count++;
    //read data
    
    dc.clearMissions();
    if (h_ans!=NULL) {
        free(h_ans);
    }
    if (h_ans1!=NULL) {
        free(h_ans1);
    }
    if (h_idx!=NULL) {
        free(h_idx);
    }
    int size = dc.readFile(fileName);
    h_ans = (ans_t *)malloc(sizeof(ans_t)*size);
    h_ans1 = (ans_t *)malloc(sizeof(ans_t)*size);
    h_idx = (int *)malloc(sizeof(int)*size);
#ifdef ABORT_COUNT 
    h_roll_back_count = (count_t *)malloc(sizeof(count_t)*size);
    
    int batch_num = (size + batch_size -1) / batch_size ;
    h_device_working_num = (int *)malloc(sizeof(int)*batch_num);
#endif 
}

void GPU_STM_Tree::prepareInput(string fileName, mission_t m){
    insert_count++;
    dc.clearMissions();
    if (h_ans!=NULL) {
        free(h_ans);
    }
    if (h_ans1!=NULL) {
        free(h_ans1);
    }
    if (h_idx!=NULL) {
        free(h_idx);
    }
    int size = dc.readFile(fileName, m);
    h_ans = (ans_t *)malloc(sizeof(ans_t)*size);
    h_ans1 = (ans_t *)malloc(sizeof(ans_t)*size);
    h_idx = (int *)malloc(sizeof(int)*size);
#ifdef ABORT_COUNT 
    h_roll_back_count = (count_t *)malloc(sizeof(count_t)*size);
    int batch_num = (size + batch_size -1) / batch_size ;
    h_device_working_num = (int *)malloc(sizeof(int)*batch_num);
#endif 
}


void GPU_STM_Tree::gpu_work_main(bool special){
/*{{{*/
    int size = dc.keys.size();
    int batch_num = size / batch_size;
    int i = 0;
   

    cout<<"bucket size\t"<<batch_size<<endl;
    cout<<"total size\t"<<size<<endl;
    //launchBuildIndexKernel();

    Time_Measure t;
    for (;i<batch_num;i++) {
#ifdef TRANSFER_TIME
        t.gpuTimeStart();
#endif         
        transferData(i*batch_size, batch_size);
#ifdef TRANSFER_TIME
        t.gpuTimeEnd();
#endif         
        
        
        launchKernelBefore(batch_size,t);
        
       //============== 
        
        launchKernel( batch_size, special, t);
       
        //============
        t.gpuTimeStart();
        
        launchKernelAfter(batch_size);
        
        t.gpuTimeEnd();
        //=============
#ifdef TRANSFER_TIME
        t.gpuTimeStart();
#endif         
        transferAns(i*batch_size, batch_size);
#ifdef TRANSFER_TIME
        t.gpuTimeEnd();
#endif         
    }
    if (size%batch_size!=0){

#ifdef TRANSFER_TIME
        t.gpuTimeStart();
#endif         
        transferData(i*batch_size, size-i*batch_size);
#ifdef TRANSFER_TIME
        t.gpuTimeEnd();
#endif         
        
        
        launchKernelBefore(size-i*batch_size,t);
        cout<<"launchKernelBefore"<<endl;
        
        launchKernel(size-i*batch_size, special, t);
        cout<<"launchKernel"<<endl;
        
        launchBuildIndexKernel();
        cout<<"launchBuildIndexKernel"<<endl;

        t.gpuTimeStart(); 
        launchKernelAfter(batch_size);
     //   cout<<"launchKernelAfter"<<endl;
        t.gpuTimeEnd();
        
#ifdef TRANSFER_TIME
        t.gpuTimeStart();
#endif         
        transferAns(i*batch_size, size-i*batch_size);
#ifdef TRANSFER_TIME
        t.gpuTimeEnd();
#endif         
    }
    //t.gpuTimeStart();
    //launchBuildIndexKernel();
    //t.gpuTimeEnd();
    rebuild_counter++;
    cout<<"insert count:"<<insert_count<<endl;
    cout<<"rebuild count: "<< rebuild_counter<<endl;
    cout<<"total gpu exec time: "<<t.getGPUTime()<<endl;




#ifdef ABORT_COUNT 

    long long total_roll_back_times = 0;
    long long total_traversal_steps = 0;
    long long total_leaf_traversal_steps = 0;
    int total_working_num = 0;
    batch_num = (size+batch_size-1) / batch_size;
    for (int i=0;i<batch_num;i++) {
        for (int j=0; j<h_device_working_num[i];j++) {
            total_roll_back_times += h_roll_back_count[i*batch_size+j].roll_back_times;
            total_traversal_steps += h_roll_back_count[i*batch_size+j].traversal_steps;
            total_leaf_traversal_steps += h_roll_back_count[i*batch_size+j].leaf_traversal_steps;
        }
        total_working_num += h_device_working_num[i];
    }
    
    cout<<"total working num:                           "<<total_working_num<<endl;
    cout<<"total roll back times: (Only Insert Kernel)  "<<total_roll_back_times<<endl;
    cout<<"total traversal_steps: (All Kernel including all steps) "<<total_traversal_steps<<endl;
    cout<<"total leaf traversal_steps: (All Kernel including all steps) "<<total_leaf_traversal_steps<<endl;
    cout<<"average traversal_steps:  "<<(double)((double)total_traversal_steps / (double)total_working_num)<<endl;

#ifdef ABORT_COUNT2
    long long histogram_all[62];
    long long histogram_leaf[62];
    // histogram[0]应该都为0,因为不存在traversal step为0的情况。
    // histogram[61]记录最后超过60个traversal step的数目
    // 
    for (int i=0; i<62; i++) {
        histogram_all[i]=0;
        histogram_leaf[i]=0;
    }
    for (int i=0; i<batch_num;i++) {
        for (int j=0; j<h_device_working_num[i];j++) {
            int step_all = h_roll_back_count[i*batch_size+j].traversal_steps;
            int step_leaf = h_roll_back_count[i*batch_size+j].leaf_traversal_steps;
            if (step_all > 60) {
                histogram_all[61]++; 
                //cout<<"ha"<<step_leaf<<"   "<<h_roll_back_count[i*batch_size+j].roll_back_times<<endl;
            }
            else {
                histogram_all[step_all]++; 
            }
            if (step_leaf > 60) {
                histogram_leaf[61]++;
            }
            else {
                histogram_leaf[step_leaf]++;
            }
        
        }
    }

    for (int i=0; i<62;i++) {
        cout<<"all_step_"<<i<<":    "<<histogram_all[i]<<endl;
    }
    cout<<"==============="<<endl;
    for (int i=0; i<62;i++) {
        cout<<"leaf_step_"<<i<<":   "<<histogram_leaf[i]<<endl;
    }
    

#endif      //ABORT_COUNT2



#endif      //ABORT_COUNT

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

    for (int i=0; i<dc.keys.size();i++) {
        h_ans1[h_idx[i] + batch_size * (i/batch_size)] = h_ans[i];
    }
    dc.test(h_ans1);


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
    indexArr_t *g_index_section=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&g_index_section,sizeof(keyArr_t)*MAX_NODE_NUM));

    next_nodeid_t *g_next_section=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&g_next_section,sizeof(next_nodeid_t)*MAX_NODE_NUM));
    offset_t * first_leaf=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&first_leaf,sizeof(offset_t)));
    
    leafHelp_t * leaf_section=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&leaf_section,sizeof(leafHelp_t)*MAX_NODE_NUM));
    offset_t * leaf_size=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&leaf_size, sizeof(offset_t)));
    offset_t * index_root=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&index_root,sizeof(offset_t)));

    offset_t * prefix_sum=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&prefix_sum,sizeof(offset_t)*MAX_NODE_NUM));
    g_tree = {root, tree_size, g_key_section, g_val_section,g_index_section,g_next_section,first_leaf,leaf_section,leaf_size,index_root,prefix_sum};
/*}}}*/
}

void GPU_STM_Tree::prepareGPU_data() {
/*{{{*/
    
    //malloc missions on GPU
    key_t *g_key = NULL;
    offset_t *g_val = NULL;
    mission_t *g_mission = NULL;
    ans_t *g_ans = NULL;
    int *g_idx = NULL;
    int *g_count = NULL;
    //int *h_num = NULL;
    int *g_num = NULL;
    int *g_boundary = NULL;
#ifdef ABORT_COUNT 
    count_t *g_roll_back_count = NULL;
#endif

    //g_data_phase_1
    CUDA_ERROR_HANDLER(cudaMalloc(&g_key, sizeof(key_t)*batch_size)); 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_idx, sizeof(int)*batch_size));
   
    g_data_phase_1.g_keys = g_key;
    g_data_phase_1.g_idx = g_idx;
    
    //g_data_phase_2
    CUDA_ERROR_HANDLER(cudaMalloc(&g_key, sizeof(key_t)*batch_size)); 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_idx, sizeof(int)*batch_size));
    
    g_data_phase_2.g_keys = g_key;
    g_data_phase_2.g_idx = g_idx;

    //g_data_phase_3
    CUDA_ERROR_HANDLER(cudaMalloc(&g_key, sizeof(key_t)*batch_size)); 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_count, sizeof(int)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_num, sizeof(int)));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_num, sizeof(int)));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_idx, sizeof(int)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_val, sizeof(offset_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_mission, sizeof(mission_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_ans, sizeof(ans_t)*batch_size));
    g_data_phase_3.g_keys =  g_key;
    g_data_phase_3.g_count = g_count;
    //g_data_phase_3.h_num = h_num;
    g_data_phase_3.g_num = g_num;
    g_data_phase_3.g_idx = g_idx;
    g_data_phase_3.g_vals = g_val;
    g_data_phase_3.g_mission = g_mission;
    g_data_phase_3.g_ans = g_ans;

    //g_data_phase_4
    CUDA_ERROR_HANDLER(cudaMalloc(&g_mission, sizeof(mission_t)*batch_size));
    g_data_phase_4.g_mission = g_mission;
    g_data_phase_4.g_idx = g_data_phase_1.g_idx;


    //init g_data 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_val, sizeof(offset_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_mission, sizeof(mission_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_ans, sizeof(ans_t)*batch_size));
    CUDA_ERROR_HANDLER(cudaMalloc(&g_boundary, sizeof(int)));
#ifdef ABORT_COUNT 
    CUDA_ERROR_HANDLER(cudaMalloc(&g_roll_back_count, sizeof(count_t)*batch_size ));
#endif
    g_data.g_keys = g_key;
    g_data.g_vals = g_val;
    g_data.g_mission = g_mission;
    g_data.g_ans = g_ans;
    g_data.g_idx = g_data_phase_3.g_idx;   //由于在用到g_data.g_idx的时候，g_data_phase_3的g_idx已经没有用了，所以为了省空间，所以进行了复用.
    g_data.g_boundary = g_boundary;
#ifdef ABORT_COUNT 
    g_data.g_roll_back_count = g_roll_back_count;
#endif
 
   
    //g_key = NULL;
    //g_idx = NULL;






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

    version_t *lock_table=NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&lock_table,sizeof(version_t) * MAX_NODE_NUM));

    g_lock={lock_table};
    /*}}}*/
}

void GPU_STM_Tree::transferData(int start, int size) {
/*{{{*/
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data_phase_1.g_keys, dc.keys.data()+start, size*sizeof(key_t), cudaMemcpyHostToDevice)); 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data_phase_3.g_vals, dc.vals.data()+start, size*sizeof(offset_t),cudaMemcpyHostToDevice)); 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(g_data_phase_3.g_mission, dc.mission.data()+start, size*sizeof(mission_t),cudaMemcpyHostToDevice)); 
/*}}}*/
}


void GPU_STM_Tree::transferAns(int start, int size) {
    /*{{{*/
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_ans+start, g_data_phase_3.g_ans, size*sizeof(ans_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_idx+start, g_data_phase_2.g_idx, size*sizeof(int), cudaMemcpyDeviceToHost));
#ifdef ABORT_COUNT 
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_roll_back_count+start, g_data.g_roll_back_count, size * sizeof(count_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_device_working_num+start/batch_size, g_data_phase_3.g_num, sizeof(int), cudaMemcpyDeviceToHost));
    //注意： 传回来的abort times是对于排好序的数据的abort times, 并且合并了相同的key的任务
    //也就是说，实际有效值应该为g_data_phase_3.g_num, 
#endif
/*}}}*/
}


