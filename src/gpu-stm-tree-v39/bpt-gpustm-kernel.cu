#include "../global_conf.h"
#include "bpt-gpustm.h"
#include "bpt-gpustm-kernel-tx.cuh"
#include "bpt-gpustm-kernel-tree.cuh"
#include "bpt-gpustm-kernel-def.h"
#include "../cuda_utils.h"

#include "cub/cub.cuh" 
using namespace std;
using namespace gpu_stm_nsp_39; 


namespace gpu_stm_nsp_39{
__global__ void tx_insert_kernel(g_tree_t g_tree, 
        g_data_t g_data,
        g_tx_t g_tx,
        int size
        ) {

    const int tx_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE;
    const int tx_thread_id = threadIdx.x % WARPSIZE;
    const int tx_id_in_block = threadIdx.x / WARPSIZE;
    if (tx_id * insert_queue_length >= size) return; //把一整个warp都在size外面的扔掉.


    //每WARPSIZE个THREAD共用一个wrtset,  一个wrtset最多装wrtSetSize个记录， 包括location, version, key_arr, val_arr。
    
    //every ORDER threads use one rdset_s and wrtset_s 
    __shared__ rdset_t rdset_ss[Block_Dim/WARPSIZE];   
    rdset_t &rdset_s = rdset_ss[tx_id_in_block];
    __shared__ wrtset_t wrtset_ss[Block_Dim/WARPSIZE];
    wrtset_t &wrtset_s = wrtset_ss[tx_id_in_block];
    
    wrtset_content_t wrtset;

    version_t root_ver;
    offset_t local_root;

//=============================================================================== 
    
    for (int l_i=0; l_i<insert_queue_length;l_i++) {

        
        key_t target_key;       //每个线程保有自己处理的任务
        mission_t mission;      //每个线程保有自己处理的任务
        offset_t target_val;     //每个线程保有自己处理的任务
        ans_t * ans;            //每个线程保有自己处理的任务
        const int mission_id = tx_id * insert_queue_length + l_i; 
       
        if (mission_id >= size) return;
        
        
        target_key = g_data.g_keys[mission_id];
        mission = g_data.g_mission[mission_id];
        target_val = g_data.g_vals[mission_id];
        ans = &(g_data.g_ans[mission_id]);   

#ifdef ABORT_COUNT 
        g_data.g_roll_back_count[mission_id].roll_back_times = -1;
        g_data.g_roll_back_count[mission_id].traversal_steps = 0;
        g_data.g_roll_back_count[mission_id].leaf_traversal_steps = 0;
        __syncwarp();
#endif 

        
        //每个thread拥有自己负责的那个任务的相关的值
        offset_t tmpCurNodeId;     
        key_t nodeInfo;                      
        key_t key;    
        offset_t val;
        static __shared__ offset_t searched_value_ss[Block_Dim/WARPSIZE];    
        static __shared__ char target_thread_ss[Block_Dim/WARPSIZE];          

        offset_t &searched_value_s = searched_value_ss[tx_id_in_block];         
        char &target_thread_s = target_thread_ss[tx_id_in_block];

        
        //record last layer 
        offset_t lastNodeId;
        key_t last_node_info;
        key_t last_key;       //记录 last_key 
        offset_t last_val;    //记录 last_val
        char last_target_thread; 
        

 
TX_START:



#ifdef ABORT_COUNT
        __syncwarp();
        if (tx_thread_id == 0)
            g_data.g_roll_back_count[mission_id].roll_back_times++;
        __syncwarp();
#endif  



        __syncwarp();

 

        //默认两个set内容为空
        rdset_s.loc[0] = -1;
        rdset_s.loc[1] = -1;
        rdset_s.offset = 0;
        wrtset_s.size = 0;
        __syncwarp();

        if (tx_read_root()==false) { goto  TX_START; }
        
        if (local_root == -1) {
            
            if (create_new_tree() == false) goto TX_START;
            if (tx_commit(true, NULL) == false) goto TX_START;
            goto TX_START;
        }
  
        //__syncwarp();
        //if (tx_thread_id == 0){
        //    printf("root: %d, mission_id, %d\n", local_root, mission_id);
        //}
        __syncwarp();


        // 初始化searched_value
        searched_value_s = local_root;
        
        //初始化tmpCurNodeId
        tmpCurNodeId = -1;
        __syncwarp();

        
        //search
        do {
/*{{{*/

            __syncwarp();
            lastNodeId = tmpCurNodeId;
            last_node_info = nodeInfo; 
            last_key = key;
            last_val = val;
            last_target_thread = target_thread_s;
           
            tmpCurNodeId = searched_value_s;
            __syncwarp();

#ifdef ABORT_COUNT 
            if (tx_thread_id == 0)
                g_data.g_roll_back_count[mission_id].traversal_steps++; 
#endif 
            // read node
            if (read_node(tmpCurNodeId, key, val, nodeInfo)==false) { 
                goto TX_START;
            }
            
            __syncwarp();
            
 //=========提前分裂 situation one 提前分裂满的internal node
/*{{{*/
            if ((getNodeSize(nodeInfo)==ORDER-1)&&(!isLeaf(nodeInfo))) {
                
                int up_key = -1;
                //Internal node split 提前分裂
                if (node_splitting(tmpCurNodeId, key, val, nodeInfo, up_key)==false) goto TX_START;
                __syncwarp();
                char place_for_new_loc;
                if (lastNodeId == -1){
                    if (insert_new_root(up_key, tmpCurNodeId, getNodeLevel(nodeInfo)+1, place_for_new_loc)==false) goto TX_START;
                    if (tx_commit(true, place_for_new_loc)==false) goto TX_START;
                }
                else {
                    if (insert_node(lastNodeId, up_key, last_key, last_val, last_target_thread, last_node_info, place_for_new_loc)==false) goto TX_START;
                    if (tx_commit(false, place_for_new_loc)==false) goto TX_START;
                }
                //tx_commit
                goto TX_START;
            }
            __syncwarp();
/*}}}*/
            //get target 
            get_target(key, val, target_key, searched_value_s, target_thread_s ); 
            
            __syncwarp();
            
            
               
            /*}}}*/
        }while (!isLeaf(nodeInfo)); 
       //==================================SEARCH DONE==================================== 


        __syncwarp();
     

        *ans = -1;
     
        __syncwarp();
        //判断是否找到结果。
        //如果能找到结果，结果必然是在target_thread_s-1的位置。
        if (key == target_key) { 
            *ans = val;
        }
        //有结果的赋值为结果，没有结果的一律-1
       
        //test
       
        __syncwarp();
       
        if (mission == GET) {
            continue;
        }

        __syncwarp();
        
        int up_key = -1; 

        if (insert_leaf(tmpCurNodeId, nodeInfo, target_key, target_val, key, val, target_thread_s, up_key) == false) {
            goto TX_START;
        }
        __syncwarp();
        if (up_key!=-1) {
            char place_for_new_loc;
            if (lastNodeId == -1) {
                if (insert_new_root(up_key, tmpCurNodeId,getNodeLevel(nodeInfo)+1, place_for_new_loc) == false) goto TX_START;
                if (tx_commit(true, place_for_new_loc)==false) goto TX_START;
            }
            else {
                if (insert_node(lastNodeId, up_key, last_key, last_val, last_target_thread, last_node_info, place_for_new_loc)==false) goto TX_START; 
                if (tx_commit(false, place_for_new_loc)==false) goto TX_START;
            }
        }
        else {
            if (tx_commit(false, NULL)==false) goto TX_START;
        }
        
        //if (tx_thread_id==0) printf("success return %d\n", target_key);
        __syncwarp();
        
        
        
        continue; 





    }
    
 
    

    return;

}





void GPU_STM_Tree::launchKernel(int size, bool special, Time_Measure &t) {

    
    int requests_per_block = Block_Dim / WARPSIZE * insert_queue_length ; 
    int grid_dim = (size + requests_per_block - 1) / requests_per_block;
    //cout<<"grid_dim1: "<<grid_dim1<<endl;
    //cout<<"grid_dim2: "<<grid_dim2<<endl;
    //cout<<"working size:      "<<size<<endl;
    t.gpuTimeStart();


    if (grid_dim != 0)  {

        tx_insert_kernel<<<grid_dim, Block_Dim>>>(g_tree, g_data, g_tx, size);
        
       // cudaError_t error = cudaGetLastError();
       // printf("cuda error: %s\n",cudaGetErrorString(error));
    }
    t.gpuTimeEnd();
    
}

__global__ void preprocessKernel(g_tree_t g_tree, g_tx_t g_tx){
    
    *(g_tree.g_root) = -1;
    *(g_tree.g_tree_size) = 0;

    g_tx.g_lock_table[MAX_NODE_NUM-1] = 0;    //root lock 初始化

}





void GPU_STM_Tree::preprocess() {
    dim3 grid_dim(1);
    dim3 block_dim(1);
    preprocessKernel<<<grid_dim, block_dim>>>(g_tree, g_tx);
    
    //for test
#if 0 
    cout<<"g_tree: "<<endl;
    cout<<(offset_t*)g_tree.g_root<<endl;
    cout<<(offset_t*)g_tree.g_tree_size<<endl;
    cout<<(keyArr_t*)g_tree.g_key_section<<endl;
    cout<<(valArr_t*)g_tree.g_val_section<<endl;
    cout<<"g_data: "<<endl;
    cout<<g_data.g_keys<<endl;
    cout<<g_data.g_vals<<endl;
    cout<<g_data.g_mission<<endl;
    cout<<g_data.g_ans<<endl;
    cout<<"g_tx: "<<endl;
    //cout<<g_tx.g_wrtSet_location<<endl;
    //cout<<g_tx.g_wrtSet_version<<endl;
    //cout<<g_tx.g_wrtSet_content_key<<endl;
    //cout<<g_tx.g_wrtSet_content_val<<endl;
    cout<<(version_t*)g_tx.g_lock_table<<endl;
#endif


}


















}
