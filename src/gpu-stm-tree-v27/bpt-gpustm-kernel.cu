#include "../global_conf.h"
#include "bpt-gpustm.h"
#include "bpt-gpustm-kernel-tx.cuh"
#include "bpt-gpustm-kernel-tree.cuh"
#include "bpt-gpustm-kernel-def.h"
#include "bpt-gpustm-kernel-aux.cuh"
#include "bpt-gpustm-kernel-radical-search.cuh"
#include "../cuda_utils.h"

#include "cub/cub.cuh" 
using namespace std;
using namespace gpu_stm_nsp_27; 


#define Radical_Max_Times 100 
namespace gpu_stm_nsp_27{
__global__ void tx_insert_kernel(g_tree_t g_tree, 
        g_data_t g_data,
        g_tx_t g_tx,
        int boundary,
        int maxSize
        ) {
    
    const int size = (boundary * tx_size_get < maxSize)?boundary*tx_size_get:maxSize;
    const int tx_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE;
    const int tx_thread_id = threadIdx.x % WARPSIZE;
    const int tx_id_in_block = threadIdx.x / WARPSIZE;
    if (tx_id * tx_size_get >= size) return; //把一整个warp都在size外面的扔掉.


    //每WARPSIZE个THREAD共用一个wrtset,  一个wrtset最多装wrtSetSize个记录， 包括location, version, key_arr, val_arr。
    
    //every ORDER threads use one rdset_s and wrtset_s 
    __shared__ rdset_t rdset_ss[Block_Dim_Put/WARPSIZE];   
    rdset_t &rdset_s = rdset_ss[tx_id_in_block];
    __shared__ wrtset_t wrtset_ss[Block_Dim_Put/WARPSIZE];
    wrtset_t &wrtset_s = wrtset_ss[tx_id_in_block];
    
    wrtset_content_t wrtset;

    
S_ROOT: 
    
    //ROOT相关
    offset_t local_root = *(g_tree.g_root);
    version_t root_ver;

    if (local_root == -1) {
        // 正经的判断并处理new tree的情况.
        if (tx_read_root()==false) goto S_ROOT;
        __syncwarp();
        if (local_root != -1) goto S_WORK;
        
        rdset_s.loc[0] = -1;
        rdset_s.loc[1] = -1;
        rdset_s.offset = 0;
        wrtset_s.size = 0;
        
        if (create_new_tree() == false) goto S_ROOT;
        if (tx_commit(true, NULL) == false) goto S_ROOT;
        goto S_ROOT;
    }

S_WORK:
//=============================================================================== 
    
    const int loop_times = tx_size_get / 1;
    for (int l_i=0; l_i<loop_times;l_i++) {
       
        
        const int mission_id = tx_id * tx_size_get + l_i; 
        key_t target_key;       //每个线程保有自己处理的任务
        mission_t mission;      //每个线程保有自己处理的任务
        offset_t target_val;     //每个线程保有自己处理的任务
        ans_t * ans;            //每个线程保有自己处理的任务
       
        if (mission_id >= size) return;
        
        
        int idx = g_data.g_idx[mission_id];
        target_key = g_data.g_keys[idx];
        mission = g_data.g_mission[mission_id];
        target_val = g_data.g_vals[idx];
        ans = &(g_data.g_ans[idx]);   

#ifdef ABORT_COUNT 
        g_data.g_roll_back_count[mission_id].roll_back_times = 0;
        g_data.g_roll_back_count[mission_id].traversal_steps = 0;
        g_data.g_roll_back_count[mission_id].leaf_traversal_steps = 0;
        __syncwarp();
#endif 

        
        //每个thread拥有自己负责的那个任务的相关的值
        offset_t tmpCurNodeId;     
        key_t nodeInfo;                      
        key_t key;    
        offset_t val;
        static __shared__ offset_t searched_value_ss[Block_Dim_Put/WARPSIZE];    
        static __shared__ char target_thread_ss[Block_Dim_Put/WARPSIZE];          

        offset_t &searched_value_s = searched_value_ss[tx_id_in_block];         
        char &target_thread_s = target_thread_ss[tx_id_in_block];

        bool backFlag = false;
        
        //record last layer 
        offset_t lastNodeId;
        key_t last_node_info;
        key_t last_key;       //记录 last_key 
        offset_t last_val;    //记录 last_val
        char last_target_thread; 
        

 
TX_START:
#ifndef ENABLE_LL_RB
TX_START1:
#endif 


#ifdef ABORT_COUNT
        //__syncwarp();
        //if (tx_thread_id == 0)
        //    g_data.g_roll_back_count[mission_id].roll_back_times++;
        //__syncwarp();
#endif  


        //这个函数要对 tmpCurNodeId, nodeInfo, key, val, target_thread_s 进行更新。
        //last系列的不需要。如果node是满的会直接返回false.
        offset_t local_root1 = *(g_tree.g_root);
        wrtset_s.size = 0;
#define THRESHOLD1 1
        int timesss = 0;
        while (timesss < THRESHOLD1) {
            timesss++;
            if (radical_search_for_insert(target_key, local_root1)==true) {

                goto S_WRITE; 
            }
        }

        __syncwarp();
 

        //默认两个set内容为空
        rdset_s.loc[0] = -1;
        rdset_s.loc[1] = -1;
        rdset_s.offset = 0;
        wrtset_s.size = 0;
        __syncwarp();

        if (tx_read_root()==false) { goto  TX_START; }
        //到这里 Local_root不可能为-1 

  


        // 初始化searched_value
        searched_value_s = local_root;
        
        __syncwarp();

        //初始化tmpCurNodeId
        tmpCurNodeId = -1;
        __syncwarp();

        
        //search
        do {
/*{{{*/
#ifdef ENABLE_LL_RB
/*{{{*/
            goto BACK;  //这里好难受....有空改成func

TX_START1:
            if (lastNodeId == -1) goto TX_START;
    
            //默认两个set内容为空
            //rdset_s.loc[0] = -1;
            //rdset_s.loc[1] = -1;
            //rdset_s.offset = 0;
            rdset_s.loc[!rdset_s.offset] = -1; 
            //事实上，我们需要保证我们再重新读last node时，它跟我们第一次读的时候version相同，
            //所以，把那个记录依旧留在rdset中，tx_read就会自动检查它,
            //在这里我们只重置另一个记录
            wrtset_s.size = 0;
            searched_value_s = lastNodeId;
            __syncwarp();
            tmpCurNodeId = -1;
            backFlag = true;
            //从这里出来的，searched_value_s所保留的nodeId一定不是leaf的.
            //所以这个循环算上这次至少要跑2次.
            //那么错误的lastNodeId最多存活在下一个循环里。（再下一个就被覆盖掉了.）
BACK: 
/*}}}*/
#endif
            lastNodeId = tmpCurNodeId;
            last_node_info = nodeInfo; 
            last_key = key;
            last_val = val;
            last_target_thread = target_thread_s;
           
            tmpCurNodeId = searched_value_s;

#ifdef ABORT_COUNT 
            if (tx_thread_id == 0)
                g_data.g_roll_back_count[mission_id].traversal_steps++; 
#endif 
            // read node
            if (read_node(tmpCurNodeId, key, val, nodeInfo)==false) { 
                goto TX_START1;
            }
            
            __syncwarp();
            
 //=========提前分裂 situation one 提前分裂满的internal node
/*{{{*/
            if ((getNodeSize(nodeInfo)==ORDER-1)&&(!isLeaf(nodeInfo))) {
                
                if (backFlag) goto TX_START;
                int up_key = -1;
                //Internal node split 提前分裂
                if (node_splitting(tmpCurNodeId, key, val, nodeInfo, up_key)==false) goto TX_START1;
                char place_for_new_loc;
                if (lastNodeId == -1){
                    if (insert_new_root(up_key, tmpCurNodeId, getNodeLevel(nodeInfo)+1, place_for_new_loc)==false) goto TX_START;
                    if (tx_commit(true, place_for_new_loc)==false) goto TX_START;
                }
                else {
                    if (insert_node(lastNodeId, up_key, last_key, last_val, last_target_thread, last_node_info, place_for_new_loc)==false) goto TX_START1;
                    if (tx_commit(false, place_for_new_loc)==false) goto TX_START1;
                }
                //tx_commit
                goto TX_START;
            }
            __syncwarp();
/*}}}*/
            //get target 
            get_target(key, val, target_key, searched_value_s, target_thread_s ); 
            
            __syncwarp();
            
            backFlag = false;
            
               
            /*}}}*/
        }while (!isLeaf(nodeInfo)); 
       //==================================SEARCH DONE==================================== 

S_WRITE:

#ifdef ABORT_COUNT
        __syncwarp();
        if (tx_thread_id == 0)
            g_data.g_roll_back_count[mission_id].roll_back_times = timesss;
        __syncwarp();
#endif
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
            goto TX_START1;
        }
        if (up_key!=-1) {
            /*if (backFlag)  {
                printf("noway!\n");
                goto TX_START;
            }*/
            char place_for_new_loc;
            if (lastNodeId == -1) {
                if (insert_new_root(up_key, tmpCurNodeId,getNodeLevel(nodeInfo)+1, place_for_new_loc) == false) goto TX_START;
                if (tx_commit(true, place_for_new_loc)==false) goto TX_START;
            }
            else {
                if (insert_node(lastNodeId, up_key, last_key, last_val, last_target_thread, last_node_info, place_for_new_loc)==false) goto TX_START1; 
                if (tx_commit(false, place_for_new_loc)==false) goto TX_START1;
            }
        }
        else {
            if (tx_commit(false, NULL)==false) goto TX_START1;
        }
        
        //if (tx_thread_id==0) printf("success return %d\n", target_key);
        __syncwarp();
        continue; 





    }
    
 
    

    return;

}





__global__ void search_kernel( g_tree_t g_tree,
        g_data_t g_data,
        int boundary,       //从第boundary个tx开始
        int size           //算上boundary大小的size (这个size是以mission个数为单位.) 
        ){
    /*{{{*/
    const int tx_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE + boundary;
    
    //const int tx_thread_id = (gridDim.x * blockDim.x + threadIdx.x) % ORDER;
    const int tx_id_in_block = threadIdx.x / WARPSIZE;
    
    if (tx_id * tx_size_get >= size) return; //把一整个warp都在size外面的扔掉.
    
    const int id_in_ntg_get = threadIdx.x % ntg_size_get;
    const int ntg_id_in_block_get = threadIdx.x / ntg_size_get;
    const int ntg_id_in_tx_get = threadIdx.x % WARPSIZE / ntg_size_get;
    const int mission_id_get = tx_id * tx_size_get + ntg_id_in_tx_get;  
    const int mask_get = __ballot_sync(0xffffffff, mission_id_get<size);

   
    
 

    //mission_t mission_get;
    key_t target_key_get;
    ans_t *ans_get;
    if (mission_id_get >= size) return;
    
#ifdef ABORT_COUNT
    g_data.g_roll_back_count[mission_id_get].roll_back_times = 0;
    g_data.g_roll_back_count[mission_id_get].traversal_steps = 0;
    g_data.g_roll_back_count[mission_id_get].leaf_traversal_steps = 0;
    __syncwarp();
#endif        
    
    //mission_get = g_data.g_mission[mission_id_get]; 
    int idx = g_data.g_idx[mission_id_get];
    target_key_get = g_data.g_keys[idx];
    ans_get = &(g_data.g_ans[idx]);   
    
      
    offset_t local_root = *(g_tree.g_root);
    if (local_root == -1) {
        *ans_get = -1;
        return ;
    }    
    
    radical_search(target_key_get, *ans_get, local_root); 
    
    return;

/*}}}*/

}









void GPU_STM_Tree::launchKernel(int size, bool special, Time_Measure &t) {

    int grid_dim1 = (h_boundary * WARPSIZE + (Block_Dim_Put-1)) / Block_Dim_Put;
    int grid_dim2 = (h_device_working_num2 - h_boundary * WARPSIZE + (Block_Dim_Get-1)) / Block_Dim_Get;
    
   
    //cout<<"true working:\t"<<h_device_working_num2<<endl;
    //cout<<"grid_dim1: "<<grid_dim1<<endl;
    //cout<<"grid_dim2: "<<grid_dim2<<endl;
    //cout<<"working size:      "<<h_device_working_num2<<endl;
    //cout<<"boundary:          "<<h_boundary*tx_size_get<<endl;
    t.gpuTimeStart();
    if (grid_dim1 != 0) 
        tx_insert_kernel<<<grid_dim1, Block_Dim_Put>>>(g_tree, g_data, g_tx, h_boundary, h_device_working_num2);
    t.gpuTimeEnd();
    t.gpuTimeStart();
    if (grid_dim2 !=0) 
        search_kernel<<<grid_dim2, Block_Dim_Get>>> (g_tree, g_data, h_boundary, h_device_working_num2);
    t.gpuTimeEnd();
}


void GPU_STM_Tree::launchKernelBefore(int size,Time_Measure &t) {
    t.gpuTimeStart();
    
    //cub::DeviceRadixSort::SortPairs<key_t, int>(g_data_phase_1.d_temp_storage, g_data_phase_1.temp_storage_bytes, g_data_phase_1.g_keys, g_data_phase_2.g_keys, g_data_phase_1.g_idx, g_data_phase_2.g_idx, batch_size, 0,32, 0, true);
    cub::DeviceRadixSort::SortPairs<key_t, int>(g_data_phase_1.d_temp_storage, g_data_phase_1.temp_storage_bytes, g_data_phase_1.g_keys, g_data_phase_2.g_keys, g_data_phase_1.g_idx, g_data_phase_2.g_idx, size);
    t.gpuTimeEnd();
    t.gpuTimeStart();
    cub::DeviceRunLengthEncode::Encode(g_data_phase_2.d_temp_storage, g_data_phase_2.temp_storage_bytes, g_data_phase_2.g_keys, g_data_phase_3.g_keys, g_data_phase_3.g_count, g_data_phase_3.g_num, size);
    
    t.gpuTimeEnd();
    t.gpuTimeStart();
    //scan真正需要的item_num应该是encode得到的g_num,所以要把g_num传回来，然后就能把g_num的值作为scan的参数传入了.
    CUDA_ERROR_HANDLER(cudaMemcpy(&h_device_working_num2, g_data_phase_3.g_num, sizeof(int), cudaMemcpyDeviceToHost));
    
    cub::DeviceScan::ExclusiveSum(g_data_phase_3.d_temp_storage, g_data_phase_3.temp_storage_bytes, g_data_phase_3.g_count, g_data_phase_3.g_idx, h_device_working_num2);
    CUDA_ERROR_HANDLER(cudaDeviceSynchronize());

    t.gpuTimeEnd();
    t.gpuTimeStart();
    //int tmpBlockDim = (Block_Dim_Get*4<=512)?Block_Dim_Get*4:512 ;   //4是随便写的
    int tmpBlockDim = 64;
    dim3 grid_dim( (size + tmpBlockDim-1) / tmpBlockDim );
    dim3 block_dim(tmpBlockDim);
    mergeMission<<<grid_dim, block_dim>>>(g_data_phase_2, g_data_phase_3, g_data_phase_4, g_data);
    t.gpuTimeEnd();

    t.gpuTimeStart();
    cub::DeviceRadixSort::SortPairs<short, int>(g_data_phase_4.d_temp_storage, g_data_phase_4.temp_storage_bytes, g_data_phase_4.g_mission, g_data.g_mission, g_data_phase_4.g_idx, g_data.g_idx, h_device_working_num2, 0, 2);
    
    t.gpuTimeEnd();
    t.gpuTimeStart();
    //CUDA_ERROR_HANDLER(cudaDeviceSynchronize());
   
    dim3 grid_dim2( (h_device_working_num2 + tmpBlockDim-1) / tmpBlockDim );
    getBoundary0<<<1, 1>>>(g_data.g_boundary);
    getBoundary1<<<grid_dim2, block_dim>>>(g_data.g_mission, g_data.g_boundary, h_device_working_num2);
    getBoundary2<<<1, 1>>>(g_data.g_mission, g_data.g_boundary, h_device_working_num2);
    
    CUDA_ERROR_HANDLER(cudaMemcpy( &h_boundary, g_data.g_boundary, sizeof(int), cudaMemcpyDeviceToHost ));
    CUDA_ERROR_HANDLER(cudaDeviceSynchronize());
    t.gpuTimeEnd();
}

void GPU_STM_Tree::launchKernelAfter(int size){
    dim3 grid_dim(Grid_Dim);
    dim3 block_dim(Block_Dim_Get / WARPSIZE * tx_size_get);
    writebackAns<<<grid_dim, block_dim>>>(g_data_phase_3, g_data, size);
}

__global__ void preprocessKernel(g_tree_t g_tree, g_tx_t g_tx){
    
    *(g_tree.g_root) = -1;
    *(g_tree.g_tree_size) = 0;

    g_tx.g_lock_table[MAX_NODE_NUM-1] = 0;    //root lock 初始化

}
__global__ void preprocessKernel_idx_init(int *g_idx_bef) {
    g_idx_bef[blockDim.x * blockIdx.x + threadIdx.x] = blockDim.x * blockIdx.x + threadIdx.x;
}




void GPU_STM_Tree::preprocess() {
    dim3 grid_dim(1);
    dim3 block_dim(1);
    preprocessKernel<<<grid_dim, block_dim>>>(g_tree, g_tx);
    dim3 grid_dim2(Grid_Dim);
    dim3 block_dim2(Block_Dim_Get / WARPSIZE * tx_size_get);
    preprocessKernel_idx_init<<<grid_dim2, block_dim2>>>(g_data_phase_1.g_idx);
    
    //为了得到temp_storage_bytes大小
    cub::DeviceRadixSort::SortPairs<key_t, int>(g_data_phase_1.d_temp_storage, g_data_phase_1.temp_storage_bytes, g_data_phase_1.g_keys, g_data_phase_2.g_keys, g_data_phase_1.g_idx, g_data_phase_2.g_idx, batch_size);
    CUDA_ERROR_HANDLER(cudaMalloc(&(g_data_phase_1.d_temp_storage), g_data_phase_1.temp_storage_bytes ));

    cub::DeviceRunLengthEncode::Encode(g_data_phase_2.d_temp_storage, g_data_phase_2.temp_storage_bytes, g_data_phase_2.g_keys, g_data_phase_3.g_keys, g_data_phase_3.g_count, g_data_phase_3.g_num, batch_size);
    CUDA_ERROR_HANDLER(cudaMalloc(&(g_data_phase_2.d_temp_storage), g_data_phase_2.temp_storage_bytes ));

    cub::DeviceScan::ExclusiveSum(g_data_phase_3.d_temp_storage, g_data_phase_3.temp_storage_bytes, g_data_phase_3.g_count, g_data_phase_3.g_idx, batch_size);
    CUDA_ERROR_HANDLER(cudaMalloc(&(g_data_phase_3.d_temp_storage), g_data_phase_3.temp_storage_bytes ));

    cub::DeviceRadixSort::SortPairs<short, int>(g_data_phase_4.d_temp_storage, g_data_phase_4.temp_storage_bytes, g_data_phase_4.g_mission, g_data.g_mission, g_data_phase_4.g_idx, g_data.g_idx, batch_size, 0, 2);
    CUDA_ERROR_HANDLER(cudaMalloc(&(g_data_phase_4.d_temp_storage), g_data_phase_4.temp_storage_bytes ));
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
    cout<<g_tx.g_wrtSet_location<<endl;
    cout<<g_tx.g_wrtSet_version<<endl;
    cout<<g_tx.g_wrtSet_content_key<<endl;
    cout<<g_tx.g_wrtSet_content_val<<endl;
    cout<<(version_t*)g_tx.g_lock_table<<endl;
#endif


}


















}
