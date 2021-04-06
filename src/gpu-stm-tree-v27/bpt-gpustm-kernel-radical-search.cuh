#ifndef BPT_GPUSTM_KERNEL_RADICAL_SEARCH_CUH_27
#define BPT_GPUSTM_KERNEL_RADICAL_SEARCH_CUH_27

#include "bpt-gpustm.h"
#include "bpt-gpustm-kernel-tree.cuh"
#include "../global_conf.h"

namespace gpu_stm_nsp_27 {



    //RADICAL SEARCH for search 其实就是普通的search.
    __forceinline__ __device__ void RADICAL_SEARCH(
            key_t target_key,
            ans_t &ret,             //会把答案直接写回ret.

            int local_root,

            g_tree_t  &g_tree,
            int tx_id_in_block,
            int id_in_ntg,
            int ntg_id_in_block,
            int ntg_id_in_tx,
            int mask 
#ifdef ABORT_COUNT 
            ,
            int mission_id, 
            count_t *g_test
#endif 
            ){
#define ntg_size ntg_size_get
#define tx_size tx_size_get
   
        offset_t nodeId = local_root;
        int height = getNodeLevel(g_tree.g_key_section[nodeId].key[ORDER-1]); //height从0开始，leaf是height 0.

        //搜索，不包括leaf
        for (int h=0; h<height ; h++) {
            
#ifdef ABORT_COUNT 
            g_test[mission_id].traversal_steps++;
#endif 
            int ii = -1;
            for (int i=0; i<ORDER; i++) {
                ii=i;       
                key_t key = g_tree.g_key_section[nodeId].key[ii];
                if (ii==ORDER-1) key = MAX_KEY;

                if (target_key < key) {
                        break;
                }
                
            }

            nodeId = g_tree.g_val_section[nodeId].val[ii];
            __syncwarp(mask);

        }
//===============================================

     
        
        //搜索leaf
        
        __syncwarp(mask);
       
#ifdef ABORT_COUNT 
        g_test[mission_id].traversal_steps++;
#endif 
        int ii;
        for (int i=0; i<ORDER; i++) {
            ii=i;     
            key_t key = g_tree.g_key_section[nodeId].key[ii];
            
            if (ii==ORDER-1) key = MAX_KEY;

            if (target_key<key) {
                break;
            }

        }
        __syncwarp(mask);
        
        ret = -1;
        __syncwarp(mask);

        if (ii>0 && g_tree.g_key_section[nodeId].key[ii-1] == target_key) {
            ret = g_tree.g_val_section[nodeId].val[ii-1];
        }
        
   
        return;
    
#undef ntg_size 
#undef tx_size
    
    }

    //ntg_size = 32.
    __forceinline__ __device__ bool RADICAL_SEARCH_FOR_INSERT(
        key_t target_key,
        offset_t local_root,
        
        offset_t &tmpCurNodeId,
        key_t &key,
        offset_t &val,
        key_t &nodeInfo,
        char &target_thread_s,

        //others:
        g_tree_t &g_tree,
        g_tx_t &g_tx,
        rdset_t &rdset_s,
        int tx_id_in_block,
        int tx_thread_id  

#ifdef ABORT_COUNT 
        ,
        int mission_id, 
        count_t *g_test
#endif 
            
        
        ) {
       
        static __shared__ bool flag_s[Block_Dim_Put/WARPSIZE][WARPSIZE+1];
        bool *flag = flag_s[tx_id_in_block];
        bool selfFlag;

        static __shared__ offset_t nodeId_s[Block_Dim_Put/WARPSIZE];
        offset_t &nodeId = nodeId_s[tx_id_in_block]; 
        
        nodeId = local_root;

        flag[tx_thread_id] = 0;
        flag[tx_thread_id-1] = 0;

        __syncwarp();
        
        int height = getNodeLevel(g_tree.g_key_section[nodeId].key[ORDER-1]); //height从0开始，leaf是height 0.
       
        
        //搜索，不包括leaf
        for (int h=0; h<height; h++) {

#ifdef ABORT_COUNT 
            if (tx_thread_id == 0)
                g_test[mission_id].traversal_steps++;
#endif 


            selfFlag = 0;
            flag[tx_thread_id+1] = 0;
            
            key_t key1 = g_tree.g_key_section[nodeId].key[tx_thread_id];
            
            key_t nodeInfo1 =  __shfl_sync(0xffffffff, key1, ORDER-1, ORDER);
            if (getNodeSize(nodeInfo1) == ORDER-1)
                return false;
            if (tx_thread_id==ORDER-1) {
                key1 = MAX_KEY;
            }
            __syncwarp();
            
            
            if (target_key < key1) {
                selfFlag = 1;
                flag[tx_thread_id+1] = 1;
            }
            __syncwarp();
            

            if (selfFlag == 1 && flag[tx_thread_id] == 0) {
                nodeId = g_tree.g_val_section[nodeId].val[tx_thread_id];
            }
            __syncwarp();
            
            if (nodeId < 0 || nodeId >= MAX_NODE_NUM) {
                return false; 
            }

        }
        
        __syncwarp();
        

#ifdef ABORT_COUNT 
        //统计leaf
        if (tx_thread_id == 0)
            g_test[mission_id].traversal_steps++;
#endif 

        //搜索 leaf 参考 tx.cuh 中TX_READ.
        
        version_t lock = g_tx.g_lock_table[nodeId];
        __syncwarp();
        
        if (getLockBit(lock)==1) {
            return false;
        }
        
        version_t version = getVersion(lock);
        
        key = g_tree.g_key_section[nodeId].key[tx_thread_id];
        val = g_tree.g_val_section[nodeId].val[tx_thread_id];
        
        __syncwarp();

        lock = g_tx.g_lock_table[nodeId];

        if ((getLockBit(lock)==1)||(getVersion(lock)!=version))  {
            return false;
        }
      
        //==
        nodeInfo = __shfl_sync(0xffffffff, key, ORDER-1, ORDER);
        key_t right_min = __shfl_sync(0xffffffff, val, ORDER-1, ORDER);

        __syncwarp();
        //不但满的是不行的，插入之后会分裂也是不行的.
        if (target_key >= right_min || getNodeSize(nodeInfo) == ORDER-1 || getNodeLevel(nodeInfo)!=0) 
            return false;
        
        
        
        
        if (tx_thread_id == ORDER-1) {
            key = MAX_KEY;
        }

        __syncwarp();
        //find target 
        selfFlag = 0; 
        flag[tx_thread_id+1] = 0;
        __syncwarp();
        
        if (target_key < key) {
            selfFlag = 1;
            flag[tx_thread_id+1] = 1;
        }
        __syncwarp();
    
        if (selfFlag == 1 && flag[tx_thread_id]==0) {
            target_thread_s = tx_thread_id; 
        }

        __syncwarp();
        
        
        if (target_thread_s > 0) {
            
            //更新rdset
            tmpCurNodeId = nodeId;
            rdset_s.loc[0] = nodeId;
            rdset_s.ver[0] = version; 
            rdset_s.loc[1] = -1;
            rdset_s.offset = 0;

            __syncwarp();
            return true;
        }

        return false;


    }



}







#endif
