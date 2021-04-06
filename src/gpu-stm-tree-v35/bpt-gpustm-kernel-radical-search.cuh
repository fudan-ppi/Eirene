#ifndef BPT_GPUSTM_KERNEL_RADICAL_SEARCH_CUH_35
#define BPT_GPUSTM_KERNEL_RADICAL_SEARCH_CUH_35

#include "bpt-gpustm.h"
#include "bpt-gpustm-kernel-tree.cuh"

namespace gpu_stm_nsp_35 {
#define PRINT_NODE_INFO 0

__forceinline__ __device__ void INDEX_SEARCH_WITH_CACHE(
        key_t target_key,
        key_t &last_target_key,
        ans_t &ret,
        int local_root, // index 结构中的rootid
        int &cacheNodeId,
        bool isLastThread,
        g_tree_t &g_tree,
        int mask,
        int index_size
#ifdef ABORT_COUNT
        ,
        int mission_id, 
        count_t *g_test
#endif 
        ){
/*{{{*/
#define ntg_size ntg_size_get 
#define tx_size tx_size_get 

        
        offset_t nodeId=0;
        const offset_t index_range=ORDER/index_size;
        offset_t indexId=0;
        int isLast;
        int ii;
#ifndef NO_BUFFER
        if (cacheNodeId >= 0)  {
            nodeId = cacheNodeId;

            //printf("test:%d:  %d - %d = %d\n",threadIdx.x, target_key, last_target_key, target_key - last_target_key);
#ifdef USING_STEP_THRESHOLD
            if (__all_sync(mask, target_key - last_target_key < BUFFER_THRESHOLD )) 
#endif      //USING_STEP_THRESHOLD
            {
                goto search_list;
            }
        }        
#endif      //NO_BUFFER
        nodeId=local_root;
        isLast = g_tree.g_index_section[nodeId].isLast;


        //int layer=0;
        while(isLast==0){

#ifdef ABORT_COUNT
            g_test[mission_id].traversal_steps++;
#endif 
           // layer++;
            int ii=-1;
            indexId=0;
            for (int i=0; i<=index_size; i++) { //由于index 只有ORDER/8=4 ,因此ORDER/ntg_size<=4
                ii=i; 
               // ii = id_in_ntg + ntg_size * i;
               
                key_t key=MAX_KEY;
                if(ii<index_size){
                    key= g_tree.g_index_section[nodeId].index[ii];
                }

                if (target_key < key) {
                    break;
                }
            }
               
            if(ii>0){
                indexId=ii-1;
            }else{
                indexId=ii;
            }
            
            for(int i=0;i<=index_range;i++){ // 在8个key中查找
               // ii = id_in_ntg + ntg_size * i;
                ii=i;
                key_t key = MAX_KEY;
                if(ii<index_range){
                    key=g_tree.g_index_section[nodeId].key[indexId*index_range+ii];
                }

                if (target_key < key) {
                    break;
                }
                
            }
            
     
            if(ii>0){
                nodeId=__ldg(&(g_tree.g_prefix_sum[nodeId]))+indexId*index_range+ii-1;
            }else{
                nodeId=__ldg(&(g_tree.g_prefix_sum[nodeId]))+indexId*index_range;
            }


            isLast=g_tree.g_index_section[nodeId].isLast;

        }
        //找到最后一层，使用child查找叶子节点

#ifdef ABORT_COUNT 
        g_test[mission_id].traversal_steps++;
#endif 
        indexId=0;
        ii = -1;
        for (int i=0; i<=index_size; i++) { //由于index 只有ORDER/8=4 ,因此ORDER/ntg_size<=4
            ii=i;    
           // ii = id_in_ntg + ntg_size * i;
                
            key_t key =MAX_KEY;
            if(ii<index_size){
                key=g_tree.g_index_section[nodeId].index[ii];
            }

            if (target_key < key) {
                break;
            }
                
        }

        if(ii>0){
            indexId=ii-1;
        }else{
            indexId=ii;
        }


        ii=-1;
        for(int i=0;i<=index_range;i++){ // 在8个key中查找
            //ii = id_in_ntg + ntg_size * i;
            ii=i;
            key_t key =MAX_KEY;
            if(ii<index_range){
                key=g_tree.g_index_section[nodeId].key[indexId*index_range+ii];
            }

            if (target_key < key) {
                break;
            }
            

        }
        if(ii>0){
            nodeId=g_tree.g_index_section[nodeId].children[indexId*index_range+ii-1];
        }else{
            nodeId=g_tree.g_index_section[nodeId].children[indexId*index_range+ii];
        }
        //nodeId 为leafnode id，接下来找value
search_list:         
        offset_t minKey=MAX_KEY;
        offset_t nextLeaf=-1;
        nextLeaf=g_tree.g_next_section[nodeId].nodeid;

        if(nextLeaf>=0){
            minKey=g_tree.g_key_section[nextLeaf].key[0];
        }
#ifdef ABORT_COUNT
        g_test[mission_id].traversal_steps++;
        g_test[mission_id].leaf_traversal_steps++;
#endif


       // int nextCount=0;
        while(target_key>=minKey){
#ifdef ABORT_COUNT 
            g_test[mission_id].traversal_steps++;
            g_test[mission_id].leaf_traversal_steps++;
#endif 
           // nextCount++;
            nodeId=nextLeaf;
            nextLeaf=g_tree.g_next_section[nodeId].nodeid;
            minKey=MAX_KEY;
            if(nextLeaf>=0){
                minKey=g_tree.g_key_section[nextLeaf].key[0];
            }
        }
     //   if(nextCount>0){
     //       printf("------$$$$$ %d\n",nextCount);
     //   }
        ii = -1;
        for (int i=0; i<ORDER; i++) {
           // ii = id_in_ntg + ntg_size * i;
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
 

        if (isLastThread) {
            cacheNodeId = target_key;   //没办法，就这么用吧。
        }
        __syncwarp(mask);
        //printf("threadIdx %d, old last_target_key %d, new_last_target_key %d\n",threadIdx.x, last_target_key, cacheNodeId);
        //printf("mission_id:   %d   aaaa:   %d\n", target_key, last_target_key);
        last_target_key = cacheNodeId;
        __syncwarp(mask);
        //printf("mission_id:   %d   bbbb:   %d\n", target_key, last_target_key);
        if (isLastThread) {
            cacheNodeId = nodeId;
        }
        __syncwarp(mask);
        //printf("mission_id:   %d   cccc:   %d\n", target_key, last_target_key);
      //  __threadfence();
  /* 
        if(ret==-1 ){
            printf("ii:%d, nodeId:%d, targetKey:%d, minKey:%d, nextCount:%d,warp_id:%d\n",ii,nodeId,target_key,minKey,nextCount,(blockDim.x*blockIdx.x+threadIdx.x)/ WARPSIZE );
        }
*/
    
        return;
#undef ntg_size 
#undef tx_size

/*}}}*/
}






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
            int mask){
#define ntg_size ntg_size_get
#define tx_size tx_size_get

        static __shared__ char flag_s[Block_Dim_Get/ntg_size][ntg_size+1];  //每ntg_size个线程拥有flag_s的一行.
        static __shared__ char nexthalf_s[Block_Dim_Get/ntg_size];            //每ntg_size个线程拥有一个.
        char *flag = flag_s[ntg_id_in_block];
        char &nexthalf = nexthalf_s[ntg_id_in_block];
        
        char selfFlag;
        flag[id_in_ntg] = 0;
        flag[id_in_ntg+1] = 0;
        __syncwarp(mask);
       
    
        static __shared__ offset_t nodeId_s[Block_Dim_Get/ntg_size];
       
        offset_t &nodeId = nodeId_s[ntg_id_in_block];
       
        nodeId = local_root;
        int height = getNodeLevel(g_tree.g_key_section[nodeId].key[ORDER-1]); //height从0开始，leaf是height 0.

        //搜索，不包括leaf
        for (int h=0; h<height ; h++) {
            int tmpMask = mask;
            nexthalf = -1;
            
            int ii = -1;
            for (int i=0; i<ORDER/ntg_size; i++) {                 
                ii = id_in_ntg + ntg_size * i;
                
                selfFlag = 0;
                flag[id_in_ntg+1] = 0;

                key_t key = g_tree.g_key_section[nodeId].key[ii];
                __syncwarp(tmpMask);
                if (ii==ORDER-1) key = MAX_KEY;
                __syncwarp(tmpMask);

                if (target_key < key) {
                    selfFlag = 1;
                    flag[id_in_ntg+1] = 1;
                    nexthalf = i;
                }
                __syncwarp(tmpMask);
                tmpMask = __ballot_sync(tmpMask, nexthalf==-1);
                if (nexthalf > -1)break;
                
            }

            if (selfFlag == 1 && flag[id_in_ntg]==0) {
               
                nodeId = g_tree.g_val_section[nodeId].val[ii];
            }
            __syncwarp(mask);
            //if (nodeId<0)||(nodeId>=MAX_NODE_NUM) return false;;
            //if (__any_sync(mask, (nodeId<0)|(nodeId>= *(g_tree.g_tree_size)))) return false;

        }
//===============================================

     
        
        //搜索leaf
        
        __syncwarp(mask);
        nexthalf = -1;
        int tmpMask = mask;
       
        int ii;
        for (int i=0; i<ORDER/ntg_size; i++) {
            ii = id_in_ntg + ntg_size * i;
            
            selfFlag = 0;
            flag[id_in_ntg+1] = 0;

            key_t key = g_tree.g_key_section[nodeId].key[ii];
            __syncwarp(tmpMask);
            
            if (ii==ORDER-1) key = MAX_KEY;

            if (target_key<key) {
                selfFlag = 1;
                flag[id_in_ntg+1] = 1;
                nexthalf = i;
            }

            __syncwarp(tmpMask);
            tmpMask = __ballot_sync(tmpMask, nexthalf==-1);
            if (nexthalf > -1) break;
        }
        __syncwarp(mask);
        
        ret = -1;
        __syncwarp(mask);

        if (selfFlag == 1 && flag[id_in_ntg]==0) {
            if (ii>0 && g_tree.g_key_section[nodeId].key[ii-1] == target_key) {
                ret = g_tree.g_val_section[nodeId].val[ii-1];
            }
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
