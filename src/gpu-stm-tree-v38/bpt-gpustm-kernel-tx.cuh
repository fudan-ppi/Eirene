#ifndef BPT_GPUSTM_KERNEL_TX_CUH_38
#define BPT_GPUSTM_KERNEL_TX_CUH_38
#include "bpt-gpustm.h"
#include <cuda_runtime.h>





namespace gpu_stm_nsp_38 {


    //写在前头，本文件里用到了一些__all,实际上，用__any应该更好理解一些..虽然都没有错误..


    __forceinline__ __device__ bool getLockBit(version_t v) {return v&1;}
    __forceinline__ __device__ version_t getVersion(version_t v){return v>>1;}
    __forceinline__ __device__ version_t constructLock(version_t v, bool isLock){return (v<<1) + isLock;}

    //对于还没有真正分配空间的结点，用负数代表其id.
    //用-1代表新root，用-2开始的值代表其他新node



    __forceinline__ __device__ bool TX_READ(
            offset_t nodeId,
            int tx_thread_id,
            volatile keyArr_t * key_section,
            volatile valArr_t * val_section,
            volatile version_t * lock_table,

            rdset_t &rdset_s,        //shared rdset
            
            key_t &key,           //reference of key
            offset_t &val
            
            ) {
       /*{{{*/
        
        //对于BplusTree来说，在读一个node时，只需要保证其父亲结点的版本号的正确性即可，因此，再前面的结点理论上可以从rdset中舍去.


        

        
        version_t lock = lock_table[nodeId];
        __syncwarp(); 
               
    
        if (getLockBit(lock)==1) {
            return false;   //abort
        }

        version_t version = getVersion(lock);

        key = key_section[nodeId].key[tx_thread_id];
        val = val_section[nodeId].val[tx_thread_id];
        
        __syncwarp(); 

        //重新确认一下version
        lock = lock_table[nodeId];
        __syncwarp(); 
        if( (getLockBit(lock)==1) ||(getVersion(lock)!=version)) {
            return false;
        }
        __syncwarp(); 

        //check father 
        if (rdset_s.loc[rdset_s.offset]!=-1) {
            version_t father_lock = lock_table[rdset_s.loc[rdset_s.offset]];
            __syncwarp(); 
            if (getVersion(father_lock) != rdset_s.ver[rdset_s.offset]) {
                return false;
            }
        }
        __syncwarp();
        //放入rdset_s中
        if (tx_thread_id == 0) {
            rdset_s.offset = !(rdset_s.offset);
            rdset_s.loc[rdset_s.offset] = nodeId ;
            rdset_s.ver[rdset_s.offset] = version;
        }
        __syncwarp();
        
        return true;
    /*}}}*/
    }



    __forceinline__ __device__ bool TX_READ_ROOT(
            volatile version_t *lock_table,
            volatile offset_t *g_root,   //root pointer in global memory 
            
            version_t &root_ver,
            offset_t &local_root

            ) {
/*{{{*/
        version_t lock = lock_table[MAX_NODE_NUM-1];

        if (getLockBit(lock)==1) {
            return false;   //abort 
        }
        version_t version = getVersion(lock);
        
        root_ver = version;
        local_root = *g_root;
        
        lock = lock_table[MAX_NODE_NUM-1];
        if ((getLockBit(lock)==1)||(getVersion(lock)!=version)){
            //printf("yes?\n");
            return false;

        }
        return true;
    /*}}}*/
    }




//------------------------------------------------------------------------------------------
    // return true, TX_WRITE success
    // return false, ABORT!
    __forceinline__ __device__ bool TX_WRITE(
    //__device__ bool TX_WRITE(
            offset_t nodeId,
            int tx_thread_id, 
            volatile version_t *lock_table,

            wrtset_content_t &wrtset, 
            wrtset_t &wrtset_s,
            rdset_t &rdset_s,   //__shared
            

            key_t key,
            offset_t val
            
            ){
/*{{{*/

        //wrtset在目前的设计中，在理论上不会产生重复的记录

        //check lock optional              ---spin or abort?        we choose abort here! 
        version_t lock;
        //nodeId == -1 : new root 
        //nodeId <= -2 : new split node 
        if (nodeId >= 0){
            lock = lock_table[nodeId];
            if (getLockBit(lock)==1) {
                return false;   //abort!
            }
        
            version_t version = getVersion(lock);
            //version check! 检查rdset里对应node的version是否与刚刚读到的相等.  

            for (int i=0; i<2; i++) {
                if (nodeId == rdset_s.loc[i]) {
                    if (version == rdset_s.ver[i]) {
                        goto Next;
                    }
                    else return false;
                }
            }
            return false;

Next:
            __syncwarp();
        
        }
        else {
            //new node
            lock = 0;
        }



        //放入 wrtset 
        wrtset.key[wrtset_s.size] = key;
        wrtset.val[wrtset_s.size] = val;
        
        __syncwarp();
        if (tx_thread_id==0) {

            wrtset_s.loc[wrtset_s.size] = nodeId;
            wrtset_s.ver[wrtset_s.size] = getVersion(lock);
            wrtset_s.size++;
            //wrtset_s.size = (wrtset_s.size + 1) % 4;
        }
        __syncwarp();
        
        

        return true;
/*}}}*/

    }
    __forceinline__ __device__ bool TX_WRITE_ROOT(
            volatile version_t *lock_table,
            version_t root_ver,
            offset_t& local_root, 
            offset_t new_root   //现阶段这个值只可能是-1
            ){ 
/*{{{*/
        version_t lock = lock_table[MAX_NODE_NUM-1];
        if (getLockBit(lock)==1) {
            return false;
        }
        if (getVersion(lock)!=root_ver) {
            return false;
        }
        local_root = new_root; 
        return true; 
    /*}}}*/
    }



/*============================================================================

    tx_commit_readonly,
    acquire_lock,
    release_lock,
    只在TX_COMMIT中调用.
    
============================================================================*/ 

#define acquire_locks()\
    ACQUIRE_LOCKS(\
            tx_thread_id,\
            wrtset_s,\
            lock_table,\
            local_lock_map\
            )
    
    __forceinline__ __device__ bool ACQUIRE_LOCKS(
            int tx_thread_id,
            wrtset_t &wrtset_s, 
            volatile version_t *lock_table,
            bool *local_lock_map        //shared 
            ) {
/*{{{*/
        
        bool flag = false; 
        if (tx_thread_id < wrtset_s.size) {
            local_lock_map[tx_thread_id] = false; 
            
            offset_t loc = wrtset_s.loc[tx_thread_id];
            version_t v = wrtset_s.ver[tx_thread_id];
            if (loc >= 0) { //loc==-1: new root, loc<-1: new node
                version_t new_lock = constructLock(v,true); 
                version_t old_lock = constructLock(v,false);
                if (atomicCAS((version_t*)&lock_table[loc], old_lock, new_lock)==old_lock) {
                    local_lock_map[tx_thread_id] = true;
                }else{
                    flag = true; 
                }
            
            }
        
        }
        __syncwarp();
        __threadfence();
        if (__any_sync(0xffffffff, flag)){
            return false;
        }
        

        return true;

/*}}}*/
    }

#define release_locks(version_change)\
    RELEASE_LOCKS(\
            tx_thread_id,\
            wrtset_s,\
            lock_table,\
            local_lock_map,\
            version_change\
            )

    __forceinline__ __device__ void RELEASE_LOCKS(
            int tx_thread_id,
            
            wrtset_t &wrtset_s, 
            
            volatile version_t *lock_table, 
            bool *local_lock_map,
            
            bool version_change) {
/*{{{*/ 

        if (tx_thread_id < wrtset_s.size) {
            offset_t loc = wrtset_s.loc[tx_thread_id];
            version_t v = wrtset_s.ver[tx_thread_id];
        
            if (local_lock_map[tx_thread_id] == true) {        
                //lock_table[loc] = constructLock(v + version_change, false);
                atomicCAS((version_t*)&lock_table[loc], constructLock(v,true),constructLock(v+version_change,false) );
            }
        }
        __syncwarp();
        __threadfence();
/*}}}*/
    }
//=======================================================================================

//NEW COMMIT
    //return false, ABORT!
    __forceinline__ __device__ bool TX_COMMIT(
            int tx_thread_id,
            int tx_id_in_block,

            volatile keyArr_t *key_section,
            volatile valArr_t *val_section, 


            volatile version_t *lock_table,
            
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s,

            bool wrtRoot,
            volatile offset_t *g_root,
            version_t root_ver,

            
            volatile offset_t *tree_size,
            char place_for_new_loc     //shared     记录哪个线程需要新的val.
                // btree need the new node's offset before commit,
                // so it will use -1, -2, -3...
                // remember to fill "place_for_new_loc" with right offset

            ) {

/*{{{*/

        //ACQUIRE LOCKS====================================================================
        static __shared__ bool local_lock_map_s[Block_Dim/WARPSIZE][wrtSetSize];
        bool *local_lock_map = local_lock_map_s[tx_id_in_block];
        
        //acquire locks for wrtset
        if (acquire_locks() == false) {
            release_locks(false);
            return false; 
        }
        
        //ACQUIRE ROOT AND CHECK ROOT======================================================
        if ((wrtRoot)) {
            int tmpFlag;
            if (tx_thread_id == 0) {
                version_t old_root_lock = constructLock(root_ver, false) ;
                version_t new_root_lock = constructLock(root_ver, true) ;
                tmpFlag = (atomicCAS((version_t *)&lock_table[MAX_NODE_NUM-1], old_root_lock, new_root_lock)==old_root_lock);
            }
            __syncwarp();
            __threadfence();
            tmpFlag = __shfl_sync(0xffffffff, tmpFlag, 0, ORDER);

            if (tmpFlag == 0) {
                __syncwarp(); 
                release_locks(false);
                return false;
            }
        }else{
            //printf("root_ver %d, %d, %d, %d\n", root_ver, lock_table[MAX_NODE_NUM-1], blockIdx.x ,threadIdx.x);
            __syncwarp();
            //if (getVersion(lock_table[MAX_NODE_NUM-1]) != root_ver) {
            //    release_locks(false);
            //    return false; 
            //}
        }


        __threadfence();


        //READSET VALIDATION=========NO NEED==============================================================


        //for new node=====================================================================
        //for new node (only one new node) 到这个时刻，已经能保证必然COMMIT成功
        
        __shared__ offset_t tmp_loc_s[Block_Dim/WARPSIZE];
        offset_t &tmp_loc = tmp_loc_s[tx_id_in_block];
        tmp_loc = -1;
        __syncwarp();

        
        if (tx_thread_id < wrtset_s.size) {
            offset_t loc = wrtset_s.loc[tx_thread_id];
            


            //NEW NODE 
            if (loc<-1) {   //loc = -2
                offset_t new_loc = atomicAdd((offset_t*)tree_size,1 );
                //printf("new_loc: %d\n", new_loc);
                __threadfence();//tett
         //       printf("new_loc: %d, blockIdx.x %d, threadIdx.x %d\n", new_loc, blockIdx.x, threadIdx.x);
                tmp_loc = new_loc;
                wrtset_s.loc[tx_thread_id] = new_loc; //更新一下wrtset_loc
                
                lock_table[new_loc] = constructLock(0, false); 
                //新的Node不需要保护，因为在提交之前没有Node会指向它
                //在表中生成新Node的lock 

            }
            //NEW ROOT NODE 
            if (loc==-1) {
                offset_t new_loc = atomicAdd((offset_t*)tree_size,1 );
                //printf("new_root: %d\n", new_loc);
                __threadfence();//tett
        //        printf("new_root: %d, blockIdx.x %d, threadIdx.x %d\n", new_loc, blockIdx.x, threadIdx.x);
                *g_root = new_loc;                  //WRTIE BACK ROOT
                wrtset_s.loc[tx_thread_id] = new_loc; //更新一下wrtset_loc
                
                lock_table[new_loc] = constructLock(0, false);

                //新的Node不需要保护，因为在提交之前没有Node会指向它
                //在表中生成新Node的lock 
            }
            


        }
        __threadfence();//tett

        __syncwarp();
        
        
        //把new_loc放到wrtset具体内容里中需要它的位置
        //btree的情况里肯定只有一个()
        //这个被插入新值的父节点，一定是第三个调用TX_WRITE的.
        if (tmp_loc!=-1 && tx_thread_id == place_for_new_loc) 
            wrtset.val[2] = tmp_loc;        
        
        
        
        __syncwarp();
        //WRITE BACK======================================================================
        offset_t loc2 = -1;
        for (int i = 0; i < wrtset_s.size; i++) {
            //offset_t loc2 = wrtset_s.loc[i];
            loc2 = wrtset_s.loc[i];
            //printf("loc2: %d\n", loc2);
            
            key_section[loc2].key[tx_thread_id] = wrtset.key[i];
            val_section[loc2].val[tx_thread_id] = wrtset.val[i];
        }

        

        __syncwarp();
        __threadfence();
        //REALSE LOCKS==================================================================
        release_locks( true );
        //REALSE ROOT LOCK
        if (wrtRoot && tx_thread_id==0) {
            //lock_table[MAX_NODE_NUM-1] = constructLock(root_ver + 1, false); 
            atomicCAS((version_t*)&lock_table[MAX_NODE_NUM-1], constructLock(root_ver,true),constructLock(root_ver+1,false) );
        }
        __syncwarp();
        __threadfence();

        /*
        if (tx_thread_id==0) {
            printf("here, %d, %d\n", threadIdx.x, blockIdx.x);
            //printf("wrtset_s.size: %d, loc2: %d, blockid %d, threadid %d, wrtset_ptr %p\n", wrtset_s.size, loc2,blockIdx.x, threadIdx.x, &wrtset_s); 
            //printf("wrtset_s.size: %d, loc2: %d, blockid %d, threadid %d\n", wrtset_s.size, loc2,blockIdx.x, threadIdx.x); 
            //printf("wrtset_s.size: %d, loc2: %d\n", wrtset_s.size, loc2); 
            //printf("wrtset_s.size %d:\n", wrtset_s.size); 
            //for (int i=0; i<wrtset_s.size; i++) {
            //    printf("    loc: %d\n", wrtset_s.loc[i]);
            //}
        }
*/
        //if (blockIdx.x == 2000  && tx_thread_id==0) {
        //    printf("here!, wrtset_s.size: %d, loc2: %d, blockid %d, threadid %d, wrtset_ptr %p\n", wrtset_s.size, loc2,blockIdx.x, threadIdx.x, &wrtset_s); 
        //}


        return true;
        
/*}}}*/
    }



}



#endif
