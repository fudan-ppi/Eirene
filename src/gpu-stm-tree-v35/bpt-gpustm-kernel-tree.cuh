#ifndef BPT_GPUSTM_KERNEL_TREE_CUH_35
#define BPT_GPUSTM_KERNEL_TREE_CUH_35


#include "bpt-gpustm-kernel-tx.cuh"
#include "bpt-gpustm-kernel-def.h"


//写在前头（也许不应该在这文件里）
/*
   线程间共享变量：
   1. 使用shared memory
   2. 使用global memory
   3. 使用Local的register然后广播
   
   使用的优先级： 3>1>2
   但是使用3必须要求事前知道需要广播哪个线程的值
   有的时候，即在2的global memory在其他的地方已经申请出来的时候，就没必要硬要再使用1方案了。shared memory要节省用。

*/




//info 从低位到高位分别 5位level,5位size(key的数目，最多ORDER-1=31个) 
//其中level 0 为child，以此类推

namespace gpu_stm_nsp_35 {
    __forceinline__ __device__ bool isLeaf(key_t k){
        return ((k&0x1f)==0);
    }

    //return key的数量 
    __forceinline__ __device__ int getNodeSize(key_t k) {
        return k>>5;
    }

    __forceinline__ __device__ int getNodeLevel(key_t k) {
        return (k&0x1f);
    }

    __forceinline__ __device__ key_t info(int size, int level){
        return ((size<<5)+level);
    }

    __forceinline__ __device__ key_t infoSizeInc(key_t info) {
        return (info+0x20);
    }

    __forceinline__ __device__ key_t infoSetSize(key_t info, int size) {
        return  ((info&0x1f) | (size<<5));
    }

    __forceinline__ __device__ key_t infoSetLevel(key_t info, int level) {
        return ((info&0xffffffe0) | (level&0x1f));
    }



    //一次性输入tx_size个node, 同时读出. 每ntg_size个线程处理一个node的read
    __forceinline__ __device__ bool READ_NODE(
            offset_t nodeId, 
            key_t &key,            //存入读到的值
            offset_t &val,
            key_t &nodeInfo,        //local for each thread, 返回前会广播
            //others:
            int tx_thread_id,
            g_tx_t &g_tx,
            g_tree_t &g_tree,
            
            rdset_t &rdset_s
            ){

/*{{{*/
        if (tx_read(nodeId, key, val)==false) return false;
        if (tx_thread_id == ORDER-1) {
            nodeInfo = key;
            key = MAX_KEY;
        }
        __syncwarp();
        nodeInfo = __shfl_sync(0xffffffff, nodeInfo, ORDER-1, ORDER );    //以每ntg_size个thread为一组，广播其中第(ntg_size-1)thread的值
        __syncwarp();
        return true;
/*}}}*/
    }


    __forceinline__ __device__ void GET_TARGET(
            key_t key,
            offset_t val,

            key_t target_key,
            offset_t &target_val,       //shared 
            char &target_thread,        //shared 
           
            //others:
            int tx_thread_id,
            int tx_id_in_block){
        /*{{{*/
        static __shared__ char flag[Block_Dim_Put/WARPSIZE][ORDER+1];

        char selfFlag = 0;
        flag[tx_id_in_block][tx_thread_id] = 0;
        
        __syncwarp();
        
        if (target_key<key) {
            flag[tx_id_in_block][tx_thread_id+1] = 1;
            selfFlag = 1;
        }
        __syncwarp();
        if (selfFlag == 1 && flag[tx_id_in_block][tx_thread_id] == 0) {
            target_val = val;
            target_thread = (char) tx_thread_id;
        }
        __syncwarp();
        /*}}}*/
    }



    __forceinline__ __device__ bool CREATE_NEW_TREE(
            //others:
            int tx_thread_id, 
            int tx_id_in_block,
            g_tx_t &g_tx,
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s,
            rdset_t &rdset_s,
            version_t &root_ver,
            offset_t &local_root
            
            ){
        /*{{{*/
        key_t key;
        offset_t val;
       

        //在一个leaf的ORDER-1的位置，key的位置记录info, ptr的位置记录其右边child的最小值, 用于激进时的比较.
        if (tx_thread_id==ORDER-1) {key = info(0,0); val = MAX_KEY;} //level = 0 leaf
        else {key = MAX_KEY; val = -1; }

        if (tx_write(-1, key, val)==false) {return false;}
        if (tx_write_root(-1)==false) {return false;}
        return true;
/*}}}*/
    }

    //一个拥有ORDER-1个KEY的NODE,分裂成两个NODE，每个各有(ORDER-2)/2个key, 多出来的那个中间的key上升。
    __forceinline__ __device__ bool NODE_SPLITTING(
            int nodeId,
            int key,        //该node的key                         
            int val,        //该node的val  
            //因为在调用这个分裂函数前，这个node一定是被读过一次并获得了key和val,并且该值还未扔掉。
            key_t nodeInfo,

            key_t &up_key,        //需要升上去的Key, local value，返回前会广播 
            
            //需要升上去的左child的offset， 必然是nodeId, 故不需要单独一个变量
            //由于右child必然是新的Node,所以升上去的值必然是-2, 所以不需要一个单独的变量。
            //other:
            int tx_thread_id,
            int tx_id_in_block,
            g_tx_t &g_tx,
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s, 
            rdset_t &rdset_s 
            ){
        /*{{{*/

        //node一定为满的，即 key的数量为ORDER-1.    //现在这个版本ORDER = 32
        //即child的编号为0 ~ (ORDER-1)，key的编号为0 ~ (ORDER-2), 那么最中间的那个key为(0+ORDER-2)/2
        int middle = (ORDER-2) / 2;     //编号15的那个线程的Key要上升
       
        
        __syncwarp();
        up_key = __shfl_sync(0xffffffff, key, middle, ORDER);
        
        int tmp_key = MAX_KEY;
        int tmp_val = -1;
        
        //old node:
        if (tx_thread_id < middle) {
            tmp_key = key; 
            tmp_val = val;
        }
        else if (tx_thread_id == middle) {
            //tmp_key = MAX_KEY; 
            tmp_val = val;
        }
        else if (tx_thread_id == ORDER-1) {
            tmp_key = info(middle,getNodeLevel(nodeInfo));   //size: middle个key, level = old level
            //tmp_val = -1;
        }

        __syncwarp();
        if (tx_write(nodeId, tmp_key, tmp_val)==false) return false;

        __syncwarp();
        
        //new node:
        tmp_key = MAX_KEY;
        tmp_val = -1;
 
        key = __shfl_down_sync(0xffffffff, key, middle+1, ORDER);
        val = __shfl_down_sync(0xffffffff, val, middle+1, ORDER);
        if (tx_thread_id <= middle) {
            tmp_key = key ;
            tmp_val = val;
            //tx_thread_id == middle时， 该线程会拿到ORDER-1那个位置的key，虽然这个位置在树里存的是Nodeinfo,但是作为参数在传进来之前就改成了MAX_KEY
        }
        else if (tx_thread_id == ORDER-1) {
            tmp_key = info(middle, getNodeLevel(nodeInfo));  //size: middle个key, level = old level
            //tmp_val = -1;
        }
        __syncwarp();
        if (tx_write(-2, tmp_key, tmp_val)==false) return false;

        return true; 
    
    
    /*}}}*/
    }







    /*  old node 
        key:        key0    key1    key2    key3  
        val:        ptr0    ptr1    ptr2    ptr3    ptr4    
        
        insert keyX between key1 and key2   (ptr2 node split to ptr2 and ptr_n) 

        new node 
        key:        key0    key1    keyX    key2    key3
        val:        ptr0    ptr1    ptr2    ptr_n   ptr3    ptr4
    
        target_thread --> thread of key2/ptr2 in old node 
     */


    __forceinline__ __device__ bool INSERT_NODE(

            int nodeId,
            key_t target_key, 
            key_t key,                  //已经读到的Key
            offset_t val,               //已经读到的val
            int target_thread,
            key_t nodeInfo,
            char &place_for_new_loc,
            
            //others:
            int tx_thread_id,
            int tx_id_in_block,
            g_tx_t &g_tx,
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s, 
            rdset_t &rdset_s 
            ) {
        
  /*{{{*/
        key_t tmp_key;
        offset_t tmp_val;


        
        key_t moved_key = __shfl_up_sync(0xffffffff, key, 1, ORDER);
        offset_t moved_val = __shfl_up_sync(0xffffffff, val, 1, ORDER);


        if (tx_thread_id < target_thread) {
            tmp_key = key;
            tmp_val = val;
        }
        else if (tx_thread_id == target_thread) {
            tmp_key = target_key;
            tmp_val = val;
        }
        else if ((tx_thread_id > target_thread) && (tx_thread_id < ORDER-1)){
            tmp_key = moved_key;
            tmp_val = moved_val;
            //其中，tx_thread_id == target_thread+1时， val应该等于新node的指针，但是由于对应的空间没有生成，所以这里暂时填什么都可以。
        }
        else if (tx_thread_id == ORDER-1) {
            tmp_key = infoSizeInc(nodeInfo);
            tmp_val = moved_val;
        }
        __syncwarp();
        
        if (tx_write(nodeId, tmp_key, tmp_val) == false) {return false;}
        place_for_new_loc = target_thread+1; 
        //place_for_new_loc = &(wrtset_valArr[wrtset_i[tx_id_in_block]-1].val[target_thread+1]);    //把应该放入right child的位置记录在place_for_new_loc中
        __syncwarp();

        return true;
/*}}}*/
    }



   
    //和INSERT_NEW_TREE的区别在于有place_for_new_loc
    __forceinline__ __device__ bool INSERT_NEW_ROOT(
            key_t target_key,
            offset_t left_child,
            int new_level,
            
            char &place_for_new_loc,   //本应存入 "right_child的offset"的空间的地址 的引用
            //others:
            int tx_thread_id,
            int tx_id_in_block,
            g_tx_t &g_tx,
            version_t &root_ver,
            offset_t &local_root,
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s,
            rdset_t &rdset_s 

            ){
/*{{{*/
        key_t key;
        offset_t val;
        if (tx_thread_id == 0)          {key = target_key; val = left_child;}
        else if (tx_thread_id==ORDER-1) {key = info(1, new_level); val = -1;}// node_size:1, new level
        else                            {key = MAX_KEY; val = -1;}


        __syncwarp();

        //build tree 
        if (tx_write(-1, key, val)==false) {return false;}
        place_for_new_loc = 1;
        //place_for_new_loc = &(wrtset_valArr[wrtset_i[tx_id_in_block]-1].val[1]);    //把应该放入right child的位置记录在place_for_new_loc中
        if (tx_write_root(-1)==false) {return false;}
        __syncwarp();
   
    
        return true;
/*}}}*/
    }


    //处理insert或update
    __forceinline__ __device__ bool INSERT_LEAF(
            
            int nodeId, 
            key_t nodeInfo,

            key_t target_key,
            offset_t target_val,
            
            key_t key,              //已经读到Leaf的Key
            offset_t val,           //已经读到Leaf的Val
            char target_thread,

            int &up_key,            //需要上升的key, 无则赋值为-1, local for each thread
            //需要升上去的左child的offset， 必然是nodeId, 故不需要单独一个变量
            //由于右child必然是新的Node,所以升上去的值必然是-2, 所以不需要一个单独的变量。
            
            //others : 
            int tx_thread_id,
            int tx_id_in_block,
            g_tx_t &g_tx,
            wrtset_content_t &wrtset,
            wrtset_t &wrtset_s,
            rdset_t &rdset_s
            ){
/*{{{*/
        //update
        up_key = -1;
        if (key == target_key) {
            val = target_val;
        }
        if (tx_thread_id==ORDER-1) {
            key = nodeInfo;
        }
        if (__any_sync(0xffffffff, key==target_key)) {
            if (tx_write(nodeId, key, val)==false) return false;
            return true; 
        }




        __syncwarp();
        
        
        key_t tmp_key;
        offset_t tmp_val;
        
        key_t moved_key = __shfl_up_sync(0xffffffff, key, 1, ORDER);
        offset_t moved_val = __shfl_up_sync(0xffffffff, val, 1, ORDER);

        if (tx_thread_id == target_thread) {
            tmp_key = target_key;
            tmp_val = target_val;
        }
        else if (tx_thread_id > target_thread) {
            tmp_key = moved_key;
            tmp_val = moved_val;
        }
        else {
            tmp_key= key;
            tmp_val = val;
        }
        __syncwarp();

        key_t right_min = val;  //仅对tx_thread_id == ORDER-1有效.
        if (getNodeSize(nodeInfo)==ORDER-1) {
            
            //上升结点为新生成结点的最小值
            up_key = __shfl_sync(0xffffffff, tmp_key, ORDER/2, ORDER);
            
            //old node: =================================
            key = MAX_KEY; 
            val = -1;
            if (tx_thread_id < (ORDER/2)) {
                key = tmp_key;
                val = tmp_val;
            }
            else if (tx_thread_id == ORDER-1) {
                key = info(ORDER/2, 0); //size: ORDER/2个key, isleaf
                val = up_key;
            }
            __syncwarp();
            if (tx_write(nodeId, key, val)==false) return false;
            
            //new node: ================================
            key = MAX_KEY;
            val = -1;
            tmp_key = __shfl_down_sync(0xffffffff, tmp_key, ORDER/2, ORDER);
            tmp_val = __shfl_down_sync(0xffffffff, tmp_val, ORDER/2, ORDER);
            if (tx_thread_id < (ORDER/2)) {
                key = tmp_key;
                val = tmp_val;
            }
            else if (tx_thread_id == ORDER-1) {
                key = info(ORDER/2, 0); //size: ORDER/2个key, isleaf
                val = right_min;
            
            }
            __syncwarp();

            if (tx_write(-2, key, val)==false) return false;


            return true;          
        }
        else {
            if (tx_thread_id == ORDER-1) {
                tmp_key = infoSizeInc(nodeInfo);
                tmp_val = val;  //right_min
            }
            __syncwarp();
            if (tx_write(nodeId, tmp_key, tmp_val) == false) {return false;}
            return true;
        }



/*}}}*/
    }

}
#endif
