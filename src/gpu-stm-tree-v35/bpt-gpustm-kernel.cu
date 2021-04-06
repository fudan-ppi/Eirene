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
using namespace gpu_stm_nsp_35; 


#define Radical_Max_Times 100 
namespace gpu_stm_nsp_35{
__global__ void build_index_up(g_tree_t g_tree,int index_size,int layer){
/*{{{*/
    const int tid=blockDim.x * blockIdx.x + threadIdx.x; // thread id 所有issue的线程的index
    const int warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE; 
    const int thread_id=threadIdx.x % WARPSIZE;// thread id in a warp

    offset_t leaf_size=*(g_tree.g_leaf_size);
    offset_t base=(leaf_size+ORDER-1)/ORDER;
    offset_t upNum=(base+ORDER-1)/ORDER;
    offset_t last_upnum=base;
    offset_t last_base=0;
    // layer 从0 开始，表示倒数第二层的创建
    for(int i=0;i<layer;i++){
        last_base=base;
        base=base+upNum;
        last_upnum=upNum;
        upNum=(upNum+ORDER-1)/ORDER;
        if(last_upnum==1){
            return;
        }
    }

    if(warp_id>=upNum){
        return ;
    }
    volatile indexArr_t* index_section=&(g_tree.g_index_section[base+warp_id]);
    offset_t lastNodeId=last_base+warp_id*WARPSIZE; //last node id 指双层索引结构中的下标；
      
    key_t key=g_tree.g_index_section[lastNodeId+thread_id].key[0];

    if(lastNodeId+thread_id>=base){
        key=MAX_KEY;
    }
    //    nodeId=lastNodeId+thread_id;
    if(tid>=last_upnum){
        key=MAX_KEY;
       //     nodeId=-1;
    }
    index_section->isLast=0;
    index_section->key[thread_id]=key;
        //index_section->children[thread_id]=nodeId;  使用prefix sum

    g_tree.g_prefix_sum[base+warp_id]=lastNodeId;

    if(thread_id%(ORDER/index_size)==0){
        index_section->index[thread_id/(ORDER/index_size)]=key;
/*        if(key==0){

                printf("~~~~~~~~Inner~~~ thread_id:%d ,layer:%d, key:%d, warpId:%d, lastNodeId:%d, lastMinKey:%d, nodeId:%d\n",thread_id,layer,key,warp_id,lastNodeId,g_tree.g_index_section[lastNodeId+thread_id].key[0],base+warp_id);
        }
*/
    }
    __threadfence();
    if(upNum==1&&warp_id==0&&thread_id==0){
        *(g_tree.g_index_root) = base;
        printf("leaf size: %d, first_leaf_Id: %d, root_id:%d,firstLeafMinKey:%d\n",leaf_size,*(g_tree.g_first_leaf),base,g_tree.g_key_section[*(g_tree.g_first_leaf)].key[0]);
    //    for(int i=0;i<ORDER;i++){
    //        printf("-- key[%d]= %d, \n",i,g_tree.g_index_section[base].key[i]);
    //    }
    }

    return ;

    /*}}}*/
}
__global__ void build_index_kernel(g_tree_t g_tree,int index_size){
/*{{{*/

    const int tid=blockDim.x * blockIdx.x + threadIdx.x; // thread id 所有issue的线程的index
    const int warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE; 
    const int thread_id=threadIdx.x % WARPSIZE;// thread id in a warp

    offset_t leaf_size=*(g_tree.g_leaf_size);
    offset_t warp_start=warp_id*WARPSIZE;
    if(warp_start>=leaf_size){
        return ;
    }
    key_t key=g_tree.g_leaf_section[warp_start+thread_id].minKey;
    offset_t nodeId=g_tree.g_leaf_section[warp_start+thread_id].nodeId;

    // node 结构：
    //keys      [k,k,k,...,k,MAX]   31个key
    //child     [id,id,id...,id]    32个child

    if(tid>=leaf_size||warp_start+thread_id>=leaf_size){
        key=MAX_KEY;
        nodeId=-1;
    }
   // if(thread_id ==WARPSIZE-1){ //每个node中最后一个key设置为MAX
   //     key=MAX_KEY; // 将最后一个warp中超过leaf size 的值设置为MAX KEY;
   // }
    volatile indexArr_t *index_section=&(g_tree.g_index_section[warp_id]);
    
    index_section->isLast=1; //倒数第二层节点置位1， 此时需要node的child中记录的nodeid为g_key_section中的数组索引
    __threadfence();
    __syncwarp();
    index_section->key[thread_id]=key;
    __threadfence();
    index_section->children[thread_id]=nodeId;  //使用prefix_sum，但是最后一层需要child指针， 只有最后一层使用children去找leaf node
    //  最后一层index 的prefixsum
    
    g_tree.g_prefix_sum[warp_id]=-1; //表示当前是最后一层索引
    __threadfence();
    if(thread_id%(ORDER/index_size)==0){
        index_section->index[thread_id/(ORDER/index_size)]=key;
        if(key==0){

            printf("~~~~~~~~lastInner~~~ thread_id:%d , key:%d, nodeId:%d\n",thread_id,key,warp_id);
        }
    }

    __threadfence();

    return;

/*}}}*/
}

__global__ void collect_leafnodeInfo(g_tree_t g_tree){
/*{{{*/
    offset_t cur_leaf=*(g_tree.g_first_leaf);
    int count=0;
    while(cur_leaf!=-1){
        key_t minKey=g_tree.g_key_section[cur_leaf].key[0];
        g_tree.g_leaf_section[count].minKey=minKey;
        g_tree.g_leaf_section[count].nodeId=cur_leaf;
        cur_leaf=g_tree.g_next_section[cur_leaf].nodeid;
        count++;
    }
    *(g_tree.g_leaf_size)=count;
    __threadfence();
   // printf(".......leaf size: %d, lastNodeMinKey: %d, nodeid:%d ,\n",count,g_tree.g_leaf_section[count-1].minKey,g_tree.g_leaf_section[count-1].nodeId);
    __syncwarp();

/*}}}*/
}
__global__ void query_kernel( g_tree_t g_tree,
        g_data_t g_data,
        int index_size,
        int boundary,       //从第boundary个tx开始
        int size           //算上boundary大小的size (这个size是以mission个数为单位.) 
        ){
    /*{{{*/
    //const int tx_id = ((blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE) * query_queue_length + boundary;
    
    //const int tx_thread_id = (gridDim.x * blockDim.x + threadIdx.x) % ORDER;
    const int tx_id_in_block = threadIdx.x / WARPSIZE;
    
    //if (tx_id * tx_size_get >= size) return; //把一整个warp都在size外面的扔掉.
    
    //const int id_in_ntg_get = threadIdx.x % ntg_size_get;
    //const int ntg_id_in_block_get = threadIdx.x / ntg_size_get;
    //const int ntg_id_in_tx_get = threadIdx.x % WARPSIZE / ntg_size_get;
    __shared__ int cacheNodeId_ss[Block_Dim_Get/WARPSIZE];
    int &cacheNodeId_s = cacheNodeId_ss[tx_id_in_block];
    cacheNodeId_s = -1;
    __syncwarp();
    bool isLastThread = (threadIdx.x % WARPSIZE == WARPSIZE-1); //判断是不是一个warp里最后一个thread
    const int base = boundary * WARPSIZE + ((blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE * tx_size_get * query_queue_length) + threadIdx.x % WARPSIZE; 


    key_t last_target_key1 = 0;
    for (int i=0; i<query_queue_length;i++) { 
        

        const int mission_id_get = base + i*tx_size_get;        //this value only worked for ntg=1
        //const int mission_id_get = base + i*WARPSIZE;        //this value only worked for ntg=1
        const int mask_get = __ballot_sync(0xffffffff, mission_id_get<size);
        if (mission_id_get >= size) return;
        if (mission_id_get == size-1) isLastThread = true;
       
        
     

        //mission_t mission_get;
        key_t target_key_get;
        ans_t *ans_get;
        
#ifdef ABORT_COUNT
        //init 
        g_data.g_roll_back_count[mission_id_get].roll_back_times = 0;
        g_data.g_roll_back_count[mission_id_get].traversal_steps = 0;
        g_data.g_roll_back_count[mission_id_get].leaf_traversal_steps = 0;
        __syncwarp();
#endif        
        
        //mission_get = g_data.g_mission[mission_id_get]; 
        int idx = g_data.g_idx[mission_id_get];
        target_key_get = g_data.g_keys[idx];
        ans_get = &(g_data.g_ans[idx]);   
        
          
        offset_t local_root = *(g_tree.g_index_root);
        if (local_root == -1) {
            *ans_get = -1;
            return ;
        }
        //*ans_get=-1;
        
        //printf("there! id:  %d,   %d, %d\n", mission_id_get, target_key_get, last_target_key1);
        //printf("there! id:  %d, l_id  %d,  %d, %d, %d\n", mission_id_get,i, target_key_get, last_target_key1, cacheNodeId_s);
        index_search_with_cache(target_key_get, last_target_key1, *ans_get,local_root, cacheNodeId_s, isLastThread);
        //printf(" here! id:  %d, l_id  %d,  %d, %d, %d\n", mission_id_get,i, target_key_get, last_target_key1, cacheNodeId_s);
        //printf("here!  id:  %d,   %d, %d\n", mission_id_get, target_key_get, last_target_key1);
        //printf("here!   %d, %d\n", target_key_get, last_target_key1);
        //radical_search(target_key_get, *ans_get, local_root); 

    }

    
    return;

/*}}}*/

}

__global__ void insert_kernel(g_tree_t g_tree,g_data_t g_data,g_lock_t g_lock,int boundary,int maxSize){ 
/*{{{*/
// 只在leafnode 插入
#ifdef ONE_KERNEL
    const int size=maxSize;
#else 
    const int size=(boundary*tx_size_get<maxSize)?boundary*tx_size_get:maxSize;
#endif
    const int warp_id =(blockDim.x* blockIdx.x+threadIdx.x) / WARPSIZE;
    const int thread_id= threadIdx.x %WARPSIZE;
    const int warp_id_in_block = threadIdx.x / WARPSIZE;
    if(warp_id*insert_queue_length>=size) return;

    __shared__ int cacheNodeId_ss[Block_Dim_Put / WARPSIZE];
    int &cacheNodeId_s =cacheNodeId_ss[warp_id_in_block];
    cacheNodeId_s=-1;
    __syncwarp();
   // bool isLastThread=(threadIdx.x % WARPSIZE==WARPSIZE - 1); // 1个warp处理同时只处理1个请求，不需要
    offset_t local_root =*(g_tree.g_index_root); // 使用prefix-sum 

    if(local_root==-1){
        printf("unsupported for empty tree!!!, please use tx_insert_kernel() to build the tree!!! \n");
        return;
    }
    
    __shared__ offset_t nodeId_s[Block_Dim_Put/ WARPSIZE];
    offset_t &nodeId=nodeId_s[warp_id_in_block];
    __shared__ offset_t new_nodeId_s[Block_Dim_Put/WARPSIZE];
    offset_t &new_loc=new_nodeId_s[warp_id_in_block];
    new_loc=-1;
    __shared__ char target_thread_s[Block_Dim_Put / WARPSIZE];
    char &target_thread = target_thread_s[warp_id_in_block];
    char selfFlag;
    offset_t old_nodeId;

    const int base=warp_id* insert_queue_length;
    key_t last_target_key;  //for buffer
    for(int l_i=0;l_i<insert_queue_length;l_i++){
        const int mission_id = base+l_i;
        key_t target_key;
        mission_t mission;
        offset_t target_val;
        ans_t * ans;
#ifdef ABORT_COUNT 
        g_data.g_roll_back_count[mission_id].roll_back_times = -1;
        g_data.g_roll_back_count[mission_id].traversal_steps = 0;
        g_data.g_roll_back_count[mission_id].leaf_traversal_steps = 0;
        __syncwarp();
#endif 
        if(mission_id >= size) return;

        int idx = g_data.g_idx[mission_id];
        target_key = g_data.g_keys[idx];
        mission = g_data.g_mission[mission_id];
        target_val = g_data.g_vals[idx];
        ans= &(g_data.g_ans[idx]);

        //key_t key;
        //offset_t val;
        __syncwarp();
        int isLast;
#ifndef NO_BUFFER 
        if(cacheNodeId_s>=0&& mission==INSERT){
            nodeId= cacheNodeId_s;
            __syncwarp();
            //insert这里是一个warp处理一个任务.
            //printf("test:%d:  %d - %d = %d\n",threadIdx.x, target_key, last_target_key, target_key - last_target_key);
#ifdef USING_STEP_THRESHOLD
            if (target_key - last_target_key < BUFFER_THRESHOLD) 
#endif
                goto search_list;
        }
#endif
        nodeId=local_root;
        __syncwarp();
        isLast=g_tree.g_index_section[nodeId].isLast;
        key_t key;
        offset_t val;
        key_t right_min;
        key_t moved_key;
        offset_t moved_val;
        while(isLast==0){
            selfFlag=1;
            key=g_tree.g_index_section[nodeId].key[thread_id];
            __syncwarp();

#ifdef ABORT_COUNT
            if (thread_id == 0)
                g_data.g_roll_back_count[mission_id].traversal_steps++;
#endif 


            if(target_key<key){
                selfFlag=0;   
            }
            __syncwarp();
            for(int off=WARPSIZE/2;off>0;off/=2){
                selfFlag+=__shfl_down_sync(0xFFFFFFFF,selfFlag,off,WARPSIZE);
            }
            __syncwarp();
            if(thread_id==0){
                if(selfFlag==0){
                    nodeId=__ldg(&(g_tree.g_prefix_sum[nodeId]));
                }else{
                    nodeId=__ldg(&(g_tree.g_prefix_sum[nodeId]))+selfFlag-1;
                }
            
            }
            __syncwarp();

            isLast=g_tree.g_index_section[nodeId].isLast;
        }
        
        // 搜索最后一层index    last_inner_node
        
        selfFlag=1;
        key=g_tree.g_index_section[nodeId].key[thread_id];
    
#ifdef ABORT_COUNT
        if (thread_id == 0)
            g_data.g_roll_back_count[mission_id].traversal_steps++;
#endif 

        if(target_key<key){
            selfFlag=0;
        }
        __syncwarp();
        for(int off=WARPSIZE/2;off>0;off/=2){
            selfFlag+=__shfl_down_sync(0xffffffff,selfFlag,off,WARPSIZE);
        }
        __syncwarp();
        if(thread_id==0){
            if(selfFlag==0){
                nodeId=g_tree.g_index_section[nodeId].children[selfFlag];
            }else{
                nodeId=g_tree.g_index_section[nodeId].children[selfFlag-1];
            }
        }
        __syncwarp();
       // continue;
     
search_list:
        old_nodeId=nodeId;

leaf_start:

#ifdef ABORT_COUNT
        if (thread_id == 0) {
            g_data.g_roll_back_count[mission_id].roll_back_times++;
        }
        __threadfence();
#endif  
        nodeId=old_nodeId;
        __threadfence();
       // __syncwarp();
        // 获取leafnode信息之前需要获取锁
        offset_t minKey= MAX_KEY;
       // offset_t nextLeaf=-1;
        key_t nodeInfo;
        minKey=g_tree.g_val_section[nodeId].val[ORDER-1];

#ifdef ABORT_COUNT
        //第一个Leafnode
        if (thread_id == 0)  {

            g_data.g_roll_back_count[mission_id].traversal_steps++;
            g_data.g_roll_back_count[mission_id].leaf_traversal_steps++;
        }
#endif 
        
        while(target_key>=minKey){
            nodeId=g_tree.g_next_section[nodeId].nodeid;
            if(nodeId<0 || nodeId>=MAX_NODE_NUM){
                goto leaf_start;
            }
            nodeInfo=g_tree.g_key_section[nodeId].key[ORDER-1];
            if(getNodeLevel(nodeInfo)!=0){
                goto leaf_start;
            }
            //minKey=MAX_KEY;
            minKey=g_tree.g_val_section[nodeId].val[ORDER-1];

#ifdef ABORT_COUNT
            if (thread_id == 0) { 
                    
                g_data.g_roll_back_count[mission_id].traversal_steps++;
                
                g_data.g_roll_back_count[mission_id].leaf_traversal_steps++;
            }
#endif 
        }

        if(thread_id==0){
            
                version_t new_lock= constructLock(0,true);
                version_t old_lock= constructLock(0,false);
                if(atomicCAS((version_t*)&(g_lock.g_lock_table[nodeId]),old_lock,new_lock)!=old_lock){
                //    printf("lock fail\n");
                    goto leaf_start;
                }
            
        }

        __syncwarp();
        key_t firstKey=g_tree.g_key_section[nodeId].key[0];
        key_t superKey=g_tree.g_val_section[nodeId].val[ORDER-1];
        if((target_key<firstKey&&nodeId!=0) || target_key>= superKey ){
            g_lock.g_lock_table[nodeId]=constructLock(0,false);
            __threadfence();
            goto leaf_start;
        }
        

        __syncwarp();
       // if(thread_id==0)
       //     printf("找到leafnode %d\n",mission_id);
        nodeInfo=g_tree.g_key_section[nodeId].key[ORDER-1];

        key=g_tree.g_key_section[nodeId].key[thread_id];
        val=g_tree.g_val_section[nodeId].val[thread_id];
      
        __syncwarp();
        *ans=-1;
        __syncwarp();
        if(thread_id==ORDER-1){
            key=MAX_KEY;
        } 
        if(key==target_key){
            *ans=val;
            val = target_val;
        }
        __syncwarp();
       
        if(thread_id==ORDER-1){
            key=nodeInfo;
        } 
        __syncwarp();
        if(mission==GET){
            goto next_mission;
        }
        if(__any_sync(0xffffffff,key==target_key)){
            g_tree.g_key_section[nodeId].key[thread_id]=key;
            g_tree.g_val_section[nodeId].val[thread_id]=val;
            goto next_mission;
        }

        //寻找正确的位置
        selfFlag=1;
        if(thread_id==ORDER-1){
            key=MAX_KEY;
        } 
        if(target_key<key){
            selfFlag=0;
        }
        __syncwarp();
        for(int off=WARPSIZE/2;off>0;off/=2){
            selfFlag+=__shfl_down_sync(0xffffffff,selfFlag,off,WARPSIZE);
        }
        __syncwarp();
        if(thread_id==0){
            target_thread=selfFlag;
        }
        __syncwarp();
            
        key_t tmp_key;
        offset_t tmp_val;
        moved_key= __shfl_up_sync(0xffffffff,key,1,ORDER);
        moved_val= __shfl_up_sync(0xffffffff,val,1,ORDER);
        if(thread_id==target_thread){
            tmp_key= target_key;
            tmp_val=target_val;
        }else if(thread_id> target_thread){
            tmp_key=moved_key;
            tmp_val=moved_val;
        }else{
            tmp_key=key;
            tmp_val=val;
        }
        __syncwarp();
        
        right_min=val;
        __syncwarp();
        if(getNodeSize(nodeInfo)==ORDER-1){
            __syncwarp();
            //创建新的leafnode,并将两个node 中的key平均分配，添加next指针
            key_t up_key=__shfl_sync(0xffffffff,tmp_key,ORDER/2,ORDER);
            key=MAX_KEY;
            val=-1;
            if(thread_id<ORDER/2){
                key=tmp_key;
                val=tmp_val;
            }else if(thread_id==ORDER-1){
                key=info(ORDER/2,0);
                val=up_key;
            }
            __syncwarp();
            g_tree.g_key_section[nodeId].key[thread_id]=key;
            g_tree.g_val_section[nodeId].val[thread_id]=val;
            __syncwarp();
           

            //new node 
            key=MAX_KEY;
            val=-1;
            tmp_key=__shfl_down_sync(0xffffffff,tmp_key,ORDER/2,ORDER);
            tmp_val=__shfl_down_sync(0xffffffff,tmp_val,ORDER/2,ORDER);
            if(thread_id<(ORDER/2)){
                key=tmp_key;
                val=tmp_val;
            }else if(thread_id==ORDER-1){
                key=info(ORDER/2,0);
                val=right_min;
            }
            __syncwarp();
        
            if(thread_id==0){
            
                new_loc=atomicAdd((offset_t*)(g_tree.g_tree_size),1);
               // offset_t tmp_loc=new_loc;
                g_lock.g_lock_table[new_loc]=constructLock(0,false);
                offset_t old_next=g_tree.g_next_section[nodeId].nodeid;
                g_tree.g_next_section[nodeId].nodeid=new_loc;
                g_tree.g_next_section[new_loc].nodeid=old_next;
            }
            __syncwarp();
            g_tree.g_key_section[new_loc].key[thread_id]=key;
            g_tree.g_val_section[new_loc].val[thread_id]=val;
            __syncwarp();
           
        }else{
       
            if(thread_id==ORDER-1){
                tmp_key= infoSizeInc(nodeInfo);
                tmp_val=val;
            }
            __syncwarp();
            g_tree.g_key_section[nodeId].key[thread_id]=tmp_key;
            g_tree.g_val_section[nodeId].val[thread_id]=tmp_val;
        }
next_mission:
        if(thread_id==0){
           // version_t old_lock=constructLock(0,true);
          //  version_t new_lock=  
            g_lock.g_lock_table[nodeId]=constructLock(0,false);

          //  atomicCAS((version_t *)&(g_lock.g_lock_table[nodeId]),old_lock,new_lock);
           // printf("释放锁 %d\n",nodeId);
        }
        cacheNodeId_s=nodeId;
        last_target_key = target_key;       //for buffer threshold
        __syncwarp();
    }


/*}}}*/
}

__global__ void tx_insert_kernel(g_tree_t g_tree, 
        g_data_t g_data,
        g_tx_t g_tx,
        int boundary,
        int maxSize
        ) {
/*{{{*/   
#ifdef ONE_KERNEL
    const int size=maxSize;
#else 
    const int size=(boundary*tx_size_get<maxSize)?boundary*tx_size_get:maxSize;
#endif
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

        //这个函数要对 tmpCurNodeId, nodeInfo, key, val, target_thread_s 进行更新。
        //last系列的不需要。如果node是满的会直接返回false.
        offset_t local_root1 = *(g_tree.g_root);
        wrtset_s.size = 0;

        
        if (radical_search_for_insert(target_key, local_root1)==true) {
            goto S_WRITE; 
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
       
        if (mission == GET) {       //理论上这么应该不会再触发了
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
/*}}}*/
}



void GPU_STM_Tree::launchBuildIndexKernel(){
    
    dim3 grid_dim1(1);
    dim3 block_dim1(1);
    collect_leafnodeInfo<<<grid_dim1, block_dim1>>>(g_tree);
    cout<< "build index... Middle!"<<endl;
    dim3 grid_dim2(Grid_Dim_prefix*2);
    dim3 block_dim2(Block_Dim_Get);
    build_index_kernel<<<grid_dim2, block_dim2>>>(g_tree,index_size);

    for(int i=0;i<6;i++){
        build_index_up<<<grid_dim2,block_dim2>>>(g_tree,index_size,i);
    }
    
    cout<< "build index...Done! total threads Num:"<<Grid_Dim*Block_Dim_Get<<endl;
    cudaError_t error=cudaGetLastError();
    printf("cuda error: %s\n",cudaGetErrorString(error));

}





void GPU_STM_Tree::launchKernel(int size, bool special, Time_Measure &t) {
#ifdef ONE_KERNEL
    int grid_dim1 = (h_device_working_num2 + (Block_Dim_Put-1)) / Block_Dim_Put;
    int grid_dim2=0;
#else 
    int grid_dim1 = (h_boundary * WARPSIZE + (Block_Dim_Put-1)) / Block_Dim_Put;
    int grid_dim2 = (((h_device_working_num2 - h_boundary * tx_size_get) * ntg_size_get / query_queue_length  + 1)  + (Block_Dim_Get-1)) / Block_Dim_Get;

#endif
      //h_boundary * tx_size_get  insert处理了多少个任务.
    
    //cout<<"grid_dim1: "<<grid_dim1<<endl;
    //cout<<"grid_dim2: "<<grid_dim2<<endl;
    /////cout<<"working size:      "<<h_device_working_num2<<endl;
    //cout<<"boundary:          "<<h_boundary*tx_size_get<<endl;
    t.gpuTimeStart();
    if (grid_dim1 != 0){
        if(insert_count<2){
            //cout<<"tx_insert_kernel"<<endl;
            tx_insert_kernel<<<grid_dim1, Block_Dim_Put>>>(g_tree, g_data, g_tx, h_boundary, h_device_working_num2);
            //cudaError_t error=cudaGetLastError();
            //printf("cuda error in after tx_insert_kernel: %s\n",cudaGetErrorString(error));
        }else{
            //cout<<"insert_kernel"<<endl;
            insert_kernel<<<grid_dim1*(tx_size_get/insert_queue_length),Block_Dim_Put>>>(g_tree,g_data,g_lock,h_boundary,h_device_working_num2);
            //cudaError_t error=cudaGetLastError();
            //printf("cuda error in after insert_kernel: %s\n",cudaGetErrorString(error));
        }
    }
    t.gpuTimeEnd();
//    cout<<"insert done"<<endl;

    t.gpuTimeStart();
    if (grid_dim2 !=0){
        query_kernel<<<grid_dim2, Block_Dim_Get>>> (g_tree, g_data,index_size, h_boundary, h_device_working_num2);
        //search_kernel<<<grid_dim2, Block_Dim_Get>>> (g_tree, g_data, h_boundary, h_device_working_num2);

    }
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
    //dim3 grid_dim( (size + tmpBlockDim-1) / tmpBlockDim );
    dim3 grid_dim( (h_device_working_num2 + tmpBlockDim-1) / tmpBlockDim );
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
    dim3 block_dim(Block_Dim_Get / WARPSIZE * tx_size_get * query_queue_length);
    writebackAns<<<grid_dim, block_dim>>>(g_data_phase_3, g_data, size);
}

__global__ void preprocessKernel(g_tree_t g_tree, g_tx_t g_tx,g_lock_t g_lock){
    
    *(g_tree.g_root) = -1;
    *(g_tree.g_tree_size) = 0;

    *(g_tree.g_first_leaf) = -1;
    *(g_tree.g_leaf_size)=0;
    *(g_tree.g_index_root) = -1;
   
    g_tx.g_lock_table[MAX_NODE_NUM-1] = 0;    //root lock 初始化
    for(int i=0;i<MAX_NODE_NUM;i++){
        g_tree.g_next_section[i].nodeid=-1;
        g_lock.g_lock_table[i]=0; //所有node lock初始化
    }
    
}
__global__ void preprocessKernel_idx_init(int *g_idx_bef) {
    g_idx_bef[blockDim.x * blockIdx.x + threadIdx.x] = blockDim.x * blockIdx.x + threadIdx.x;
}




void GPU_STM_Tree::preprocess() {
    dim3 grid_dim(1);
    dim3 block_dim(1);
    preprocessKernel<<<grid_dim, block_dim>>>(g_tree, g_tx,g_lock);
    dim3 grid_dim2(Grid_Dim);
    dim3 block_dim2(Block_Dim_Get / WARPSIZE * tx_size_get * query_queue_length);
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
