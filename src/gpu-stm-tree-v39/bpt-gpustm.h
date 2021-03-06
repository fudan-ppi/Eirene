#ifndef BPT_GPUSTM_H_39 
#define BPT_GPUSTM_H_39

#include <cuda_runtime.h>
#include <limits>
#include "../input-helper.h"
#include "../tree.h"
#include "../global_conf.h"
#include "../cuda_utils.h"
#include "../time_measure.cuh"

namespace gpu_stm_nsp_39{
    typedef int key_t;
    typedef int val_t;

    //typedef short version_t;    //bit 0-14: version number; bit 15: lock
    typedef unsigned int version_t;    //cuda CAS不支持16位！！！
    const char* const formatString="%d";
  
    const int MAX_KEY = INT_MAX; 
    const int ORDER = 32;
    const int WARPSIZE = 32;
    const int MAX_NODE_NUM = 10000000;
    
    const int SM = 80;
    const int Block_Per_SM = 512;  //512*24
    const int Block_Dim = WARPSIZE * 2;    //32*16=512
    const int Grid_Dim = Block_Per_SM * SM;     //计划中如果全是get的话的grid_dim。实际的grid_dim要后面再算


    const int insert_queue_length = 16;
    //const int tx_size = 1; // 这里这个值只能为1，代表一个TX一次处理一个任务.
    
    const int tx_num = Block_Dim * Grid_Dim / WARPSIZE * insert_queue_length;   //可以处理多少TX
                            //(默认由一个warp处理一个TX)


    const int batch_size = tx_num ;

    const int wrtSetSize = 3;
    typedef struct keyArr_t{
        key_t key[ORDER];
    }keyArr_t; 
    //规定key[ORDER-1] 
    // key[ORDER-1] & 0X1  : isLeaf                 (1bit)
    // key[ORDER-1] >> 1  :   node_size   (5bit)

    //actually value is offset_t as internal node's child and leafnode's record offset
    typedef struct valArr_t{
        offset_t val[ORDER];
    }valArr_t;  
    

    typedef struct g_tree_t {
        volatile offset_t *g_root;
        volatile offset_t *g_tree_size;
        volatile keyArr_t *g_key_section;
        volatile valArr_t *g_val_section;
         
    }g_tree_t; 
   
    
#ifdef ABORT_COUNT
    typedef struct count_t {
        int roll_back_times;
        int traversal_steps;
        int leaf_traversal_steps;
    } count_t;
#endif 



    typedef struct g_data_t {
        key_t *g_keys;      
        offset_t *g_vals;   
        mission_t *g_mission;  
        ans_t *g_ans;

        //g_keys, g_vals,  g_ans, g_mission一一对应.
        
#ifdef ABORT_COUNT
        count_t *g_roll_back_count;
#endif 
    }g_data_t;
    

    //TODO: use tx_num instead of batch_size 
    typedef struct g_tx_t {

        volatile version_t *g_lock_table;            // sizoef(version_t) * MAX_NODE_NUM
                                        //g_lock_table[MAX_NODE_NUM-1] is used for root
        

    } g_tx_t;

    typedef struct rdset_t {
        offset_t loc[2];
        version_t ver[2];
        bool offset;
    } rdset_t;

    //for Warp 
    typedef struct wrtset_t {
        offset_t loc[3];
        version_t ver[3];
        int size;
    } wrtset_t;

    //for one thread
    typedef struct wrtset_content_t{
        key_t key[3];
        val_t val[3];
    } wrtset_content_t; 



    class GPU_STM_Tree : public GPU_BTree {

        public:

            std::vector<val_t> warehouse;
            Data_Collector<key_t, val_t> dc;
            
            int gpu_start;

            //gpu btree structure and host returned answer 
            //be initialized in prepareGPU series functions
            g_tree_t g_tree;      
            g_data_t g_data;
            g_tx_t g_tx;
            ans_t *h_ans = NULL;            //存排序后顺序的答案
            ans_t *h_ans1 = NULL;           //存排序前顺序的答案
#ifdef ABORT_COUNT
            count_t *h_roll_back_count = NULL;
#endif

            void prepareGPU();
            void prepareInput(std::string fileName);
            void prepareInput(std::string fileName, mission_t m);
            void gpu_work();
            void gpu_work_sp();
            void gpu_emulate_query_per_conflict();
            void test();

            ~GPU_STM_Tree(){} 
            
            GPU_STM_Tree():gpu_start(0),dc(&warehouse){}
            GPU_STM_Tree(int _gpu_start):gpu_start(_gpu_start),dc(&warehouse){}
        protected: 
            void prepareGPU_tree();        //prepare tree on GPU  
            void prepareGPU_data();     //malloc data info on GPU and malloc return data on CPU
            void prepareGPU_tx();           //malloc tx structures on GPU
            void transferData(int start, int size);
            void transferAns(int start, int size);
            
            void launchKernel(int size, bool special, Time_Measure &t);        //size :  mission numbers

            void gpu_work_main(bool special);
            
            //don't consider transfer time (no double buffer or pipeline)
            void preprocess();    // initialize  


    };
}




#endif 
