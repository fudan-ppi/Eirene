#ifndef BPT_GPUSTM_KERNEL_AUX_CUH_27
#define BPT_GPUSTM_KERNEL_AUX_CUH_27
#include "bpt-gpustm.h"
#include "../cuda_utils.h"

//把pos变成暂时的ans,  ans = -(pos+1)
namespace gpu_stm_nsp_27 {
    __forceinline__ __device__ int posToTmpAns(int pos) {
        return -(pos+1);
    }  

    __forceinline__ __device__ int TmpAnsToPos(int ans) {
        return -(ans+1);
    }  

    __global__ void mergeMission(g_data_phase_2_t phase_2, g_data_phase_3_t phase_3, g_data_phase_4_t phase_4, g_data_t ret){

        int id = blockDim.x * blockIdx.x + threadIdx.x ;
        int new_size = *(phase_3.g_num);
        
        if (id >= new_size) return;

        int count = phase_3.g_count[id];

        mission_t passed_mission = GET;
        ans_t forNextAns = posToTmpAns(id);

        for (int i=0; i<count; i++) {
            int idx = phase_2.g_idx[__ldg(&phase_3.g_idx[id]) + i];     //当前处理的任务对应到原input的idx
            int idx1 = __ldg(&phase_3.g_idx[id]) + i;     //
            mission_t m = phase_3.g_mission[idx];
            
            phase_3.g_ans[idx1] = forNextAns;
            if (m==INSERT) {
                passed_mission = m;
                forNextAns = phase_3.g_vals[idx];
            }

        }
        ret.g_vals[id] = forNextAns; 
        phase_4.g_mission[id] = passed_mission;
        

    }
    //tx_kernel函数一律，找到返回找到的val,未找到返回-1
    __global__ void writebackAns(g_data_phase_3_t phase_3, g_data_t data, int size){
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id >= size) return;

        if (phase_3.g_ans[id] < 0) {
            int pos = TmpAnsToPos(phase_3.g_ans[id]);
            ans_t true_ans = data.g_ans[pos];
            phase_3.g_ans[id] = true_ans;

        }
    }
    
    //第一个get在第boundary个tx中.
    __global__ void getBoundary0(volatile int * boundary) {
        *boundary = -1;
    }
    __global__ void getBoundary1(mission_t *g_mission, volatile int *boundary, int size) {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        int mask = __ballot_sync(0xffffffff, id<size);
        if (id >= size) return;
        if (id > 0 && g_mission[id-1] == INSERT && g_mission[id] == GET) {
            *boundary = (id + tx_size_get-1) / tx_size_get; 
        }
        return;
    }
    __global__ void getBoundary2(mission_t *g_mission, int *boundary, int size) {
    
        if (*boundary==-1 && g_mission[0]==INSERT ) {
            *boundary = (size + tx_size_get-1) /tx_size_get;
        }
        else if(*boundary==-1 && g_mission[0]==GET) {
            *boundary = 0;
        }

        return; 
    }

}
#endif

