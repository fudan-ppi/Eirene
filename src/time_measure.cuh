#ifndef TIME_MEASURE_CUH
#define TIME_MEASURE_CUH

#include <cuda_runtime.h>
#include <vector>
class Time_Measure{




    std::vector<cudaEvent_t> g_start;
    std::vector<cudaEvent_t> g_stop;
    public:
    void gpuTimeInit(){
        
    }
    void gpuTimeStart(){
        g_start.emplace_back();
        CUDA_ERROR_HANDLER(cudaEventCreate(&(g_start.back())));
        g_stop.emplace_back();
        CUDA_ERROR_HANDLER(cudaEventCreate(&(g_stop.back())));

        CUDA_ERROR_HANDLER(cudaEventRecord(g_start.back()));

    };
    void gpuTimeEnd(){
        CUDA_ERROR_HANDLER(cudaEventRecord(g_stop.back()));
    };

    float getGPUTime(){
        float time = 0;
        int size = g_start.size();
        for (int i=0;i<size;i++) {
            CUDA_ERROR_HANDLER(cudaEventSynchronize(g_start[i])); 
            CUDA_ERROR_HANDLER(cudaEventSynchronize(g_stop[i])); 
            float time_gpu = 0;
            CUDA_ERROR_HANDLER(cudaEventElapsedTime(&time_gpu, g_start[i], g_stop[i]));
       // printf("time_phase_%d: %f\n",i, time_gpu/1000);
            time += time_gpu;
        }
        return time/1000;
    }
};


#endif
