#ifndef TREE_H
#define TREE_H 

#include <stdlib.h>
#include <iostream>
#include "input-helper.h"
/*
 * 
 *               key[i]     key[i+1]
 *         child[i]   child[i+1]   child[i+2]      
 * 
 *          
 * 
 * child[i+1]--> range: [ key[i], key[i+1] )
 * 
 * 
 * 
 * 
 */  








class GPU_BTree {
    public: 

        virtual ~GPU_BTree(){};
        virtual void prepareGPU()=0;
        virtual void prepareInput(std::string fileName)=0;
        virtual void prepareInput(std::string fileName, mission_t m){};
        virtual void gpu_work()=0;
        virtual void gpu_work_sp(){};   //为可能的特殊的任务留一个虚函数。 sp for special
        virtual void gpu_work_range(){};   //为可能的特殊的任务留一个虚函数。 sp for special
        virtual void test()=0;
        virtual void test_range(){};



};


#endif
