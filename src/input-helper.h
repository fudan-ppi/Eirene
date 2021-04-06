#ifndef INPUT_HELPER_H 
#define INPUT_HELPER_H 

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <limits.h>
#define offset_t int 

#define mission_t short

#define INSERT 0
#define GET 1
#define DELETE 2
#define RANGE 3

/*
typedef enum mission_t {
    INSERT, 
    DELETE,
    GET
}mission_t;
*/

/*
 * INSERT:
 *          inexist:    insert the new key and val, and -1(new version) //////return the new val (old version) /////  
 *          exist:      update the key using new val, and return  the old val 
 * DELETE:
 *          exist:      delete the key, and return the val 
 *          inexist:    return -1
 * GET: 
 *          exist:      return the val 
 *          inexist:    return -1
 * 
 */  
 //Read的格式： R key 
 //Put的格式： P key val 
 //Range的格式： G start end
typedef offset_t ans_t;



template<typename key_t, typename val_t> 
class Data_Collector {
        public:

            std::vector<key_t> keys;         //Original Data
            std::vector<offset_t> vals;      //Actually it is value's record number!
            std::vector<mission_t> mission;      //Actually it is value's record number!
            std::vector<val_t> *warehouse;    //True values are stored here!
            
            

            Data_Collector(std::vector<val_t> *wh):warehouse(wh) { 

            }
            /*
             * 读取文件， 把keys,mission存入vector keys, mission 
             * 对于mission==insert的, 将true value插入到warehouse中。并将这个true value所在的warehouse中的offset作为val插入vector vals中。
             * 对于其他情况，插入-1到vector vals进行占位
             * 返回Key的个数
             *       
             */ 
            int readFile( std::string fileName, mission_t m );
            int readFile( std::string fileName);
            void clearMissions();
            void test(ans_t *ans);

};

template class Data_Collector<int,int>;




template<typename key_t, typename val_t> 
int Data_Collector<key_t, val_t>::readFile( std::string fileName, mission_t m ) {

    /*{{{*/
    std::ifstream file; 
    file.open(fileName);

    std::string s;

    int i = 0;
    int size = warehouse->size();
    while (getline(file, s)){
        key_t key;
        std::stringstream ss;
        ss.str(s);
        
        if (!(ss>>key)) {
            continue;
        }

        
        if (m==INSERT) {
            val_t val;
            if (!(ss>>val)) {
                continue;
            }
            keys.push_back(key);
            mission.push_back(m);
            warehouse->push_back(key);  //在wareHouse存入要插入的value（这里我们假定val=key）
            vals.push_back(size);   //将这个value所在的Offset写入vector vals中
            size++;
        }
        else if (m==RANGE) {
            //for range, keys: key_start, vals: key_end
            val_t key2;
            if (!(ss>>key2)) {
                vals.push_back(INT_MAX);
            }
            else {
                vals.push_back(key2); 
            }
            keys.push_back(key);
            mission.push_back(m);
        }
        else {
            keys.push_back(key);
            mission.push_back(m);
            vals.push_back(-1);        //在非INSERT的情况下插入一个-1来占位
        }
        i++;
    }
   
    return i;       //return mission number
/*}}}*/
}

template<typename key_t, typename val_t>
int Data_Collector<key_t, val_t>::readFile(std::string fileName){
   /*{{{*/
    std::ifstream file; 
    file.open(fileName);

    std::string s;

    int i = 0;
    int size = warehouse->size();


    //Don't use sscanf, type checking is a problem!
    //use sscanf_s (MSV only) or stringstream (C++)
    while (getline(file, s)){
        key_t key;
        val_t val;
        
        std::stringstream ss;
        ss.str(s);
        char type;
        ss>>type;
        if (type == 'R') {
            if (!(ss>>key)) {
                //std::cout << "ha? "<<ss.str()<<" "<<key<<std::endl;
                continue;
            }
            keys.push_back(key);
            mission.push_back(GET);
            vals.push_back(-1);
            
        }
        else if (type == 'P') {
            if (!(ss>>key && ss>>val)) {
                continue;
            }
            //std::cout << "ha? "<<ss.str()<<" "<<key<<" "<<val<<std::endl;
            keys.push_back(key);
            mission.push_back(INSERT);
            warehouse->push_back(val);  //在wareHouse存入要插入的value
            vals.push_back(size);   //将这个value所在的Offset写入vector vals中
            size++;
        }
        else if (type == 'G') {
            if (!(ss>>key && ss>>val)){
                continue;
            }
            keys.push_back(key);        //start key
            mission.push_back(RANGE);
            if (val == INT_MAX) val--;  //INT_MAX作为end key会有点麻烦..
            vals.push_back(val);        //end key
        }
        
        i++;
    }


    return i;       //return mission number
/*}}}*/

}

template<typename key_t, typename val_t>
void Data_Collector<key_t, val_t>::clearMissions(){
    keys.clear();
    vals.clear();
    mission.clear();
}


template<typename key_t, typename val_t> 
void Data_Collector<key_t, val_t>::test(ans_t *ans){
/*{{{*/
    int task_size = keys.size();
    //for (int i = task_size/2; i< task_size; i++) {
    for (int i = 0; i< task_size; i++) { 
        if (ans[i]!=-1)
            std::cout<<"key: "<<keys[i]<<"\t operator "<<mission[i]<<"\t ans: "<<ans[i]<<"\t vals: "<<(*warehouse)[ans[i]]<<std::endl;
        else 
            std::cout<<"key: "<<keys[i] <<"\t operator "<<mission[i]<<"\t ans: "<<ans[i]<<std::endl;
        //std::cout<<keys[i]<<std::endl;
    }
/*}}}*/
}




#endif 
