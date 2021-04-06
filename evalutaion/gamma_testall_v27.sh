#!/bin/bash
test_size=50M
#test_set=(query_90_${test_size}.txt query_95_${test_size}.txt query_99_${test_size}.txt query_100_${test_size}.txt)
test_set=(query_95_50M.txt)

#test_set=(data_90_8M_R4_1_I9_1_shuf.txt data_95_8M_R4_1_I9_1_shuf.txt data_100_8M_R4_1_I1_1_shuf.txt)
#
#test_set=(data_100M_read.txt)
tree_size_list=(23 24 25 26)
test_date=$(date "+%Y%m%d-%H%M%S")
dist="gamma"
ver=27
echo "stm_test v$ver, $test_date"
for t_set in ${test_set[*]}
do
#    echo "version:$ver, tree size: $tree_size ,  test_size: $test_size\n"

    for tree_size in ${tree_size_list[*]}
    do
        echo "version:$ver, tree size: $tree_size ,  test_size: $test_size , test_file: $t_set\n"
        for i in {1..3}
        do 
            echo "测试文件：$t_set" " " "次数：$i"
            ../stm_test -v$ver -i ../dataset/input_data_$tree_size.txt -i ../dataset/${tree_size}_${dist}/$t_set
            echo -e "\n============================\n\n"
        done
    echo "\n+++++++++++++++++++++++++++++++++++\n"
    #echo "$t_set"
    #../stm_test -v9 -i ../dataset/input_data_23.txt -i ../dataset/$t_set
    #echo -e "\n"
    done
done

