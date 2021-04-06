#ifndef BPT_GPUSTM_KERNEL_DEF_H_39
#define BPT_GPUSTM_KERNEL_DEF_H_39




#define tx_read(nodeId, key, val)         \
            TX_READ(                        \
                nodeId,                         \
                tx_thread_id,                   \
                g_tree.g_key_section,           \
                g_tree.g_val_section,           \
                g_tx.g_lock_table,              \
                rdset_s,\
                key,                           \
                val                           \
                )
                



#define tx_read_root()                  \
            TX_READ_ROOT(\
                    g_tx.g_lock_table,\
                    g_tree.g_root,\
                    root_ver,\
                    local_root\
                    )

#define tx_write(nodeId, key, val)              \
            TX_WRITE(                           \
                    nodeId,                     \
                    tx_thread_id,               \
                    g_tx.g_lock_table,          \
                    wrtset,\
                    wrtset_s,\
                    rdset_s,                    \
                    key,                        \
                    val                         \
                    )

#define tx_write_root(new_root)                 \
            TX_WRITE_ROOT(                      \
                    g_tx.g_lock_table,          \
                    root_ver,                   \
                    local_root,                 \
                    new_root                    \
                    )


#define tx_commit(isWrtRoot, place_for_new_loc)     \
            TX_COMMIT(                              \
                    tx_thread_id,                   \
                    tx_id_in_block,                \
                    g_tree.g_key_section,           \
                    g_tree.g_val_section,           \
                    g_tx.g_lock_table,              \
                    wrtset,\
                    wrtset_s,\
                    isWrtRoot,                      \
                    g_tree.g_root,                  \
                    root_ver,                       \
                    g_tree.g_tree_size,             \
                    place_for_new_loc              \
                    )   

//============================================================================================

#define read_node(nodeId, key, val, nodeInfo)\
    READ_NODE(\
            nodeId,\
            key,\
            val,\
            nodeInfo,\
            tx_thread_id,\
            g_tx,\
            g_tree,\
            rdset_s\
            )


#define get_target(key, val, target_key, target_val, target_thread)\
    GET_TARGET(\
            key,\
            val,\
            target_key,\
            target_val,\
            target_thread,\
            tx_thread_id,\
            tx_id_in_block\
            )

#define create_new_tree()\
    CREATE_NEW_TREE(\
            tx_thread_id,\
            tx_id_in_block,\
            g_tx,\
            wrtset,\
            wrtset_s,\
            rdset_s,\
            root_ver,\
            local_root\
            )







#define node_splitting(nodeId, key, val, nodeInfo, up_key)\
    NODE_SPLITTING(\
            nodeId,\
            key,\
            val,\
            nodeInfo,\
            up_key,\
            tx_thread_id,\
            tx_id_in_block,\
            g_tx,\
            wrtset,\
            wrtset_s,\
            rdset_s\
            )



#define insert_node(nodeId, target_key, key, val, target_thread, nodeInfo, place_for_new_loc)\
    INSERT_NODE(\
            nodeId,\
            target_key,\
            key,\
            val,\
            target_thread,\
            nodeInfo,\
            place_for_new_loc,\
            tx_thread_id,\
            tx_id_in_block,\
            g_tx,\
            wrtset,\
            wrtset_s,\
            rdset_s\
            )

#define insert_new_root(target_key, left_child, new_level, place_for_new_loc)\
    INSERT_NEW_ROOT(\
            target_key,\
            left_child,\
            new_level,\
            place_for_new_loc,\
            tx_thread_id,\
            tx_id_in_block,\
            g_tx,\
            root_ver,\
            local_root,\
            wrtset,\
            wrtset_s,\
            rdset_s\
            )
            


#define insert_leaf(nodeId, nodeInfo, target_key, target_val, key, val, target_thread, up_key)\
    INSERT_LEAF(\
            nodeId,\
            nodeInfo,\
            target_key,\
            target_val,\
            key,\
            val,\
            target_thread,\
            up_key,\
            tx_thread_id,\
            tx_id_in_block,\
            g_tx,\
            wrtset,\
            wrtset_s,\
            rdset_s\
            )



#endif
