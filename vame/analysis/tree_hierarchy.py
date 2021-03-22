#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:23:43 2020

@author: luxemk
"""

import numpy as np
import networkx as nx
import random

from pathlib import Path
from matplotlib import pyplot as plt

from apted import APTED
from apted.helpers import Tree

from vame.util.auxiliary import read_config

def hierarchy_pos(G, root=None, width=.5, vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def merge_func(transition_matrix, n_cluster, motif_norm, merge_sel):
    
    if merge_sel == 0:
        # merge nodes with highest transition probability
        cost = np.max(transition_matrix)
        merge_nodes = np.where(cost == transition_matrix)
        
    if merge_sel == 1:
            
        cost_temp = 100
        for i in range(n_cluster):
            for j in range(n_cluster):
                cost = motif_norm[i] + motif_norm[j] / np.abs(transition_matrix[i,j] + transition_matrix[j,i] )
                if cost <= cost_temp:
                    cost_temp = cost
                    merge_nodes = (np.array([i]), np.array([j]))                
            
    return merge_nodes


def graph_to_tree(motif_usage, transition_matrix, n_cluster, merge_sel=1):
    
    if merge_sel == 1:
        # motif_usage_temp = np.load(path_to_file+'/behavior_quantification/motif_usage.npy')
        motif_usage_temp = motif_usage
        motif_usage_temp_colsum = motif_usage_temp.sum(axis=0)
        motif_norm = motif_usage_temp/motif_usage_temp_colsum
        motif_norm_temp = motif_norm.copy()
    else:
        motif_norm_temp = None
    
    merging_nodes = []
    hierarchy_nodes = []
    trans_mat_temp = transition_matrix.copy()
    is_leaf = np.ones((n_cluster), dtype='int')
    node_label = []
    leaf_idx = []
    
    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1)==0)
        reduction = len(temp) + 1
    else:
        reduction = 1
    
    for i in range(n_cluster-reduction):
    
#        max_tr = np.max(trans_mat_temp) #merge function
#        nodes = np.where(max_tr == trans_mat_temp)
        nodes = merge_func(trans_mat_temp, n_cluster, motif_norm_temp, merge_sel)
        
        if np.size(nodes) >= 2:
            nodes = np.array([nodes[0][0], nodes[1][0]])
        
        if is_leaf[nodes[0]] == 1:
            is_leaf[nodes[0]] = 0
            node_label.append('leaf_left_'+str(i))
            leaf_idx.append(1)
        
        elif is_leaf[nodes[0]] == 0:
            node_label.append('h_'+str(i)+'_'+str(nodes[0]))
            leaf_idx.append(0)
            
        if is_leaf[nodes[1]] == 1:
            is_leaf[nodes[1]] = 0
            node_label.append('leaf_right_'+str(i))
            hierarchy_nodes.append('h_'+str(i)+'_'+str(nodes[1]))            
            leaf_idx.append(1)
            
        elif is_leaf[nodes[1]] == 0:
            node_label.append('h_'+str(i)+'_'+str(nodes[1]))
            hierarchy_nodes.append('h_'+str(i)+'_'+str(nodes[1]))
            leaf_idx.append(0)
            
        merging_nodes.append(nodes)

        node1_trans_x = trans_mat_temp[nodes[0],:]
        node2_trans_x = trans_mat_temp[nodes[1],:]
        
        node1_trans_y = trans_mat_temp[:,nodes[0]]
        node2_trans_y = trans_mat_temp[:,nodes[1]]
        
        new_node_trans_x = node1_trans_x + node2_trans_x
        new_node_trans_y = node1_trans_y + node2_trans_y
        
        trans_mat_temp[nodes[1],:] = new_node_trans_x
        trans_mat_temp[:,nodes[1]] = new_node_trans_y
        
        trans_mat_temp[nodes[0],:] = 0
        trans_mat_temp[:,nodes[0]] = 0
        
        trans_mat_temp[nodes[1],nodes[1]] = 0
        
        if merge_sel == 1:
            motif_norm_1 = motif_norm_temp[nodes[0]]
            motif_norm_2 = motif_norm_temp[nodes[1]]
            
            new_motif = motif_norm_1 + motif_norm_2
            
            motif_norm_temp[nodes[0]] = 0
            motif_norm_temp[nodes[1]] = 0
            
            motif_norm_temp[nodes[1]] = new_motif

    merge = np.array(merging_nodes)
#    merge = np.concatenate((merge),axis=1).T
    
    T = nx.Graph()
      
    T.add_node('Root')
    node_dict = {}
    
    if leaf_idx[-1] == 0:
        temp_node = 'h_'+str(merge[-1,1])+'_'+str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1,1]] = temp_node
        
    if leaf_idx[-1] == 1:
        T.add_edge(merge[-1,1], 'Root')
        
    if leaf_idx[-2] == 0:
        temp_node = 'h_'+str(merge[-1,0])+'_'+str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1,0]] = temp_node
        
    if leaf_idx[-2] == 1:
        T.add_edge(merge[-1,0], 'Root')
        
    idx = len(leaf_idx)-3
    
    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1)==0)
        reduction = len(temp) + 2
    else:
        reduction = 2
        
    for i in range(n_cluster-reduction)[::-1]:
        
        if leaf_idx[idx-1] == 1:
            if merge[i,1] in node_dict:
                T.add_edge(merge[i,0], node_dict[merge[i,1]])
            else:
                T.add_edge(merge[i,0], temp_node)
            
        if leaf_idx[idx] == 1:
            if merge[i,1] in node_dict:
                T.add_edge(merge[i,1], node_dict[merge[i,1]])
            else:
                T.add_edge(merge[i,1], temp_node)
            
        if leaf_idx[idx] == 0:
            new_node = 'h_'+str(merge[i,1])+'_'+str(i)
            if merge[i,1] in node_dict:
                T.add_edge(node_dict[merge[i,1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
#            node_dict[merge[i,1]] = new_node
            
            if leaf_idx[idx-1] == 1:
                temp_node = new_node
                node_dict[merge[i,1]] = new_node
            else:
                new_node_2 = 'h_'+str(merge[i,0])+'_'+str(i)
#                temp_node = 'h_'+str(merge[i,0])+'_'+str(i)
                T.add_edge(node_dict[merge[i,1]], new_node_2)
#                node_dict[merge[i,0]] = temp_node
                node_dict[merge[i,1]] = new_node
                node_dict[merge[i,0]] = new_node_2
#                temp_node = new_node
                
        elif leaf_idx[idx-1] == 0:
            new_node = 'h_'+str(merge[i,0])+'_'+str(i)
            if merge[i,1] in node_dict:
                T.add_edge(node_dict[merge[i,1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
            node_dict[merge[i,0]] = new_node
            
            if leaf_idx[idx] == 1:
                temp_node = new_node
            else:
                new_node = 'h_'+str(merge[i,1])+'_'+str(i)
                T.add_edge(temp_node, new_node)
                node_dict[merge[i,1]] = new_node
                temp_node = new_node
                
        idx -= 2
            
#        for i in range(n_cluster-1)[::-1]:
#            T.add_edge(merge[i,1],merge[i,0])

    return T


def draw_tree(T):
    # pos = nx.drawing.layout.fruchterman_reingold_layout(T)
    pos = hierarchy_pos(T,'Root',width=.5, vert_gap = 0.1, vert_loc = 0, xcenter = 50) 
    fig = plt.figure()
    nx.draw_networkx(T, pos)  
    

def traverse_tree(T, root_node=None):
    if root_node == None:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    traverse_preorder = '{'
    
    def _traverse_tree(T, node, traverse_preorder):
        traverse_preorder += str(node[0])
        traverse_list.append(node[0])
        children = list(T.neighbors(node[0]))
        
        if len(children) == 3:
    #        print(children)
            for child in children:
                if child in traverse_list:
#                    print(child)
                    children.remove(child)
            
        if len(children) > 1:
            traverse_preorder += '{'
            traverse_preorder_temp = _traverse_tree(T, [children[0]], '')
            traverse_preorder += traverse_preorder_temp
             
            traverse_preorder += '}{'
            
            traverse_preorder_temp = _traverse_tree(T, [children[1]], '')
            traverse_preorder += traverse_preorder_temp
            traverse_preorder += '}'
        
        return traverse_preorder
    
    traverse_preorder = _traverse_tree(T, node, traverse_preorder)
    traverse_preorder += '}'
    
    return traverse_preorder



def _traverse_tree(T, node, traverse_preorder,traverse_list):
    traverse_preorder += str(node[0])
    traverse_list.append(node[0])
    children = list(T.neighbors(node[0]))
    
    if len(children) == 3:
#        print(children)
        for child in children:
            if child in traverse_list:
#                    print(child)
                children.remove(child)
        
    if len(children) > 1:
        traverse_preorder += '{'
        traverse_preorder_temp = _traverse_tree(T, [children[0]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
         
        traverse_preorder += '}{'
        
        traverse_preorder_temp = _traverse_tree(T, [children[1]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
        traverse_preorder += '}'
    
    return traverse_preorder
    
def traverse_tree(T, root_node=None):
    if root_node == None:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    traverse_preorder = '{'
    traverse_preorder = _traverse_tree(T, node, traverse_preorder,traverse_list)
    traverse_preorder += '}'
    
    return traverse_preorder


def _traverse_tree_cutline(T, node, traverse_list, cutline, level, community_bag, community_list=None): 
    cmap = plt.get_cmap("tab10")
    traverse_list.append(node[0])
    if community_list is not None and type(node[0]) is not str:
        community_list.append(node[0])
    children = list(T.neighbors(node[0]))
    
    if len(children) == 3:
#        print(children)
        for child in children:
            if child in traverse_list:
#                    print(child)
                children.remove(child)
                    
    if len(children) > 1:
        if nx.shortest_path_length(T,'Root',node[0])==cutline:
            #create new list
            traverse_list1 = []
            traverse_list2 = []
            community_bag = _traverse_tree_cutline(T, [children[0]], traverse_list, cutline, level+1, community_bag, traverse_list1)
            community_bag = _traverse_tree_cutline(T, [children[1]], traverse_list, cutline, level+1, community_bag, traverse_list2)
            joined_list=traverse_list1+traverse_list2
            community_bag.append(joined_list)
            if type(node[0]) is not str: #append itself
                community_bag.append([node[0]])            
        else:
             community_bag = _traverse_tree_cutline(T, [children[0]], traverse_list, cutline, level+1, community_bag, community_list)
             community_bag = _traverse_tree_cutline(T, [children[1]], traverse_list, cutline, level+1, community_bag, community_list)            

    return  community_bag


def traverse_tree_cutline(T, root_node=None,cutline=2):
    if root_node == None:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    color_map = []
    community_bag=[]
    level = 0
    community_bag = _traverse_tree_cutline(T, node, traverse_list,cutline, level, color_map,community_bag)
    
    return community_bag



def TED(T1, T2, root_node_1=None, root_node_2=None):
#    print(root_node_1)
    order_T1 = traverse_tree(T1, root_node_1)
#    print(root_node_2)
    order_T2 = traverse_tree(T2, root_node_2)
    
    tree1=Tree.from_text(order_T1)
    tree2=Tree.from_text(order_T2)
        
    apted = APTED(tree1,tree2)
    return apted.compute_edit_distance()


def tree_edit_distance(tree_list, root_node_list=None):
    dim = len(tree_list)
    distance_matrix = np.zeros((dim, dim))
    
    if root_node_list == None:
        for i, tree1 in enumerate(tree_list):
            for j,tree2 in enumerate(tree_list):
                ted = TED(tree1,tree2)
                distance_matrix[i,j] = ted
                
    else:
        for i, tree1 in enumerate(tree_list):
            root_node_1 = root_node_list[i]
            for j,tree2 in enumerate(tree_list):
                root_node_2 = root_node_list[j]
                ted = TED(tree1,tree2, root_node_1,root_node_2)
                distance_matrix[i,j] = ted
    
    return distance_matrix 


def behavior_hierarchy(config, model_name, cluster_method, n_cluster):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to quantify your entire dataset? \n"
                     "If you only want to use a specific dataset type filename: \n"
                     "yes/no/filename ")
    else: 
        all_flag = 'yes'
        
    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)
            
    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + str(file) + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)
    
    trees = []
    for file in files:
        path_to_file=cfg['project_path']+'results/'+str(file)+'/'+model_name+'/'+cluster_method+'-'+str(n_cluster)
        transition_matrix =  np.load(path_to_file+'/behavior_quantification/transition_matrix.npy')
        T = graph_to_tree(path_to_file, transition_matrix, n_cluster, merge_sel=1)   
        trees.append(T)
        
    draw_tree(trees[0])
    
    # tree edit distance
    ted = tree_edit_distance(trees)
    
    
    
    
    
    