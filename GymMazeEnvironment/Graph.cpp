// Author: Felix Rutard
#include "Graph.hpp"

void Graph::add_node(uint64_t node){
    this->nodes.push_back(node);
}

void Graph::add_connection(std::array<uint64_t,3> connection){
    this->connections[connection[0]][connection[1]]=connection[2];
}

void Graph::set_nodes(std::vector<uint64_t> nodes){
    this->nodes=nodes;
}
void Graph::set_connections(std::unordered_map<uint64_t,std::unordered_map<uint64_t,uint64_t>> connections){
    this->connections=connections;
}

std::vector<uint64_t> Graph::get_nodes(){
    return this->nodes;
}

std::unordered_map<uint64_t,std::unordered_map<uint64_t,uint64_t>> Graph::get_connections(){
    return this->connections;
}

std::unordered_map<uint64_t,std::vector<uint64_t>> Graph::dijkstra_optimal_connections(uint64_t start_node, uint64_t end_node){
    std::unordered_map<uint64_t,std::vector<uint64_t>> optimal_connections;
    std::unordered_map<uint64_t,int64_t> dist;
    for (size_t i(0);i<this->nodes.size();++i){
        dist[this->nodes[i]]=-1;
    }
    std::vector<uint64_t> Q(this->nodes);
    dist[start_node]=0;
    std::unordered_map<uint64_t, std::vector<uint64_t> > predecessors;

    int64_t mini;
    std::vector<uint64_t>::iterator node_itr;
    uint64_t selected_node;
    std::unordered_map<uint64_t, uint64_t >::iterator neighbour_iterator;
    while (Q.size()!=0){
        mini=-1;
        for (node_itr=Q.begin(); node_itr!=Q.end();++node_itr){
            if (mini==-1 || (dist[*node_itr]>=0 && dist[*node_itr]<mini)){
                mini=dist[*node_itr];
                selected_node=*node_itr;
            }
        }
        Q.erase(std::remove(Q.begin(), Q.end(), selected_node), Q.end());
        if (this->connections.find(selected_node)!=this->connections.end()){
            for (neighbour_iterator = this->connections[selected_node].begin();neighbour_iterator!=this->connections[selected_node].end();++neighbour_iterator){
                if (dist[neighbour_iterator->first]<0 || dist[neighbour_iterator->first]>(dist[selected_node]+static_cast<int64_t>(this->connections[selected_node][neighbour_iterator->first]))){
                    dist[neighbour_iterator->first] = dist[selected_node] + static_cast<int64_t>(this->connections[selected_node][neighbour_iterator->first]);
                    predecessors[neighbour_iterator->first]={selected_node};
                }
                else if (dist[neighbour_iterator->first]==(dist[selected_node]+static_cast<int64_t>(this->connections[selected_node][neighbour_iterator->first]))){
                    predecessors[neighbour_iterator->first].push_back(selected_node);
                }
            }
        }
    }
    std::vector<uint64_t> nodes_to_explore = {end_node};
    std::vector<uint64_t> nodes_already_explored = {};

    while (nodes_to_explore.size()!=0){
        for (size_t i(0);i<predecessors[nodes_to_explore[0]].size();++i){
            if (std::find(optimal_connections[predecessors[nodes_to_explore[0]][i]].begin(), optimal_connections[predecessors[nodes_to_explore[0]][i]].end(), nodes_to_explore[0]) == optimal_connections[predecessors[nodes_to_explore[0]][i]].end()){
                optimal_connections[predecessors[nodes_to_explore[0]][i]].push_back(nodes_to_explore[0]);
            }
            if (predecessors[nodes_to_explore[0]][i]!=start_node){
                if (std::find(nodes_already_explored.begin(), nodes_already_explored.end(), predecessors[nodes_to_explore[0]][i]) == nodes_already_explored.end()){
                    nodes_to_explore.push_back(predecessors[nodes_to_explore[0]][i]);
                }
            }
        }
        nodes_already_explored.push_back(nodes_to_explore[0]);
        nodes_to_explore.erase(nodes_to_explore.begin());
    }
    return optimal_connections;
}

std::unordered_map<uint64_t,std::vector<uint64_t>> Graph::optimal_connections_full_graph(uint64_t end_node){
    std::unordered_map<uint64_t,std::vector<uint64_t>> optimal_connections_full_graph;
    std::unordered_map<uint64_t,std::vector<uint64_t>> optimal_connections;
    std::unordered_map<uint64_t,std::vector<uint64_t>>::iterator optimal_connections_iterator;
    for (size_t i(0) ; i < this->nodes.size() ; ++i){
        optimal_connections = dijkstra_optimal_connections(this->nodes[i],end_node);
        for (optimal_connections_iterator=optimal_connections.begin();optimal_connections_iterator!=optimal_connections.end();++optimal_connections_iterator){
            optimal_connections_full_graph[optimal_connections_iterator->first]=optimal_connections_iterator->second;
        }
    }
    return optimal_connections_full_graph;
}

uint64_t Graph::dijkstra_distance(uint64_t start_node, uint64_t end_node){
    std::unordered_map<uint64_t,int64_t> dist;
    for (size_t i(0);i<this->nodes.size();++i){
        dist[this->nodes[i]]=-1;
    }
    std::vector<uint64_t> Q(this->nodes);
    dist[start_node]=0;
    std::unordered_map<uint64_t, uint64_t> predecessors;

    int64_t mini;
    std::vector<uint64_t>::iterator node_itr;
    uint64_t selected_node;
    std::unordered_map<uint64_t, uint64_t >::iterator neighbour_iterator;
    while (Q.size()!=0){
        mini=-1;
        for (node_itr=Q.begin(); node_itr!=Q.end();++node_itr){
            if (mini==-1 || (dist[*node_itr]>=0 && dist[*node_itr]<mini)){
                mini=dist[*node_itr];
                selected_node=*node_itr;
            }
        }
        Q.erase(std::remove(Q.begin(), Q.end(), selected_node), Q.end());
        if (this->connections.find(selected_node)!=this->connections.end()){
            for (neighbour_iterator = this->connections[selected_node].begin();neighbour_iterator!=this->connections[selected_node].end();++neighbour_iterator){
                if (dist[neighbour_iterator->first]<0 || dist[neighbour_iterator->first]>(dist[selected_node]+static_cast<int64_t>(this->connections[selected_node][neighbour_iterator->first]))){
                  dist[neighbour_iterator->first] = dist[selected_node] + static_cast<int64_t>(this->connections[selected_node][neighbour_iterator->first]);
                  predecessors[neighbour_iterator->first]=selected_node;
                }
            }
        }
    }
    uint64_t current_node=end_node;
    uint64_t predecessor;
    uint64_t distance(0);
    while (current_node!=start_node){
        predecessor = predecessors[current_node];
        distance += this->connections[predecessor][current_node];
        current_node=predecessor;
    }
    return distance;
}

uint64_t Graph::dtw(std::vector<uint64_t> trajectory_1, std::vector<uint64_t> trajectory_2){
    std::vector<uint64_t> dtw;
    uint64_t t1 = trajectory_1.size();
    uint64_t t2 = trajectory_2.size();
    dtw.reserve(t1*t2);
    dtw[0]=0;
    for (size_t i(1);i<t1*t2;++i){
        dtw[i]=UINT64_MAX;
    }
    uint64_t cost;
    uint64_t idx_1;
    uint64_t idx_2;
    uint64_t dtw_idx;
    for (size_t i(0);i<(t1-1)*(t2-1);++i){
        idx_1 = i/(t2-1)+1;
        idx_2 = i+1-(idx_1-1)*(t2-1);
        dtw_idx= idx_1*t2+idx_2;
        cost = this->dijkstra_distance(trajectory_1[idx_1],trajectory_2[idx_2]);
        dtw[dtw_idx]=cost+std::min(dtw[(idx_1-1)*t2+idx_2],std::min(dtw[idx_1*t2+idx_2-1],dtw[(idx_1-1)*t2+idx_2-1]));
    }
    return dtw[t1*t2-1];
}

uint64_t Graph::frechet(std::vector<uint64_t> trajectory_1, std::vector<uint64_t> trajectory_2){
    std::vector<uint64_t> frechet_matrix;
    uint64_t t1 = trajectory_1.size();
    uint64_t t2 = trajectory_2.size();
    frechet_matrix.reserve(t1*t2);
    frechet_matrix[0]=0;
    for (size_t i(1);i<t1*t2;++i){
        frechet_matrix[i]=UINT64_MAX;
    }
    uint64_t cost;
    uint64_t idx_1;
    uint64_t idx_2;
    uint64_t frechet_idx;
    for (size_t i(0);i<(t1-1)*(t2-1);++i){
        idx_1 = i/(t2-1)+1;
        idx_2 = i+1-(idx_1-1)*(t2-1);
        frechet_idx= idx_1*t2+idx_2;
        cost = this->dijkstra_distance(trajectory_1[idx_1],trajectory_2[idx_2]);
        frechet_matrix[frechet_idx]=std::max(cost,std::min(frechet_matrix[(idx_1-1)*t2+idx_2],std::min(frechet_matrix[idx_1*t2+idx_2-1],frechet_matrix[(idx_1-1)*t2+idx_2-1])));
    }
    return frechet_matrix[t1*t2-1];
}
