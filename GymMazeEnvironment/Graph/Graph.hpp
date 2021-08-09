// Author: Felix Rutard 
#include <vector>
#include <unordered_map>
#include <iostream>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Graph{
    public:
        void add_node(uint64_t node);
        void add_connection(std::array<uint64_t,3> connection_array);
        std::vector<uint64_t> get_nodes();
        std::unordered_map<uint64_t,std::unordered_map<uint64_t,uint64_t>> get_connections();
        void set_nodes(std::vector<uint64_t> nodes);
        void set_connections(std::unordered_map<uint64_t,std::unordered_map<uint64_t,uint64_t>> connections);
        std::unordered_map<uint64_t,std::vector<uint64_t > > dijkstra_optimal_connections(uint64_t start_node, uint64_t end_node);
        std::unordered_map<uint64_t,std::vector<uint64_t > > optimal_connections_full_graph(uint64_t end_node);
        uint64_t dijkstra_distance(uint64_t start_node, uint64_t end_node);
        uint64_t dtw(std::vector<uint64_t> trajectory_1,std::vector<uint64_t> trajectory_2);
        uint64_t frechet(std::vector<uint64_t> trajectory_1,std::vector<uint64_t> trajectory_2);
    private:
        std::vector<uint64_t> nodes;
        std::unordered_map<uint64_t,std::unordered_map<uint64_t,uint64_t>> connections;
};

PYBIND11_MODULE(Graph, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node", &Graph::add_node)
        .def("add_connection", &Graph::add_connection)
        .def("get_nodes", &Graph::get_nodes)
        .def("get_connections", &Graph::get_connections)
        .def("set_nodes", &Graph::set_nodes)
        .def("set_connections", &Graph::set_connections)
        .def("optimal_connections_full_graph", & Graph::optimal_connections_full_graph)
        .def("dijkstra_optimal_connections", &Graph::dijkstra_optimal_connections)
        .def("dijkstra_distance", &Graph::dijkstra_distance)
        .def("dtw", &Graph::dtw)
        .def("frechet", &Graph::frechet);
}
