# Author: Felix Rutard

####### README #######
# install pybind11 : "sudo apt install python3-pybind11"
# check your current python3 version : "python3 -V" -> "Python 3.x.y"
# modify 'x' regarding your python3 version in the following line and
# compile c++ code : "g++ -O3 -Wall -shared -std=c++11 -I/usr/include/python3.x -lpython3.x -fPIC $(python3 -m pybind11 --includes) Graph.cpp -o Graph$(python3-config --extension-suffix)"

# demo of the package using all possible graph's class methods:

import Graph

graph = Graph.Graph()

nodes = [0,1,2,3,4]
graph.set_nodes(nodes)
graph.add_node(5)

connections = {0: {0: 0,1: 10, 4: 20}, 1: {1: 0, 0: 10, 2: 20}, 2: {2: 0, 1: 20, 3: 20}, 3: {3: 0, 2: 20, 5: 20}, 4: {4:0, 0: 20, 5: 10}, 5: {5: 0, 4: 10}}
graph.set_connections(connections)
graph.add_connection([5,3,20])

print("nodes : ",graph.get_nodes())
print("nconnections : ",graph.get_connections())

start_node = 0
end_node = 3

optimal_connections = graph.dijkstra_optimal_connections(start_node,end_node)
print(optimal_connections) # optimal sub-graph connections

print("distance: ", graph.dijkstra_distance(0,3)) # djikstra distance (integer)

dtw = graph.dtw([0,1,2,3],[0,4,5,3]) # dynamic time warping (integer)
print("dtw distance : ",dtw)

frechet = graph.frechet([0,1,2,3],[0,4,5,3]) # frechet distance (integer)
print("frechet distance : ",frechet)
