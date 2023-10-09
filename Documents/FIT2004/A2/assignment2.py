"""
assignment2.py
Name: Wong Chee Hao
Student ID: 32734751
E-mail: cwon0112@student.monash.edu
"""
import math, heapq

class RoadGraph:
	
	def __init__(self, roads, cafes):
		"""
	    This function is the constructor of RoadGraph, represented as adjancency list.

	    Input:
	        roads: A list of tuples (u, v, w):
	        	   u is the starting location ID for a road, represented as a non-negative integer
				   v is the ending location ID for a road, represented as a non-negative integer
				   w is the time taken to travel from location u to location v, represented as a non-negative integer.

	        cafes: A list of tuples (location, waiting_time):
	        	   location is the location of the cafe; represented as a non-negative integer.
				   waiting_time is the waiting time for a coffee in the cafe, represented as a non-negative integer.
	    Return:
	        -

		|E| = The number of roads
		|V| = the number of locations

	    Time complexity: O(|V| + |E|)
	    Space complexity: O(|V| + |E|)
	    """

	    # count the set of unique vertices(locations) in roads
		vertices_count = 0
		for i in range(len(roads)):
			for j in range(2):
				if roads[i][j] > vertices_count:
					vertices_count = roads[i][j]
		vertices_count = vertices_count + 1

		# create array
		self.vertices = [None] * vertices_count

		# Add vertex to array, the vertex IDs are continuous from 0 to |V| − 1.
		for i in range(vertices_count):
			self.vertices[i] = Vertex(i)

		# Add edges to each vertex
		for edge in roads:
			u = edge[0]
			v = edge[1]
			w = edge[2]
			current_edge = Edge(u,v,w)
			current_vertex = self.vertices[u]
			current_vertex.add_edge(current_edge)

		# Label cafes
		for cafe in cafes:
			self.vertices[cafe[0]].is_cafe = True
			self.vertices[cafe[0]].waiting_time = cafe[1]

	def edge_relaxation(self, u, v, w):
		"""
	    This function update the route if the distance from u to v is shorter than the distance from s to v, 

	    Input:
	        u: u is the starting vertex ID
	        v: is the ending vertex ID
	        w: is the weight from u to v
	    Return:
	        -

	    Time complexity: O(1)
	    Space complexity: O(1)

	    """

		if self.vertices[v].distance > self.vertices[u].distance + w:
			self.vertices[v].distance = self.vertices[u].distance + w
			self.vertices[v].previous = self.vertices[u]

		return

	def reset_vertices(self):
		"""
	    This function reset the attributes of each vertex in Graph.

	    Input:
			-
	    Return:
	        -
	    Time complexity: O(1)
	    Space complexity: O(1)
	    """
		for vertex in self.vertices:
			vertex.distance = math.inf
			vertex.previous = None
			vertex.discovered = False
			vertex.finalized = False

		return

	def reverse_graph(self):
		"""
	    This function reverses the direction of each edges in graph.

	    Input:
			-
	    Return:
	        -

		|E| = The number of roads
		|V| = the number of locations

	    Time complexity: O(|V| + |E|)
	    Space complexity: O(|V| + |E|)

	    """
	    # temp array to store the edges
		reverse_edges  = [[] for i in range(len(self.vertices))]

		# traverse all edges of each vertices and store it to the temp array
		for vertex in self.vertices:
			for edge in vertex.edges:
				reverse_edges[edge.v].append(Edge(edge.v,edge.u,edge.w))
		
		# replace the edges of original graph with the reversed edges
		for i in range(len(self.vertices)):
			self.vertices[i].edges = reverse_edges[i]

		return


	def dijkstra(self, source):
		"""
	    This function finding the shortest paths from source node to every other nodes in graph

	    Input:
	        source: the starting node of the path
	    Return:
	        -

		|E| = The number of roads
		|V| = the number of locations

	    Time complexity: O(|E| log |V|)
	    Space complexity: O(|V| + |E|)

	    """
		self.reset_vertices()										# reset the vertices
		discovered = []
		heapq.heapify(discovered)									# discovered is a priority queue

		if source in range(len(self.vertices)):
			self.vertices[source].distance = 0						# set the distance from source vertex to itself to 0 
			heapq.heappush(discovered, source)						# enqueue the source vertex

			while len(discovered) > 0:								# if queue is not empty
				u = heapq.heappop(discovered) 						# dequeue the min vertex in queue
				self.vertices[u].finalized = True					# the distance of u is finalized

				for edge in self.vertices[u].edges:					# handle all the adjacent vertices of u
					v = edge.v
					if self.vertices[v].finalized == False:			# if distance is not finalized
						self.edge_relaxation(u, v, edge.w)			# perform edge relaxation
						if self.vertices[v].discovered == False:	# if distance is still inf
							self.vertices[v].discovered = True			
							heapq.heappush(discovered, v)			# means I have discovered v, enqueue it

		return

	def routing(self, start, end):
		"""
	    The function return the shortest route from the start location to the end location, going through at least 1 of the locations listed in cafes.
	    
	    Implementation:
		    1. We run 1 dijkstra from start location to every cafes, and run 1 dijkstra from end location to every cafes, then combines the distance and path we calculated in both dijkstra.
		    2. Choose the shortest path from all possible paths.

	    Input:
	        start: a non-negative integer that represents the starting location of the journey
	        end: non-negative integer that represents the ending location of the journey
	        
	    Return:
	         The shortest route from the start location to the end location, going through at least 1 of the locations listed in cafes.
	         If multiple shortest routes, return any one them.
	         If no possible route, return None.
		
		|E| = The number of roads
		|V| = the number of locations

	    Time complexity: O(|E| log |V|) 
	    Space complexity: O(|V| + |E|)

	    """
		shortest_route = None										# Initialize the return route to None
		shortest_time = [0] * len(self.vertices)					# list to store shortest time of each routes from starting location to location i 
																	# if shortest_time[i] == 0 means the location i is not a cafe
																	# if shortest_time[i] == inf means there is no reachable route from starting location to the location

		all_paths = [[] for i in range(len(self.vertices))]			# list to store all possible paths

		self.dijkstra(start)										# Run dijkstra from start to every cafe
		
		for i in range(len(self.vertices)):							
			if self.vertices[i].is_cafe == True:
				shortest_time[i] += self.vertices[i].distance + self.vertices[i].waiting_time	# Add shortest distance from start to cafe and waiting time at cafe to list
				vertex = self.vertices[i]														
				all_paths[i] = all_paths[i] + [vertex.id]										# Add cafe location to the shortest path(last location)
				while vertex.id != start:														# While we have not added start location to path, means the route is incomplete
					if vertex.previous != None:													# If the location is reachable
						all_paths[i] = [vertex.previous.id] + all_paths[i]						# Add the previous location to the route
						vertex = vertex.previous	
					else:
						break

		self.reverse_graph()										# Reverse the graph
		self.dijkstra(end)											# Run dijkstra from end to every cafe; If there is a route from end to cafe, means there is a route from cafe to end(with the same distance)

		for i in range(len(self.vertices)):
			if self.vertices[i].is_cafe == True:
				shortest_time[i] += self.vertices[i].distance									# Add shortest distance from end to cafe 
				vertex = self.vertices[i]														
				while vertex.id != end:															# While we have not added end location to path, means the route is incomplete
					if vertex.previous != None:													# If the location is reachable
						all_paths[i] = all_paths[i] + [vertex.previous.id] 						# Add the previous location to the route
						vertex = vertex.previous
					else:
						break

		min_index = None
		min_time = math.inf

		# Identify the shortest route
		for i in range(len(shortest_time)):
			if shortest_time[i] > 0 and shortest_time[i] < min_time:	# If there is a possible route 
				min_time = shortest_time[i]								
				min_index = i
		if min_index != None:											# If there is a possible route 
			shortest_route = all_paths[min_index]						# store the shortest route

		return shortest_route


class Vertex:
	def __init__(self, id):
		"""
	    This function is the constructor of Vertex

	    Input:
	        id: The label of vertex.
	    Return:
	        -
	    Time complexity: O(1)
	    Space complexity: O(1)
	    """
		self.id = id 				# the label of vertex
		self.edges = []				# edges from the vertex to other vertex

		self.distance = math.inf	# the optimal distance of route from a particular starting vertex 
		self.previous = None		# the previous vertex of the optimal route
		self.discovered = False		# True if the vertex is discovered
		self.finalized = False		# True if the distance of the shortest route from a particular starting vertex is finalized. 

		self.is_cafe = False		# True if the location has a cafe
		self.waiting_time = 0		# The waiting time for a coffee in the cafe, represented as a non-negative integer

	def add_edge(self,edge):
		"""
	    This function add edge to vertex

	    Input:
	        edge: A tuple (u, v, w)
	        	  u is the starting vertex ID
				  v is the ending vertex ID
				  w is the weight from u to v
	    Return:
	        -
	    Time complexity: O(1)
	    Space complexity: O(1)
	    """
		self.edges.append(edge)

class Edge:
	def __init__(self, u, v, w):
		"""
	    This function is the constructor of Edge

	    Input:
	        u: u is the starting vertex ID
	        v: is the ending vertex ID
	        w: is is the weight from u to v
	    Return:
	        -
	    Time complexity: O(1)
	    Space complexity: O(1)
	    """

		self.u = u
		self.v = v
		self.w = w 

class downhillGraph:
	
	def __init__(self, downhillScores):
		"""
	    This function is the constructor of downhillGraph, represented as adjancency list.

	    Input:
	        downhillScores: A list of tuples (a, b, c):
	        	a is the start point of a downhill segment, a ∈ {0, 1, . . . , |P| − 1}
				b is the end point of a downhill segment, b ∈ {0, 1, . . . , |P| − 1}
				c is the integer score that you would get for using this downhill segment to go from point a to point b

	    Return:
	        -

		|P| = The number of intersection points
		|D| = the number of downhill segments

	    Time complexity: O(|P| + |D|)
	    Space complexity: O(|P| + |D|)
	    """

	    # count the set of unique vertices(intersection point)
		vertices_count = 0
		for i in range(len(downhillScores)):
			for j in range(2):
				if downhillScores[i][j] > vertices_count:
					vertices_count = downhillScores[i][j]
		vertices_count = vertices_count + 1

		# create array
		self.vertices = [None] * vertices_count

		# Add vertex to array, the vertex IDs are continuous from 0 to |P| − 1.
		for i in range(vertices_count):
			self.vertices[i] = Vertex(i)

		# Add edges to each vertex
		for edge in downhillScores:
			u = edge[0]
			v = edge[1]
			w = edge[2]
			current_edge = Edge(u,v,w)
			current_vertex = self.vertices[u]
			current_vertex.add_edge(current_edge)

	def reverse_graph(self):
		"""
	    This function reverses the direction of each edges in graph.

	    Input:
			-
	    Return:
	        -

		|P| = The number of intersection points
		|D| = the number of downhill segments

	    Time complexity: O(|P| + |D|)
	    Space complexity: O(|P| + |D|)

	    """
	    # temp array to store the edges
		reverse_edges  = [[] for i in range(len(self.vertices))]

		# traverse all edges of each vertice and store it to the temp array
		for vertex in self.vertices:
			for edge in vertex.edges:
				reverse_edges[edge.v].append(Edge(edge.v,edge.u,edge.w))
		
		# replace the edges of original graph with the reversed edges
		for i in range(len(self.vertices)):
			self.vertices[i].edges = reverse_edges[i]

	def kahn(self):
		"""
	    This function do Topological sorting for Directed Acyclic Graph (DAG)

	    Input:
			-
	    Return:
	        sorted_list:  list of linear ordering of vertices such that for every directed edge uv, vertex u comes before v in the ordering. 

		|P| = The number of intersection points
		|D| = the number of downhill segments

	    Time complexity: O(|P| + |D|)
	    Space complexity: O(|P| + |D|)

	    """
		sorted_list = []

		# count incoming edges
		incoming_edges = [0] * len(self.vertices)
		for vertex in self.vertices:
			for edge in vertex.edges:
				incoming_edges[edge.v] = incoming_edges[edge.v] + 1
	
		process = []	# process is a stack

		for i in range(len(incoming_edges)):
			if incoming_edges[i] == 0:
				process.append(i)

		# kahn's
		while len(process) > 0:
			vertex_u = process.pop()
			sorted_list.append(vertex_u)
			for edge in self.vertices[vertex_u].edges:
				incoming_edges[edge.v] -= 1
				if incoming_edges[edge.v] == 0:
					process.append(edge.v)

		return sorted_list

def optimalRoute(downhillScores, start, finish):
	"""
    This function output the route from the starting point start to the finishing point while using only downhill segments and obtaining the maximum score.

    Input:
        downhillScores: A list of tuples (a, b, c):
	    	a is the start point of a downhill segment, a ∈ {0, 1, . . . , |P| − 1}
			b is the end point of a downhill segment, b ∈ {0, 1, . . . , |P| − 1}
			c is the integer score that you would get for using this downhill segment to go from point a to point b
		start: starting point of the tournament
		finish: finishing point of the tournament
    Return:
        The route from the starting point start to the finishing point while using only downhill segments and obtaining the maximum score.

	|P| = The number of intersection points
	|D| = the number of downhill segments

    Time complexity:  O(|P| + |D|) = O(|D|), because |P| <= |D|
    Space complexity: O(|P| + |D|)

    """

	# create a graph representation for tournament
	graph = downhillGraph(downhillScores)

	# reverse the graph so we can know the incoming edges of a vertex(downhill segments link to a certain point).
	graph.reverse_graph()

	# Topological sort using kahn algorithm 
	# so we can first solve the sub-problems, then use the optimal solutions of subproblems to solve the bigger problems. 
	kahn_sorted_list = graph.kahn()

	return optimalRoute_aux(graph, kahn_sorted_list, start, finish)

def optimalRoute_aux(graph, kahn_sorted_list, start, finish):
	"""
    This function use Dynamic Programming output the route from the starting point start to the finishing point while using only downhill segments and obtaining the maximum score.

    Input:
    	graph: graph representation for tournament(which is reversed)
		kahn_sorted_list: list of linear ordering of vertices such that for every directed edge uv, vertex u comes before v in the ordering. 
		start: starting point of the tournament
		finish: finishing point of the tournament
    Return:
        The route from the starting point start to the finishing point while using only downhill segments and obtaining the maximum score.

	|P| = The number of intersection points
	|D| = the number of downhill segments

    Time complexity: O(|P| + |D|) = O(|D|)
    Space complexity: O(|P| + |D|)

    """
	memo = [-math.inf for i in range(len(kahn_sorted_list))]		# initialize memo[i] to -inf
																	# memo[i] = max score obtained from starting point [i] to finishing point [finish] 
	memo[finish] = 0												# max score obtained from finish point to finish point is 0
																	# base case = starting point and finishing point are the same point. 

	optimal_solution = [None for i in range(len(kahn_sorted_list))]	# store the optimal solution for subproblem(only store the next intersection point)
	return_route = []

	# find optimal solutions for the subproblems
	for i in kahn_sorted_list:										
		for edge in graph.vertices[i].edges:						# go through each downhill segment that link to graph.vertices[i]
			if edge.w + memo[edge.u] > memo[edge.v]:				# if we have new optimal 
				memo[edge.v] = edge.w + memo[edge.u]				# update optimal score
				optimal_solution[edge.v] = edge.u 					# update optimal solution

	while start != finish:											# while the route is incomplete		
		return_route = return_route + [start]						# add the starting point to the first position of route
		start = optimal_solution[start]								# solution reconstruction using memoisation
		if start == None:											# if there is no possible route from starting point start to the finishing point finish.
			return None												

	return_route = return_route + [finish]							# add the finishing point to the last position of route

	return return_route

