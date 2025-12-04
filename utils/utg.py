import os
import glob
import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import xml.etree.ElementTree as ET
from typing import Set, List, Tuple, Optional, Dict


class utg:
    def __init__(self, xml_file, component_info_file=None, manifest_file=None):
        """
        Initialize the UTG graph by reading a CTG.xml file.
        
        Args:
            xml_file: Path to the CTG.xml file
            component_info_file: Path to the componentInfo.xml file
            manifest_file: Path to the AndroidManifest.txt file
        """
        self.xml_file = xml_file
        self.component_info_file = component_info_file
        self.manifest_file = manifest_file
        self.graph = nx.DiGraph()
        self.main_activity = None
        self.pure_utg = None
        self._parse_xml()
        
        # Initialize dynamic tracking attributes
        self.current_node = None
        self.targets = {}  # {node_name: visited_status}
        self.visited_nodes = set()  # Track all visited nodes
        self.dynamically_added_edges = []  # Track edges added during exploration
        
        # Prefer component info file if available
        if component_info_file:
            self._parse_component_info()
        elif manifest_file:
            self._parse_manifest()
        
        # Set initial current node to main activity
        # if self.main_activity and self.main_activity in self.graph:
        #     self.current_node = self.main_activity
        #     self.visited_nodes.add(self.main_activity)
        
        # Calculate longest shortest paths if main activity is found
        self.longest_shortest_paths = {}
        if self.main_activity and self.main_activity in self.graph:
            self._calculate_longest_shortest_path()
    
    def set_current_node(self, node_name: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Set the current node and return information about it.
        
        Args:
            node_name: Name of the node to set as current
            
        Returns:
            Tuple of (node_name, node_attributes) if successful, (None, None) otherwise
        """
        if node_name in self.graph:
            self.current_node = node_name
            self.visited_nodes.add(node_name)
            
            # Update target status if this is a target
            if node_name in self.targets:
                self.targets[node_name] = True
            
            return node_name, dict(self.graph.nodes[node_name])
        else:
            print(f"Node {node_name} not found in graph")
            return None, None
    
    def set_targets(self, target_nodes: List[str]) -> Dict[str, bool]:
        """
        Set the target nodes to visit.
        
        Args:
            target_nodes: List of target node names
            
        Returns:
            Dictionary of targets with their visited status
        """
        self.targets = {}
        for target in target_nodes:
            if target in self.graph:
                # Check if already visited
                visited = target in self.visited_nodes
                self.targets[target] = visited
            else:
                print(f"Warning: Target node {target} not found in graph")
        
        return self.targets
    
    def get_unvisited_targets(self) -> Set[str]:
        """Return set of unvisited target nodes."""
        return {node for node, visited in self.targets.items() if not visited}
    

    def dynamic_update_graph(self, new_node_name: str, node_type: str = "Activity") -> bool:
        """
        Dynamically update the graph with a new node or edge.
        
        Args:
            new_node_name: Name of the new node discovered
            node_type: Type of the node (default: "Activity")
            
        Returns:
            True if graph was updated, False otherwise
        """
        if not self.current_node:
            print("No current node set, cannot add edge")
            return False
        
        # Check if this is an existing node
        if new_node_name in self.graph:
            # Node exists, check if edge exists
            if not self.graph.has_edge(self.current_node, new_node_name):
                # Add new edge
                edge_attrs = {
                    'edgeType': 'DynamicallyDiscovered',
                    'ICCType': 'runtime',
                    'discovered_at': len(self.dynamically_added_edges) + 1
                }
                self.graph.add_edge(self.current_node, new_node_name, **edge_attrs)
                self.dynamically_added_edges.append((self.current_node, new_node_name))
                print(f"Added new edge: {self.current_node} -> {new_node_name}")
                
                # Update pure UTG if it exists
                if self.pure_utg and (self.current_node in self.pure_utg or new_node_name in self.pure_utg):
                    self.pure_utg.add_edge(self.current_node, new_node_name, **edge_attrs)
                
                return True
            else:
                print(f"Edge {self.current_node} -> {new_node_name} already exists")
                return False
        else:
            # New node, add it to the graph
            self.graph.add_node(new_node_name, type=node_type, dynamically_added=True)
            
            # Add edge from current node
            edge_attrs = {
                'edgeType': 'DynamicallyDiscovered',
                'ICCType': 'runtime',
                'discovered_at': len(self.dynamically_added_edges) + 1
            }
            self.graph.add_edge(self.current_node, new_node_name, **edge_attrs)
            self.dynamically_added_edges.append((self.current_node, new_node_name))
            
            print(f"Added new node {new_node_name} and edge: {self.current_node} -> {new_node_name}")
            
            # Update pure UTG if applicable
            if self.pure_utg and node_type == "Activity":
                self.pure_utg.add_node(new_node_name, type=node_type, dynamically_added=True)
                self.pure_utg.add_edge(self.current_node, new_node_name, **edge_attrs)
            
            return True
    
    def get_exploration_stats(self) -> Dict:
        """Get statistics about the exploration progress."""
        stats = {
            "current_node": self.current_node,
            "visited_nodes": len(self.visited_nodes),
            "total_nodes": self.graph.number_of_nodes(),
            "coverage": len(self.visited_nodes) / max(1, self.graph.number_of_nodes()),
            "targets_total": len(self.targets),
            "targets_visited": sum(1 for visited in self.targets.values() if visited),
            "targets_remaining": len(self.get_unvisited_targets()),
            "dynamically_added_edges": len(self.dynamically_added_edges),
            "visited_nodes_list": list(self.visited_nodes)
        }
        
        # Add reachability information if available
        if self.current_node:
            reachable = set()
            for node in self.graph.nodes():
                if node != self.current_node:
                    try:
                        nx.shortest_path(self.graph, self.current_node, node)
                        reachable.add(node)
                    except nx.NetworkXNoPath:
                        pass
            
            stats["reachable_from_current"] = len(reachable)
            stats["reachable_unvisited"] = len(reachable - self.visited_nodes)
            stats["reachable_targets"] = len(reachable & set(self.targets.keys()))
        
        return stats
    
    def reset_exploration(self):
        """Reset exploration state while keeping the graph structure."""
        self.current_node = self.main_activity if self.main_activity else None
        self.visited_nodes = {self.current_node} if self.current_node else set()
        self.targets = {}
        # Note: We keep dynamically_added_edges to preserve the exploration history
    
    def get_path_to_target(self, target: str) -> Tuple[List[str], float]:
        """
        Get shortest path from current node to a target.
        
        Args:
            target: Target node name
            
        Returns:
            Tuple of (path, cost) or ([], inf) if no path exists
        """
        if not self.current_node or target not in self.graph:
            return [], float('inf')
        
        try:
            path = nx.shortest_path(self.graph, self.current_node, target, weight='weight')
            cost = nx.shortest_path_length(self.graph, self.current_node, target, weight='weight')
            return path, cost
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    def suggest_exploration_target(self) -> Optional[str]:
        """
        Suggest a node to explore based on various heuristics.
        
        Returns:
            Suggested node name or None
        """
        if not self.current_node:
            return None
        
        # Priority 1: Unvisited targets reachable from current
        unvisited_targets = self.get_unvisited_targets()
        for target in unvisited_targets:
            path, _ = self.get_path_to_target(target)
            if path:
                return target
        
        # Priority 2: Unvisited nodes with high connectivity
        unvisited = set(self.graph.nodes()) - self.visited_nodes
        if unvisited:
            # Sort by in-degree + out-degree
            node_scores = []
            for node in unvisited:
                try:
                    path_len = nx.shortest_path_length(self.graph, self.current_node, node)
                    connectivity = self.graph.in_degree(node) + self.graph.out_degree(node)
                    # Prefer closer nodes with high connectivity
                    score = connectivity / (path_len + 1)
                    node_scores.append((score, node))
                except nx.NetworkXNoPath:
                    pass
            
            if node_scores:
                node_scores.sort(reverse=True)
                return node_scores[0][1]
        
        return None
    
    # Keep all existing methods from the original class
    def _set_targets(self, targets):
        """Legacy method - redirects to set_targets."""
        return self.set_targets(targets)
    
    def _parse_component_info(self):
        try:
            # Parse the XML file
            tree = ET.parse(self.component_info_file)
            root = tree.getroot()
            
            # Find all component nodes
            components = root.findall('.//component')
            
            main_activity = None
            
            # Check each component
            for component in components:
                # Check if it's an activity
                if component.get('type') == 'Activity':
                    # Get component name
                    component_name = component.get('name')
                    
                    # Check for intent_filter with MAIN and LAUNCHER
                    manifest = component.find('./manifest')
                    if manifest is not None:
                        intent_filters = manifest.findall('./intent_filter')
                        
                        for intent_filter in intent_filters:
                            action = intent_filter.get('action', '')
                            category = intent_filter.get('category', '')
                            
                            # Check if this intent filter has MAIN action and LAUNCHER category
                            if 'android.intent.action.MAIN' in action and 'android.intent.category.LAUNCHER' in category:
                                main_activity = component_name
                                break
                    
                    if main_activity:
                        break
            
            self.main_activity = main_activity
            
            # If the main activity is in the graph, we're good
            if self.main_activity and self.main_activity in self.graph:
                print(f"Found main activity in graph: {self.main_activity}")
            else:
                print(f"Main activity {self.main_activity} not found in graph nodes")
                
        except Exception as e:
            print(f"Error parsing component info file: {e}")
            self.main_activity = None
    
    def _create_pure_utg(self):
        """
        Create a pure UTG which is a subgraph consisting of edges 
        that either start from an Activity node or end at an Activity node.
        """
        # Create a new graph for the pure UTG
        self.pure_utg = nx.DiGraph()
        
        # Find all nodes of type Activity
        activity_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                         if attrs.get('type') == 'Activity']
        
        # Add all Activity nodes to the pure UTG
        for node in activity_nodes:
            node_attrs = self.graph.nodes[node]
            self.pure_utg.add_node(node, **node_attrs)
        
        # Add edges where either the source or target is an Activity
        for u, v, edge_attrs in self.graph.edges(data=True):
            if u in activity_nodes or v in activity_nodes:
                # Only add the nodes if they don't exist already
                if not self.pure_utg.has_node(u):
                    self.pure_utg.add_node(u, **self.graph.nodes[u])
                if not self.pure_utg.has_node(v):
                    self.pure_utg.add_node(v, **self.graph.nodes[v])
                
                # Add the edge with all its attributes
                self.pure_utg.add_edge(u, v, **edge_attrs)
        
        print(f"Created pure UTG with {self.pure_utg.number_of_nodes()} nodes and {self.pure_utg.number_of_edges()} edges")
        self.graph = self.pure_utg
    
    def _calculate_longest_shortest_path(self):
        """Calculate the longest shortest path from main activity to all reachable nodes."""
        if not self.main_activity or self.main_activity not in self.graph:
            print(f"Cannot calculate path metrics: main activity {self.main_activity} not in graph")
            self.longest_shortest_paths = {"max_length": 0, "paths": {}}
            self.reverse_longest_shortest_paths = {"max_length": 0, "paths": {}}
            return
        
        # Get all shortest paths from main activity to all reachable nodes
        try:
            # Calculate shortest paths from main activity to all other nodes
            shortest_paths = {}
            for target in self.graph.nodes():
                if target != self.main_activity:
                    try:
                        path_length = nx.shortest_path_length(self.graph, self.main_activity, target)
                        shortest_paths[target] = path_length
                    except nx.NetworkXNoPath:
                        # No path to this node from main activity
                        pass
            
            # Find the maximum shortest path length
            max_length = max(shortest_paths.values()) if shortest_paths else 0
            
            self.longest_shortest_paths = {
                "max_length": max_length,
                "paths": shortest_paths,
                "reachable_nodes": len(shortest_paths)
            }
            
            # Also calculate paths to main activity
            reverse_shortest_paths = {}
            for node in self.graph.nodes():
                if node != self.main_activity:
                    try:
                        path_length = nx.shortest_path_length(self.graph, node, self.main_activity)
                        reverse_shortest_paths[node] = path_length
                    except nx.NetworkXNoPath:
                        # No path from this node to main activity
                        pass
            
            max_reverse_length = max(reverse_shortest_paths.values()) if reverse_shortest_paths else 0
            
            self.reverse_longest_shortest_paths = {
                "max_length": max_reverse_length,
                "paths": reverse_shortest_paths,
                "nodes_that_can_reach_main": len(reverse_shortest_paths)
            }
            
            print(f"Calculated path metrics for {self.main_activity}:")
            print(f"  - Longest shortest path from main: {max_length}")
            print(f"  - Reachable nodes from main: {len(shortest_paths)}")
            print(f"  - Longest shortest path to main: {max_reverse_length}")
            print(f"  - Nodes that can reach main: {len(reverse_shortest_paths)}")
            
        except Exception as e:
            print(f"Error calculating paths: {e}")
            self.longest_shortest_paths = {"max_length": 0, "paths": {}}
            self.reverse_longest_shortest_paths = {"max_length": 0, "paths": {}}
    
    def _parse_xml(self):
        """Parse the XML file and build the graph."""
        # Parse the XML file
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Extract all source nodes
        sources = root.findall('.//source')
        
        # Add all sources as nodes first (including standalone sources)
        for source in sources:
            source_name = source.get('name')
            source_type = source.get('type')
            self.graph.add_node(source_name, type=source_type)
        
        # Now add all edges
        for source in sources:
            source_name = source.get('name')
            destinations = source.findall('.//destination')
            
            for dest in destinations:
                dest_name = dest.get('name')
                
                # Extract other attributes as edge attributes
                edge_attributes = {
                    'ICCType': dest.get('ICCType', ''),
                    'desType': dest.get('desType', ''),
                    'edgeType': dest.get('edgeType', ''),
                    'method': dest.get('method', ''),
                    'instructionId': dest.get('instructionId', ''),
                    'unit': dest.get('unit', ''),
                    'extras': dest.get('extras', ''),
                    'flags': dest.get('flags', ''),
                    'action': dest.get('action', ''),
                    'type': dest.get('type', '')
                }
                
                # Add the edge if it doesn't exist already
                if not self.graph.has_edge(source_name, dest_name):
                    self.graph.add_edge(source_name, dest_name, **edge_attributes)
        self._create_pure_utg()
    
    def to_dot(self, output_file="utg_graph.dot", subgraph=None, highlight_visited=True):
        """
        Export the graph to a DOT file for visualization with Graphviz.
        
        Args:
            output_file: Path to the output DOT file
            subgraph: If provided, export this subgraph instead of the main graph
            highlight_visited: If True, highlight visited nodes and current node
        """
        # Determine which graph to use
        graph = subgraph if subgraph is not None else self.graph
        
        # Configure node attributes
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            
            # Add visual attributes based on node type
            if node_type == "Activity":
                graph.nodes[node]['color'] = 'blue'
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = 'lightblue'
            elif node_type == "Service":
                graph.nodes[node]['color'] = 'green'
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = 'lightgreen'
            elif node_type == "Receiver":
                graph.nodes[node]['color'] = 'red'
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = 'lightpink'
            elif node_type == "NotComponentSource":
                graph.nodes[node]['color'] = 'gray'
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = 'lightgray'
            
            # Highlight visited nodes
            if highlight_visited and node in self.visited_nodes:
                graph.nodes[node]['style'] = 'filled,bold'
                graph.nodes[node]['penwidth'] = '2'
            
            # Highlight target nodes
            if node in self.targets:
                if self.targets[node]:  # Visited target
                    graph.nodes[node]['shape'] = 'doublecircle'
                    graph.nodes[node]['fillcolor'] = 'green'
                else:  # Unvisited target
                    graph.nodes[node]['shape'] = 'doublecircle'
                    graph.nodes[node]['fillcolor'] = 'yellow'
        
        # Highlight current node
        if self.current_node and self.current_node in graph:
            graph.nodes[self.current_node]['color'] = 'red'
            graph.nodes[self.current_node]['penwidth'] = '3'
            graph.nodes[self.current_node]['style'] = 'filled,bold'
        
        # Highlight the main activity if it exists
        if hasattr(self, 'main_activity') and self.main_activity and self.main_activity in graph:
            graph.nodes[self.main_activity]['peripheries'] = '2'  # Double border
        
        # Configure edge attributes
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('edgeType', '')
            icc_type = data.get('ICCType', '')
            
            # Add visual attributes based on edge type
            if edge_type == "Act2Act":
                graph.edges[u, v]['color'] = 'blue'
                graph.edges[u, v]['style'] = 'solid'
            elif edge_type == "Act2NonAct":
                graph.edges[u, v]['color'] = 'green'
                graph.edges[u, v]['style'] = 'dashed'
            elif edge_type == "NonAct2Act":
                graph.edges[u, v]['color'] = 'red'
                graph.edges[u, v]['style'] = 'dotted'
            elif edge_type == "Class2Any":
                graph.edges[u, v]['color'] = 'gray'
                graph.edges[u, v]['style'] = 'solid'
            elif edge_type == "DynamicallyDiscovered":
                graph.edges[u, v]['color'] = 'orange'
                graph.edges[u, v]['style'] = 'bold'
            
            # Add different arrowhead based on ICC type
            if icc_type == "explicit":
                graph.edges[u, v]['arrowhead'] = 'normal'
            elif icc_type == "implicit":
                graph.edges[u, v]['arrowhead'] = 'open'
            elif icc_type == "runtime":
                graph.edges[u, v]['arrowhead'] = 'diamond'
        
        # Create a safer version of the graph for nx_pydot
        safe_graph = nx.DiGraph()
        
        # Add nodes with safe names
        for node, attrs in graph.nodes(data=True):
            # Clean the node name by replacing problematic characters
            safe_node = str(node).replace(':', '_COLON_').replace('<', '_LT_').replace('>', '_GT_')
            safe_attrs = {}
            
            # Clean attribute values
            for k, v in attrs.items():
                if isinstance(v, str):
                    # Replace problematic characters in attribute values
                    safe_attrs[k] = v.replace(':', '_COLON_').replace('<', '_LT_').replace('>', '_GT_')
                else:
                    safe_attrs[k] = v
            
            safe_graph.add_node(safe_node, **safe_attrs)
        
        # Add edges with safe names and attributes
        for u, v, attrs in graph.edges(data=True):
            # Clean the node names
            safe_u = str(u).replace(':', '_COLON_').replace('<', '_LT_').replace('>', '_GT_')
            safe_v = str(v).replace(':', '_COLON_').replace('<', '_LT_').replace('>', '_GT_')
            
            safe_attrs = {}
            # Clean attribute values
            for k, v in attrs.items():
                if isinstance(v, str):
                    # Replace problematic characters in attribute values
                    safe_attrs[k] = v.replace(':', '_COLON_').replace('<', '_LT_').replace('>', '_GT_')
                else:
                    safe_attrs[k] = v
            
            safe_graph.add_edge(safe_u, safe_v, **safe_attrs)
        
        # Write the DOT file using the safe graph
        nx.drawing.nx_pydot.write_dot(safe_graph, output_file)
        print(f"Graph exported to {output_file}")
    
    def get_graph(self):
        """Return the NetworkX graph object."""
        return self.graph
    
    def get_nodes(self):
        """Return all nodes in the graph."""
        return list(self.graph.nodes())
    
    def get_edges(self):
        """Return all edges in the graph."""
        return list(self.graph.edges())
    
    def get_node_count(self):
        """Return the number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    def get_edge_count(self):
        """Return the number of edges in the graph."""
        return self.graph.number_of_edges()
    
    def get_node_types(self):
        """Return a dictionary of node types and their counts."""
        node_types = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        return node_types
    
    def get_edge_types(self):
        """Return a dictionary of edge types and their counts."""
        edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edgeType', 'Unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        return edge_types
    
    def get_icc_types(self):
        """Return a dictionary of ICC types and their counts."""
        icc_types = {}
        for u, v, data in self.graph.edges(data=True):
            icc_type = data.get('ICCType', 'Unknown')
            icc_types[icc_type] = icc_types.get(icc_type, 0) + 1
        return icc_types
    
    def get_summary(self):
        """Return a summary of the graph."""
        summary = {
            "nodes": self.get_node_count(),
            "edges": self.get_edge_count(),
            "node_types": self.get_node_types(),
            "edge_types": self.get_edge_types(),
            "icc_types": self.get_icc_types()
        }
        
        # Add pure UTG information if available
        if self.pure_utg:
            summary["pure_utg_nodes"] = self.pure_utg.number_of_nodes()
            summary["pure_utg_edges"] = self.pure_utg.number_of_edges()
        
        # Add main activity and path information if available
        if hasattr(self, 'main_activity') and self.main_activity:
            summary["main_activity"] = self.main_activity
            
            if hasattr(self, 'longest_shortest_paths') and self.longest_shortest_paths:
                summary["longest_shortest_path"] = self.longest_shortest_paths["max_length"]
                summary["reachable_nodes_from_main"] = len(self.longest_shortest_paths["paths"])
            
            if hasattr(self, 'reverse_longest_shortest_paths') and self.reverse_longest_shortest_paths:
                summary["longest_shortest_path_to_main"] = self.reverse_longest_shortest_paths["max_length"]
                summary["nodes_that_can_reach_main"] = len(self.reverse_longest_shortest_paths["paths"])
        
        # Add exploration statistics
        if self.current_node or self.visited_nodes:
            exploration_stats = self.get_exploration_stats()
            summary["exploration"] = exploration_stats
        
        return summary




class UTGStatsAnalyzer:
    def __init__(self, base_dir):
        """
        Initialize the UTG Stats Analyzer.
        
        Args:
            base_dir: Base directory containing SHA subdirectories with CTG.xml files
        """
        self.base_dir = base_dir
        self.utg_files = []
        self.manifest_files = []
        self.utg_graphs = []
        self.sha_list = []
        self.stats = {}
        
        # Find all CTG.xml files and corresponding manifest files
        self._find_files()
    
    def _find_files(self):
        """Find all CTG.xml files, componentInfo.xml files and manifest files in the SHA subdirectories."""
    # Find CTG.xml files
        pattern = os.path.join(self.base_dir, "*", "CTGResult", "CTG.xml")
        self.utg_files = glob.glob(pattern)
        
        # Find corresponding component info files and manifest files
        self.component_info_files = []
        self.manifest_files = []
        
        for utg_file in self.utg_files:
            # Extract the SHA part from the path
            parts = utg_file.split(os.sep)
            
            # Find the index of "ctg" in the path
            try:
                ctg_index = parts.index(self.base_dir)
                sha_index = ctg_index + 1
                if sha_index < len(parts):
                    sha = parts[sha_index]
                    
                    # Construct the component info file path
                    component_info_path = os.path.join(self.base_dir, sha, "CTGResult", "componentInfo.xml")
                    
                    # Construct the manifest file path
                    manifest_path = os.path.join(self.base_dir, sha, "ManifestInfo", "AndroidManifest.txt")
                    
                    # Check if the files exist
                    if os.path.exists(component_info_path):
                        self.component_info_files.append(component_info_path)
                    else:
                        self.component_info_files.append(None)
                        
                    if os.path.exists(manifest_path):
                        self.manifest_files.append(manifest_path)
                    else:
                        self.manifest_files.append(None)
                        
            except ValueError:
                # "ctg" not found in path
                self.component_info_files.append(None)
                self.manifest_files.append(None)
        
        print(f"Found {len(self.utg_files)} UTG files, "
            f"{len([f for f in self.component_info_files if f])} component info files, and "
            f"{len([f for f in self.manifest_files if f])} manifest files.")
    
    def load_all_utgs(self):
        """Load all UTG files into graph objects."""
        for i, utg_file in enumerate(self.utg_files):
            # Extract the SHA from the path
            parts = utg_file.split(os.sep)
            sha_index = parts.index("CTGResult") - 1
            sha = parts[sha_index]
            
            # Get the corresponding component info file and manifest file
            component_info_file = self.component_info_files[i] if i < len(self.component_info_files) else None
            manifest_file = self.manifest_files[i] if i < len(self.manifest_files) else None
            
            try:
                print(f"Loading UTG for SHA: {sha}")
                graph = utg(utg_file, component_info_file, manifest_file)
                # graph.to_dot(output_file=os.path.join(self.base_dir, sha, f"{sha}.dot"))
                self.utg_graphs.append(graph)
                self.sha_list.append(sha)
            except Exception as e:
                print(f"Error processing {utg_file}: {e}")
        
        print(f"Successfully loaded {len(self.utg_graphs)} UTG graphs.")
    def compute_stats(self):
        """Compute statistics for all UTG graphs."""
        if not self.utg_graphs:
            print("No UTG graphs loaded. Call load_all_utgs() first.")
            return
        
        # Initialize stats
        self.stats = {
            "sha": [],
            "nodes": [],
            "edges": [],
            "node_types": [],
            "edge_types": [],
            "icc_types": [],
            "density": [],
            "avg_degree": [],
            "main_activity": [],
            "longest_shortest_path": [],
            "reachable_nodes_from_main": [],
            "longest_shortest_path_to_main": [],
            "nodes_that_can_reach_main": [],
            "pure_utg_nodes": [],
            "pure_utg_edges": []
        }
        
        # Aggregate node types and edge types across all graphs
        all_node_types = Counter()
        all_edge_types = Counter()
        all_icc_types = Counter()
        
        # For calculating average longest shortest path
        total_longest_path = 0
        total_longest_path_to_main = 0
        graphs_with_main = 0
        
        # Collect stats for each graph
        for i, graph in enumerate(self.utg_graphs):
            sha = self.sha_list[i]
            summary = graph.get_summary()
            
            self.stats["sha"].append(sha)
            self.stats["nodes"].append(summary["nodes"])
            self.stats["edges"].append(summary["edges"])
            self.stats["node_types"].append(summary["node_types"])
            self.stats["edge_types"].append(summary["edge_types"])
            self.stats["icc_types"].append(summary["icc_types"])
            
            # Pure UTG statistics
            self.stats["pure_utg_nodes"].append(summary.get("pure_utg_nodes", 0))
            self.stats["pure_utg_edges"].append(summary.get("pure_utg_edges", 0))
            
            # Calculate graph density: E / (V * (V - 1)) for directed graphs
            if summary["nodes"] > 1:
                density = summary["edges"] / (summary["nodes"] * (summary["nodes"] - 1))
            else:
                density = 0
            self.stats["density"].append(density)
            
            # Calculate average degree: 2E / V
            if summary["nodes"] > 0:
                avg_degree = summary["edges"] / summary["nodes"]
            else:
                avg_degree = 0
            self.stats["avg_degree"].append(avg_degree)
            
            # Main activity and path information
            self.stats["main_activity"].append(summary.get("main_activity", None))
            self.stats["longest_shortest_path"].append(summary.get("longest_shortest_path", 0))
            self.stats["reachable_nodes_from_main"].append(summary.get("reachable_nodes_from_main", 0))
            self.stats["longest_shortest_path_to_main"].append(summary.get("longest_shortest_path_to_main", 0))
            self.stats["nodes_that_can_reach_main"].append(summary.get("nodes_that_can_reach_main", 0))
            
            # For average calculation
            if "main_activity" in summary and summary["main_activity"]:
                graphs_with_main += 1
                
                if "longest_shortest_path" in summary:
                    total_longest_path += summary["longest_shortest_path"]
                
                if "longest_shortest_path_to_main" in summary:
                    total_longest_path_to_main += summary["longest_shortest_path_to_main"]
            
            # Update counters for node types and edge types
            for node_type, count in summary["node_types"].items():
                all_node_types[node_type] += count
            
            for edge_type, count in summary["edge_types"].items():
                all_edge_types[edge_type] += count
                
            for icc_type, count in summary["icc_types"].items():
                all_icc_types[icc_type] += count
        
        # Calculate average longest shortest path
        avg_longest_path = total_longest_path / graphs_with_main if graphs_with_main > 0 else 0
        avg_longest_path_to_main = total_longest_path_to_main / graphs_with_main if graphs_with_main > 0 else 0
        
        # Save global stats
        self.global_stats = {
            "total_graphs": len(self.utg_graphs),
            "graphs_with_main_activity": graphs_with_main,
            "avg_nodes": np.mean(self.stats["nodes"]),
            "median_nodes": np.median(self.stats["nodes"]),
            "min_nodes": np.min(self.stats["nodes"]),
            "max_nodes": np.max(self.stats["nodes"]),
            "avg_edges": np.mean(self.stats["edges"]),
            "median_edges": np.median(self.stats["edges"]),
            "min_edges": np.min(self.stats["edges"]),
            "max_edges": np.max(self.stats["edges"]),
            "avg_density": np.mean(self.stats["density"]),
            "avg_degree": np.mean(self.stats["avg_degree"]),
            "avg_longest_shortest_path": avg_longest_path,
            "avg_longest_shortest_path_to_main": avg_longest_path_to_main,
            "avg_pure_utg_nodes": np.mean(self.stats["pure_utg_nodes"]),
            "avg_pure_utg_edges": np.mean(self.stats["pure_utg_edges"]),
            "all_node_types": dict(all_node_types),
            "all_edge_types": dict(all_edge_types),
            "all_icc_types": dict(all_icc_types)
        }
    
    def print_statistics(self):
        """Print statistics for all UTG graphs."""
        if not hasattr(self, 'global_stats'):
            print("No statistics computed. Call compute_stats() first.")
            return
        
        print("\n===== UTG STATISTICS =====")
        print(f"Total graphs analyzed: {self.global_stats['total_graphs']}")
        print(f"Graphs with main activity: {self.global_stats['graphs_with_main_activity']}")
        
        print("\n--- Node Statistics ---")
        print(f"Average nodes per graph: {self.global_stats['avg_nodes']:.2f}")
        print(f"Median nodes per graph: {self.global_stats['median_nodes']:.2f}")
        print(f"Min nodes: {self.global_stats['min_nodes']}")
        print(f"Max nodes: {self.global_stats['max_nodes']}")
        
        print("\n--- Edge Statistics ---")
        print(f"Average edges per graph: {self.global_stats['avg_edges']:.2f}")
        print(f"Median edges per graph: {self.global_stats['median_edges']:.2f}")
        print(f"Min edges: {self.global_stats['min_edges']}")
        print(f"Max edges: {self.global_stats['max_edges']}")
        
        print("\n--- Pure UTG Statistics ---")
        print(f"Average nodes in pure UTG: {self.global_stats['avg_pure_utg_nodes']:.2f}")
        print(f"Average edges in pure UTG: {self.global_stats['avg_pure_utg_edges']:.2f}")
        
        print("\n--- Graph Properties ---")
        print(f"Average density: {self.global_stats['avg_density']:.4f}")
        print(f"Average degree: {self.global_stats['avg_degree']:.2f}")
        
        print("\n--- Path Statistics ---")
        print(f"Average longest shortest path from main activity: {self.global_stats['avg_longest_shortest_path']:.2f}")
        print(f"Average longest shortest path to main activity: {self.global_stats['avg_longest_shortest_path_to_main']:.2f}")
        
        print("\n--- Node Type Distribution ---")
        for node_type, count in sorted(self.global_stats['all_node_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"{node_type}: {count}")
        
        print("\n--- Edge Type Distribution ---")
        for edge_type, count in sorted(self.global_stats['all_edge_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"{edge_type}: {count}")
        
        print("\n--- ICC Type Distribution ---")
        for icc_type, count in sorted(self.global_stats['all_icc_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"{icc_type}: {count}")
    
# Example usage
if __name__ == "__main__":


    utg=utg(xml_file='ICCBot_output/tasks/CTGResult/CTG.xml')
    pure_utg_df = nx.to_pandas_edgelist(utg.pure_utg)
    pure_utg_df.to_csv('pure_utg.csv', index=False)
