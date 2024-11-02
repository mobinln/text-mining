import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TermGraph:
    def __init__(self, min_support: float = 0.1):
        """
        Initialize the Term Graph model.
        
        Args:
            min_support (float): Minimum support threshold for frequent itemsets (0-1)
        """
        self.min_support = min_support
        self.nodes = set()  # Set of unique terms
        self.edge_weights = defaultdict(float)  # Dictionary to store edge weights
        self.frequent_itemsets = []  # List to store frequent itemsets with their support
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_document(self, text: str) -> List[str]:
        """
        Preprocess the text document by removing stop words and stemming.
        
        Args:
            text (str): Input text document
            
        Returns:
            List[str]: Preprocessed list of terms
        """
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and apply stemming
        processed_terms = [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return processed_terms
        
    def find_frequent_itemsets(self, documents: List[str]) -> List[Tuple[Set[str], int]]:
        """
        Find frequent itemsets using the Apriori algorithm.
        
        Args:
            documents (List[str]): List of text documents
            
        Returns:
            List[Tuple[Set[str], int]]: List of frequent itemsets with their support count
        """
        # Process all documents and create transactions
        transactions = [set(self.preprocess_document(doc)) for doc in documents]
        n_transactions = len(transactions)
        min_support_count = int(self.min_support * n_transactions)
        
        # Generate frequent 1-itemsets
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
                
        frequent_1_itemsets = {
            frozenset([item]): count 
            for item, count in item_counts.items() 
            if count >= min_support_count
        }
        
        k = 2
        frequent_itemsets = []
        current_frequent = frequent_1_itemsets
        
        while current_frequent:
            frequent_itemsets.extend([
                (set(itemset), count) 
                for itemset, count in current_frequent.items()
            ])
            
            # Generate candidate k-itemsets
            candidates = self._generate_candidates(current_frequent.keys(), k)
            
            # Count support for candidates
            candidate_counts = defaultdict(int)
            for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1
                        
            # Filter candidates by minimum support
            current_frequent = {
                itemset: count 
                for itemset, count in candidate_counts.items() 
                if count >= min_support_count
            }
            
            k += 1
            
        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets
    
    def _generate_candidates(self, prev_frequent: Set[frozenset], k: int) -> Set[frozenset]:
        """
        Generate candidate k-itemsets from (k-1)-itemsets.
        
        Args:
            prev_frequent (Set[frozenset]): Set of frequent (k-1)-itemsets
            k (int): Size of itemsets to generate
            
        Returns:
            Set[frozenset]: Set of candidate k-itemsets
        """
        candidates = set()
        for itemset1 in prev_frequent:
            for itemset2 in prev_frequent:
                union = itemset1.union(itemset2)
                if len(union) == k:
                    candidates.add(union)
        return candidates
    
    def build_graph(self, documents: List[str]) -> Tuple[Set[str], Dict[Tuple[str, str], float]]:
        """
        Build the term graph from a collection of documents.
        
        Args:
            documents (List[str]): List of text documents
            
        Returns:
            Tuple[Set[str], Dict[Tuple[str, str], float]]: Nodes and edge weights of the graph
        """
        # Find frequent itemsets
        frequent_itemsets = self.find_frequent_itemsets(documents)
        
        # Create nodes for each term in frequent itemsets
        self.nodes = set()
        for itemset, _ in frequent_itemsets:
            self.nodes.update(itemset)
            
        # Create edges and assign weights
        self.edge_weights = defaultdict(float)
        for itemset, support in frequent_itemsets:
            # Create edges between all pairs of terms in the itemset
            terms = list(itemset)
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    edge = (terms[i], terms[j])
                    # Update edge weight if current support is larger
                    self.edge_weights[edge] = max(self.edge_weights[edge], support)
                    
        return self.nodes, self.edge_weights
    
    def calculate_distances(self) -> np.ndarray:
        """
        Calculate the distance matrix using Dijkstra's algorithm.
        
        Returns:
            np.ndarray: Distance matrix
        """
        # Convert nodes to indices
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        n_nodes = len(self.nodes)
        
        # Initialize distance matrix
        distances = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(distances, 0)
        
        # Fill initial distances from edge weights
        for (u, v), weight in self.edge_weights.items():
            i, j = node_to_idx[u], node_to_idx[v]
            distances[i, j] = weight
            distances[j, i] = weight  # Assuming undirected graph
            
        # Floyd-Warshall algorithm for all-pairs shortest paths
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
                        
        return distances
    
    def calculate_similarity(self, test_doc: str, category_docs: List[str]) -> float:
        """
        Calculate similarity between test document and category documents.
        
        Args:
            test_doc (str): Test document
            category_docs (List[str]): List of documents in a category
            
        Returns:
            float: Similarity score
        """
        # Build graph for category documents
        self.build_graph(category_docs)
        
        # Process test document
        test_terms = set(self.preprocess_document(test_doc))
        
        # Calculate similarity based on graph distances
        distances = self.calculate_distances()
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        
        # Calculate average distance between test terms and category terms
        total_distance = 0
        count = 0
        
        for test_term in test_terms:
            if test_term in node_to_idx:
                test_idx = node_to_idx[test_term]
                # Calculate average distance to all category terms
                valid_distances = distances[test_idx][distances[test_idx] != np.inf]
                if len(valid_distances) > 0:
                    total_distance += np.mean(valid_distances)
                    count += 1
                    
        if count == 0:
            return 0.0
        
        # Convert distance to similarity (inverse relationship)
        similarity = 1.0 / (1.0 + total_distance / count)
        return similarity

# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the fox",
        "The lazy fox sleeps under the tree"
    ]
    
    # Create and build term graph
    term_graph = TermGraph(min_support=0.3)
    nodes, edge_weights = term_graph.build_graph(documents)
    
    # Print graph structure
    print("Nodes:", nodes)
    print("\nEdge weights:")
    for (u, v), weight in edge_weights.items():
        print(f"{u} -- {weight} --> {v}")
    
    # Calculate distances
    distances = term_graph.calculate_distances()
    print("\nDistance matrix:")
    print(distances)
    
    # Calculate similarity with a test document
    test_doc = "A brown fox jumps quickly"
    similarity = term_graph.calculate_similarity(test_doc, documents)
    print(f"\nSimilarity score: {similarity:.4f}")