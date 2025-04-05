#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from pyvis.network import Network
import time
from typing import List, Dict, Tuple, Optional, Union, Set
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Import utility functions from existing scripts
try:
    from keybert_unified import debug_print, load_chapter, normalize_keyword
except ImportError:
    # Fallback implementation if imports fail
    def debug_print(message: str, important: bool = False):
        """Print debug message with timestamp and flush immediately."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        prefix = "🔴" if important else "🔹"
        print(f"{prefix} [{timestamp}] {message}", flush=True)
    
    def normalize_keyword(keyword: str) -> str:
        """Normalize a keyword by removing special characters and converting to lowercase."""
        import re
        keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
        keyword = re.sub(r'\s+', ' ', keyword).strip()
        return keyword
    
    def load_chapter(chapter_num: int, verbose: bool = True) -> str:
        """Load a chapter from the textbook folder."""
        try:
            file_path = f'textbook/ch{chapter_num}.txt'
            with open(file_path, 'r') as f:
                content = f.read()
                return content
        except Exception as e:
            if verbose:
                debug_print(f"Error loading chapter {chapter_num}: {e}", important=True)
            return ""

# Create directories
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MINDMAP_DIR = RESULTS_DIR / "mindmaps"
MINDMAP_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

class MindMapGenerator:
    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.65,
        clustering_threshold: float = 0.25,
        max_keywords: int = 150,
        min_edge_weight: float = 0.6,
        node_scaling_factor: float = 5,
        edge_scaling_factor: float = 5,
        use_cache: bool = True,
        verbose: bool = True
    ):
        """Initialize the mind map generator with the specified parameters.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Threshold for connecting keywords in the graph
            clustering_threshold: Threshold for clustering related keywords
            max_keywords: Maximum number of keywords to include in the graph
            min_edge_weight: Minimum weight for edges to be included
            node_scaling_factor: Factor to scale node sizes
            edge_scaling_factor: Factor to scale edge widths
            use_cache: Whether to use cached embeddings
            verbose: Whether to print detailed progress messages
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.clustering_threshold = clustering_threshold
        self.max_keywords = max_keywords
        self.min_edge_weight = min_edge_weight
        self.node_scaling_factor = node_scaling_factor
        self.edge_scaling_factor = edge_scaling_factor
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Load model
        if self.verbose:
            debug_print(f"Loading model: {model_name}...", important=True)
        self.model = SentenceTransformer(model_name)
        
        # Initialize cache for embeddings
        self.embedding_cache = {}
        self.load_embedding_cache()
    
    def load_embedding_cache(self):
        """Load cached embeddings if they exist."""
        cache_file = CACHE_DIR / f"embeddings_{self.model_name.replace('/', '_')}.json"
        if cache_file.exists() and self.use_cache:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Convert string keys back to embeddings
                    for keyword, embedding_list in cached_data.items():
                        self.embedding_cache[keyword] = np.array(embedding_list)
                if self.verbose:
                    debug_print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                if self.verbose:
                    debug_print(f"Error loading embedding cache: {e}")
    
    def save_embedding_cache(self):
        """Save embeddings cache to disk."""
        cache_file = CACHE_DIR / f"embeddings_{self.model_name.replace('/', '_')}.json"
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_cache = {k: v.tolist() for k, v in self.embedding_cache.items()}
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f)
            if self.verbose:
                debug_print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            if self.verbose:
                debug_print(f"Error saving embedding cache: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string, using cache if available."""
        text = normalize_keyword(text)
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.embedding_cache[text] = embedding
        return embedding
    
    def load_keywords_from_results(self, results_file: str) -> Dict[int, List[Tuple[str, float]]]:
        """Load keywords from evaluation results file."""
        if self.verbose:
            debug_print(f"Loading keywords from {results_file}...")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            keywords_by_chapter = {}
            
            # Check for different possible structures
            if "chapter_metrics" in results:
                # Format from keybert_evaluator.py
                for chapter_str, chapter_data in results.get("chapter_metrics", {}).items():
                    try:
                        chapter_num = int(chapter_str)
                    except ValueError:
                        # Try to extract chapter number from string like "chapter_6"
                        try:
                            chapter_num = int(chapter_str.split('_')[-1])
                        except (ValueError, IndexError):
                            continue
                            
                    extracted_keywords = chapter_data.get("extracted_keywords", [])
                    # Convert to list of tuples if needed
                    if isinstance(extracted_keywords, dict):
                        extracted_keywords = [(k, v) for k, v in extracted_keywords.items()]
                    keywords_by_chapter[chapter_num] = extracted_keywords
            
            elif "results" in results:
                # Format from keybert_unified.py
                for chapter_key, chapter_data in results.get("results", {}).items():
                    # Extract chapter number from string like "chapter_6"
                    try:
                        if chapter_key.startswith("chapter_"):
                            chapter_num = int(chapter_key.split("_")[1])
                        else:
                            continue
                        
                        extracted_keywords = chapter_data.get("extracted_keywords", [])
                        # Check if it's already a list of lists/tuples
                        if extracted_keywords and isinstance(extracted_keywords[0], (list, tuple)):
                            keywords_by_chapter[chapter_num] = [
                                (kw, score) for kw, score in extracted_keywords
                            ]
                        else:
                            # Handle other formats if needed
                            keywords_by_chapter[chapter_num] = extracted_keywords
                    except (ValueError, IndexError, AttributeError) as e:
                        if self.verbose:
                            debug_print(f"Error processing chapter key {chapter_key}: {e}")
                        continue
            
            if self.verbose:
                debug_print(f"Loaded keywords for {len(keywords_by_chapter)} chapters")
                if not keywords_by_chapter:
                    debug_print("No keywords were found in the results file - check the file format", important=True)
                    debug_print("If using the default file, try running: python keybert_unified.py evaluate", important=True)
            
            return keywords_by_chapter
        
        except Exception as e:
            if self.verbose:
                debug_print(f"Error loading keywords from results: {e}", important=True)
            return {}
    
    def build_graph(self, chapters: List[int], keywords_by_chapter: Dict[int, List[Tuple[str, float]]]) -> nx.Graph:
        """Build the knowledge graph from keywords across specified chapters."""
        if self.verbose:
            debug_print(f"Building knowledge graph for chapters {chapters}...", important=True)
        
        # Create graph
        G = nx.Graph()
        
        # Collect all keywords from specified chapters
        all_keywords = []
        for chapter in chapters:
            if chapter in keywords_by_chapter:
                chapter_keywords = keywords_by_chapter[chapter]
                # Limit to top keywords by score
                sorted_keywords = sorted(chapter_keywords, key=lambda x: x[1], reverse=True)
                top_keywords = sorted_keywords[:min(len(sorted_keywords), self.max_keywords // len(chapters))]
                for keyword, score in top_keywords:
                    all_keywords.append((keyword, score, chapter))
        
        # Further limit if still too many
        if len(all_keywords) > self.max_keywords:
            all_keywords = sorted(all_keywords, key=lambda x: x[1], reverse=True)[:self.max_keywords]
        
        if self.verbose:
            debug_print(f"Processing {len(all_keywords)} keywords...")
        
        # Check if we have any keywords
        if not all_keywords:
            if self.verbose:
                debug_print("No keywords found. Creating empty graph.", important=True)
            # Return empty graph with a single placeholder node
            G.add_node("No keywords found", size=20, title="No keywords were found in the specified chapters", color="#ff0000")
            return G
            
        # Add nodes
        for keyword, score, chapter in all_keywords:
            # Add node with attributes
            if not G.has_node(keyword):
                G.add_node(
                    keyword,
                    score=score,
                    chapter=chapter,
                    size=score * self.node_scaling_factor,
                    title=f"Keyword: {keyword}<br>Score: {score:.3f}<br>Chapter: {chapter}"
                )
        
        # Compute embeddings for all keywords
        if self.verbose:
            debug_print("Computing embeddings for all keywords...")
        
        embeddings = {}
        for keyword, _, _ in all_keywords:
            embeddings[keyword] = self.get_embedding(keyword)
        
        # Connect related keywords based on semantic similarity
        if self.verbose:
            debug_print("Connecting related keywords...")
        
        # Calculate similarity matrix
        keywords = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[k] for k in keywords])
        
        # Check if embedding matrix is not empty before calculating similarity
        if embedding_matrix.size > 0:
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            # Add edges based on similarity
            edge_count = 0
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    similarity = similarity_matrix[i, j]
                    if similarity >= self.similarity_threshold:
                        weight = similarity
                        if weight >= self.min_edge_weight:
                            G.add_edge(
                                keywords[i], 
                                keywords[j], 
                                weight=weight,
                                width=weight * self.edge_scaling_factor,
                                title=f"Similarity: {weight:.3f}"
                            )
                            edge_count += 1
            
            if self.verbose:
                debug_print(f"Created graph with {len(G.nodes)} nodes and {edge_count} edges")
            
            # Apply clustering to identify keyword clusters
            if len(G.nodes) > 1:
                self.apply_clustering(G, embeddings)
        else:
            if self.verbose:
                debug_print("No valid embeddings found. Unable to calculate similarities.", important=True)
        
        return G
    
    def apply_clustering(self, G: nx.Graph, embeddings: Dict[str, np.ndarray]):
        """Apply clustering to identify groups of related keywords."""
        if self.verbose:
            debug_print("Applying clustering to identify keyword groups...")
        
        # Prepare data for clustering
        nodes = list(G.nodes())
        X = np.array([embeddings[node] for node in nodes])
        
        # Determine optimal number of clusters
        n_clusters = min(max(3, len(nodes) // 10), 15)  # Between 3 and 15 clusters
        
        # Apply clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',
            distance_threshold=None
        )
        labels = clustering.fit_predict(X)
        
        # Get a good color palette
        palette = sns.color_palette("hls", n_clusters).as_hex()
        
        # Apply cluster information to nodes
        for i, node in enumerate(nodes):
            cluster_id = labels[i]
            G.nodes[node]['group'] = int(cluster_id)
            G.nodes[node]['color'] = palette[cluster_id]
    
    def export_interactive_html(self, G: nx.Graph, output_file: str, title: str = "KeyBERT Knowledge Graph"):
        """Create an interactive HTML visualization of the graph."""
        if self.verbose:
            debug_print(f"Creating interactive visualization: {output_file}")
        
        # Create network
        net = Network(
            height="900px",
            width="100%",
            notebook=False,
            directed=False,
            bgcolor="#222222",
            font_color="white"
        )
        
        # Set options
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -100,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true,
            "hover": true
          }
        }
        """)
        
        # Add nodes
        for node, attrs in G.nodes(data=True):
            # Convert numpy types to native Python types
            size = float(attrs.get('size', 10)) if hasattr(attrs.get('size', 10), 'item') else attrs.get('size', 10)
            group = int(attrs.get('group', 0)) if hasattr(attrs.get('group', 0), 'item') else attrs.get('group', 0)
            
            net.add_node(
                node,
                label=node,
                title=attrs.get('title', node),
                size=size,
                color=attrs.get('color', "#1f77b4"),
                group=group
            )
        
        # Add edges
        for u, v, attrs in G.edges(data=True):
            # Convert numpy types to native Python types
            width = float(attrs.get('width', 1)) if hasattr(attrs.get('width', 1), 'item') else attrs.get('width', 1)
            weight = float(attrs.get('weight', 0.5)) if hasattr(attrs.get('weight', 0.5), 'item') else attrs.get('weight', 0.5)
            
            net.add_edge(
                u, v,
                width=width,
                title=attrs.get('title', ''),
                arrowStrikethrough=False,
                color={'opacity': min(1.0, weight + 0.2)}
            )
        
        # Add custom HTML header with title
        net.html = net.html.replace('<center>', f'<center><h1>{title}</h1>')
        
        # Save the visualization
        net.save_graph(output_file)
        
        if self.verbose:
            debug_print(f"Interactive visualization saved to {output_file}", important=True)
    
    def generate_mindmap(self, 
                         chapters: List[int], 
                         results_file: str,
                         output_file: Optional[str] = None,
                         title: str = "KeyBERT Knowledge Graph"):
        """Generate a mind map from extracted keywords."""
        # Load keywords from results
        keywords_by_chapter = self.load_keywords_from_results(results_file)
        
        # Build graph
        G = self.build_graph(chapters, keywords_by_chapter)
        
        # Generate output filename if not provided
        if output_file is None:
            chapters_str = "-".join(str(ch) for ch in sorted(chapters))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = MINDMAP_DIR / f"mindmap_ch{chapters_str}_{timestamp}.html"
        
        # Export interactive visualization
        self.export_interactive_html(G, str(output_file), title)
        
        # Save the embedding cache for future use
        self.save_embedding_cache()
        
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate interactive knowledge graph mind maps from extracted keywords")
    parser.add_argument(
        "--chapters", 
        type=int, 
        nargs="+", 
        help="Chapter numbers to include in the mind map"
    )
    parser.add_argument(
        "--results", 
        type=str, 
        default="results/keybert_evaluation_results.json",
        help="Path to keybert evaluation results file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-mpnet-base-v2",
        help="Sentence transformer model to use"
    )
    parser.add_argument(
        "--similarity", 
        type=float, 
        default=0.65, 
        help="Threshold for connecting keywords (0-1)"
    )
    parser.add_argument(
        "--clustering", 
        type=float, 
        default=0.25, 
        help="Threshold for clustering related keywords (0-1)"
    )
    parser.add_argument(
        "--max_keywords", 
        type=int, 
        default=150,
        help="Maximum number of keywords to include in the graph"
    )
    parser.add_argument(
        "--min_edge_weight", 
        type=float, 
        default=0.6,
        help="Minimum weight for edges to be included"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output HTML file path"
    )
    parser.add_argument(
        "--title", 
        type=str, 
        default="KeyBERT Knowledge Graph",
        help="Title for the knowledge graph visualization"
    )
    parser.add_argument(
        "--no_cache", 
        action="store_true", 
        help="Disable embedding cache"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Default to all test chapters if none specified
    if not args.chapters:
        args.chapters = [6, 10, 11, 12]  # Test chapters from unified script
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"Error: Results file '{args.results}' not found.")
        print("To fix this, make sure you've run the keyword extraction first:")
        print("  python keybert_unified.py evaluate --clean_results")
        print("\nAlternatively, specify a different results file with --results")
        sys.exit(1)
    
    # Create mind map generator
    generator = MindMapGenerator(
        model_name=args.model,
        similarity_threshold=args.similarity,
        clustering_threshold=args.clustering,
        max_keywords=args.max_keywords,
        min_edge_weight=args.min_edge_weight,
        use_cache=not args.no_cache,
        verbose=not args.quiet
    )
    
    # Generate mind map
    output_file = generator.generate_mindmap(
        chapters=args.chapters,
        results_file=args.results,
        output_file=args.output,
        title=args.title
    )
    
    print(f"\nMind map successfully generated: {output_file}")
    print(f"Open this file in a web browser to view the interactive knowledge graph.")

if __name__ == "__main__":
    main() 