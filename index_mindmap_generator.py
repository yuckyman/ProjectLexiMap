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
import re
from typing import List, Dict, Tuple, Optional, Union, Set
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Create directories
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MINDMAP_DIR = RESULTS_DIR / "mindmaps"
MINDMAP_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def debug_print(message: str, important: bool = False):
    """Print debug message with timestamp and flush immediately."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    prefix = "🔴" if important else "🔹"
    print(f"{prefix} [{timestamp}] {message}", flush=True)

def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword by removing special characters and converting to lowercase."""
    # Remove special characters and extra whitespace
    keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword

class IndexMindMapGenerator:
    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.65,
        clustering_threshold: float = 0.25,
        max_keywords_per_chapter: int = 50,
        min_edge_weight: float = 0.6,
        node_scaling_factor: float = 5,
        edge_scaling_factor: float = 5,
        use_cache: bool = True,
        verbose: bool = True
    ):
        """Initialize the mind map generator with the specified parameters."""
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.clustering_threshold = clustering_threshold
        self.max_keywords_per_chapter = max_keywords_per_chapter
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
        cache_file = CACHE_DIR / f"index_embeddings_{self.model_name.replace('/', '_')}.json"
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
        cache_file = CACHE_DIR / f"index_embeddings_{self.model_name.replace('/', '_')}.json"
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
    
    def parse_index_file(self, index_file: str) -> Dict[int, List[str]]:
        """Parse the index_by_chapter.txt file to extract keywords by chapter."""
        if self.verbose:
            debug_print(f"Parsing index file: {index_file}...")
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            keywords_by_chapter = {}
            current_chapter = None
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('Chapter '):
                    try:
                        current_chapter = int(line.split()[1])
                        keywords_by_chapter[current_chapter] = []
                        if self.verbose:
                            debug_print(f"Found Chapter {current_chapter}")
                    except (ValueError, IndexError):
                        debug_print(f"Error parsing chapter from line: {line}", important=True)
                        continue
                elif current_chapter is not None:
                    # Add the keyword to the current chapter
                    keyword = line.strip()
                    keywords_by_chapter[current_chapter].append(keyword)
            
            if self.verbose:
                total_keywords = sum(len(kws) for kws in keywords_by_chapter.values())
                debug_print(f"Parsed {total_keywords} keywords from {len(keywords_by_chapter)} chapters")
                
            return keywords_by_chapter
            
        except Exception as e:
            debug_print(f"Error parsing index file: {e}", important=True)
            return {}
    
    def build_graph(self, chapters: List[int], keywords_by_chapter: Dict[int, List[str]]) -> nx.Graph:
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
                # Limit to top keywords if too many
                if len(chapter_keywords) > self.max_keywords_per_chapter:
                    debug_print(f"Limiting chapter {chapter} keywords from {len(chapter_keywords)} to {self.max_keywords_per_chapter}")
                    chapter_keywords = chapter_keywords[:self.max_keywords_per_chapter]
                
                for keyword in chapter_keywords:
                    # Assign a default score of 1.0 for index keywords
                    all_keywords.append((keyword, 1.0, chapter))
        
        if self.verbose:
            debug_print(f"Processing {len(all_keywords)} keywords...")
        
        # Check if we have any keywords
        if not all_keywords:
            if self.verbose:
                debug_print("No keywords found. Creating empty graph.", important=True)
            # Return empty graph with a single placeholder node
            G.add_node("No keywords found", size=20, title="No keywords were found in the specified chapters", color="#ff0000")
            return G
        
        # Get a good color palette for chapters
        chapter_palette = sns.color_palette("hls", len(chapters)).as_hex()
        chapter_colors = {ch: chapter_palette[i] for i, ch in enumerate(sorted(chapters))}
            
        # Add nodes
        for keyword, score, chapter in all_keywords:
            # Add node with attributes
            if not G.has_node(keyword):
                G.add_node(
                    keyword,
                    score=score,
                    chapter=chapter,
                    size=score * self.node_scaling_factor,
                    title=f"Keyword: {keyword}<br>Chapter: {chapter}",
                    group=chapter,  # Use chapter number as group
                    color=chapter_colors[chapter]  # Assign color based on chapter
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
            
            # Add chapter nodes and connect them to their keywords
            for chapter in chapters:
                if chapter in keywords_by_chapter and keywords_by_chapter[chapter]:
                    # Add chapter node
                    chapter_node_name = f"Chapter {chapter}"
                    G.add_node(
                        chapter_node_name,
                        title=f"Chapter {chapter}",
                        size=15,  # Make chapter nodes bigger
                        shape="diamond",  # Different shape for chapter nodes
                        group=chapter,
                        color=chapter_colors[chapter],
                        font={"size": 20, "bold": True}  # Bigger font for chapter labels
                    )
                    
                    # Connect chapter node to all its keywords
                    for keyword, _, kw_chapter in all_keywords:
                        if kw_chapter == chapter and G.has_node(keyword):
                            G.add_edge(
                                chapter_node_name,
                                keyword,
                                weight=1.0,  # Strong connection
                                width=1.0,   # Consistent width
                                color={"color": chapter_colors[chapter], "opacity": 0.7},
                                dashes=True  # Dashed lines for chapter connections
                            )
            
        else:
            if self.verbose:
                debug_print("No valid embeddings found. Unable to calculate similarities.", important=True)
        
        return G
    
    def export_interactive_html(self, G: nx.Graph, output_file: str, title: str = "Textbook Index Knowledge Graph"):
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
              "springLength": 200,
              "springConstant": 0.05
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
          },
          "layout": {
            "improvedLayout": true,
            "hierarchical": {
              "enabled": false
            }
          }
        }
        """)
        
        # Add nodes
        for node, attrs in G.nodes(data=True):
            # Convert numpy types to native Python types
            size = float(attrs.get('size', 10)) if hasattr(attrs.get('size', 10), 'item') else attrs.get('size', 10)
            group = int(attrs.get('group', 0)) if hasattr(attrs.get('group', 0), 'item') else attrs.get('group', 0)
            
            # Prepare node attributes
            node_attrs = {
                "label": node,
                "title": attrs.get('title', node),
                "size": size,
                "color": attrs.get('color', "#1f77b4"),
                "group": group
            }
            
            # Add shape if specified
            if 'shape' in attrs:
                node_attrs['shape'] = attrs['shape']
                
            # Add font attributes if specified
            if 'font' in attrs:
                node_attrs['font'] = attrs['font']
            
            net.add_node(
                node,
                **node_attrs
            )
        
        # Add edges
        for u, v, attrs in G.edges(data=True):
            # Convert numpy types to native Python types
            width = float(attrs.get('width', 1)) if hasattr(attrs.get('width', 1), 'item') else attrs.get('width', 1)
            
            # Prepare edge attributes
            edge_attrs = {
                "width": width,
                "title": attrs.get('title', ''),
                "arrowStrikethrough": False
            }
            
            # Add color if specified
            if 'color' in attrs:
                edge_attrs['color'] = attrs['color']
            else:
                weight = float(attrs.get('weight', 0.5)) if hasattr(attrs.get('weight', 0.5), 'item') else attrs.get('weight', 0.5)
                edge_attrs['color'] = {'opacity': min(1.0, weight + 0.2)}
                
            # Add dashes if specified
            if 'dashes' in attrs and attrs['dashes']:
                edge_attrs['dashes'] = True
            
            net.add_edge(
                u, v,
                **edge_attrs
            )
        
        # Add custom HTML header with title
        net.html = net.html.replace('<center>', f'<center><h1>{title}</h1>')
        
        # Save the visualization
        net.save_graph(output_file)
        
        if self.verbose:
            debug_print(f"Interactive visualization saved to {output_file}", important=True)
    
    def generate_mindmap(self, 
                         chapters: List[int],
                         index_file: str,
                         output_file: Optional[str] = None,
                         title: str = "Textbook Index Knowledge Graph"):
        """Generate a mind map from index keywords."""
        # Parse index file to get keywords by chapter
        keywords_by_chapter = self.parse_index_file(index_file)
        
        # Build graph
        G = self.build_graph(chapters, keywords_by_chapter)
        
        # Generate output filename if not provided
        if output_file is None:
            chapters_str = "-".join(str(ch) for ch in sorted(chapters))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = MINDMAP_DIR / f"index_mindmap_ch{chapters_str}_{timestamp}.html"
        
        # Export interactive visualization
        self.export_interactive_html(G, str(output_file), title)
        
        # Save the embedding cache for future use
        self.save_embedding_cache()
        
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth knowledge graph from textbook index")
    parser.add_argument(
        "--chapters", 
        type=int, 
        nargs="+", 
        help="Chapter numbers to include in the mind map"
    )
    parser.add_argument(
        "--index", 
        type=str, 
        default="textbook/index_by_chapter.txt",
        help="Path to the index_by_chapter.txt file"
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
        default=50,
        help="Maximum number of keywords per chapter"
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
        default="Textbook Index Knowledge Graph (Ground Truth)",
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
    
    # Default to test chapters if none specified
    if not args.chapters:
        args.chapters = [6, 10, 11, 12]  # Test chapters
    
    # Check if index file exists
    if not os.path.exists(args.index):
        print(f"Error: Index file '{args.index}' not found.")
        print("Make sure you have index_by_chapter.txt in the textbook directory.")
        return
    
    # Create mind map generator
    generator = IndexMindMapGenerator(
        model_name=args.model,
        similarity_threshold=args.similarity,
        clustering_threshold=args.clustering,
        max_keywords_per_chapter=args.max_keywords,
        min_edge_weight=args.min_edge_weight,
        use_cache=not args.no_cache,
        verbose=not args.quiet
    )
    
    # Generate mind map
    output_file = generator.generate_mindmap(
        chapters=args.chapters,
        index_file=args.index,
        output_file=args.output,
        title=args.title
    )
    
    print(f"\nGround truth mind map successfully generated: {output_file}")
    print(f"Open this file in a web browser to view the interactive knowledge graph.")

if __name__ == "__main__":
    main() 