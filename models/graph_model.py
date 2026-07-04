import pandas as pd
import networkx as nx
import numpy as np

def build_transaction_graph(df):
    """
    Nodes: account IDs + merchant IDs
    Edges: transactions between them
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        src = f"ACC_{int(row['V1']*100) % 500}"   # proxy account ID
        dst = f"MRC_{int(row['V2']*100) % 200}"   # proxy merchant ID
        G.add_edge(src, dst,
                   amount=row['Amount'],
                   time=row['Time'],
                   fraud=row['Class'])
    return G

def compute_graph_features(G):
    """Compute node-level features for GNN or heuristic scoring"""
    features = {}
    degree_cent = nx.degree_centrality(G)
    pagerank    = nx.pagerank(G, alpha=0.85)

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        fraud_edges = [e for e in out_edges if e[2].get('fraud', 0) == 1]
        features[node] = {
            'degree_centrality': degree_cent.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'out_degree': G.out_degree(node),
            'fraud_ratio': len(fraud_edges) / max(len(out_edges), 1),
            'avg_amount': np.mean([e[2]['amount'] for e in out_edges]) if out_edges else 0
        }
    return pd.DataFrame(features).T

def detect_fraud_rings(G, min_size=3):
    """Find strongly connected components — likely fraud rings"""
    rings = [c for c in nx.strongly_connected_components(G) if len(c) >= min_size]
    print(f"Found {len(rings)} suspected fraud rings")
    return rings