# src/models/knowledge_graph.py

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

class KGModel:
    def __init__(self, embedding_dim=32, walk_length=10, num_walks=80, workers=1):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.feature_embeddings = None
        self.disease_embedding = None
        self.clf = None
        self.scaler = StandardScaler()

    def build_graph(self, leakage_map, feature_names, disease):
        G = nx.Graph()
        for feat in feature_names:
            G.add_node(feat, type='feature')
        label = f"label_{disease}"
        G.add_node(label, type='disease')
        exclude = leakage_map.get(label, [])
        for feat in feature_names:
            if feat not in exclude:
                G.add_edge(feat, label, weight=1.0)
        return G

    def fit(self, X_train, y_train, disease, leakage_map, feature_names):
        # Ensure feature_names length matches X_train columns
        n_features = min(len(feature_names), X_train.shape[1])
        feature_names_trimmed = feature_names[:n_features]

        G = self.build_graph(leakage_map, feature_names_trimmed, disease)

        node2vec = Node2Vec(G, dimensions=self.embedding_dim,
                            walk_length=self.walk_length,
                            num_walks=self.num_walks, workers=self.workers)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        self.feature_embeddings = {}
        for feat in feature_names_trimmed:
            if feat in model.wv:
                self.feature_embeddings[feat] = model.wv[feat]
            else:
                self.feature_embeddings[feat] = np.zeros(self.embedding_dim)

        label = f"label_{disease}"
        self.disease_embedding = model.wv[label] if label in model.wv else np.zeros(self.embedding_dim)

        def patient_projection(X):
            n_samples = X.shape[0]
            proj = np.zeros((n_samples, self.embedding_dim))
            n_cols = min(len(feature_names_trimmed), X.shape[1])
            for i in range(n_cols):
                feat = feature_names_trimmed[i]
                if feat in self.feature_embeddings:
                    proj += np.outer(X[:, i], self.feature_embeddings[feat])
            return proj

        X_proj = patient_projection(X_train)
        self.clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                      class_weight='balanced')
        self.clf.fit(X_proj, y_train)
        return self

    def predict_proba(self, X, feature_names):
        n_cols = min(len(feature_names), X.shape[1])
        X_proj = np.zeros((X.shape[0], self.embedding_dim))
        for i in range(n_cols):
            feat = feature_names[i]
            if feat in self.feature_embeddings:
                X_proj += np.outer(X[:, i], self.feature_embeddings[feat])
        return self.clf.predict_proba(X_proj)

    def predict(self, X, feature_names):
        proba = self.predict_proba(X, feature_names)
        return (proba[:, 1] >= 0.5).astype(int)