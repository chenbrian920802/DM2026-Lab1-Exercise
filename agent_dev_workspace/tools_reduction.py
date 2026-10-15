import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Literal, Dict, Any, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool

def make_tools(session):
    """
    Returns the list of dimensionality reduction and label tools initialized with session context.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reduces feature dimensions using PCA, t-SNE, or UMAP, stores coordinates,
        and generates a scatter plot saved to session.pending_figure.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            n_components: Number of components/dimensions to reduce to (default: 2).
            perplexity: Perplexity parameter for t-SNE (default: 30.0).
            n_neighbors: Number of neighbors for UMAP (default: 15).
            random_state: Seed for reproducible dimensionality reduction.

        Returns:
            A summary dictionary containing result_id, method, coordinates,
            and explained_variance_ratio (if method is PCA).
        """
        if session.feature_matrix is None and session.dataframe is None:
            return {"error": "Neither feature_matrix nor dataframe exists in session."}

        # Prefer session.feature_matrix if available; otherwise use dataframe numeric columns
        if session.feature_matrix is not None:
            X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.array(session.feature_matrix)
        else:
            num_df = session.dataframe.select_dtypes(include=[np.number])
            if num_df.empty:
                return {"error": "No numeric feature columns found in dataframe."}
            X = num_df.values

        # Category labels for plotting and coloring
        categories = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                categories = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                categories = session.dataframe["category"].tolist()

        explained_variance_ratio = None

        # Perform reduction
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Unknown method '{method}'."}

        # Build required scatter plot figure
        fig, ax = plt.subplots(figsize=(8, 6))
        if categories is not None:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=categories,
                style=categories,
                s=100,
                ax=ax
            )
            ax.legend(title="Category")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Attach figure to session state
        session.pending_figure = fig

        coords_list = coords.tolist()
        artifact_key = f"reduced_coords_{method}"
        session.artifacts[artifact_key] = coords_list

        result_id = session.next_result_id("reduction")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_list
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state
        }, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """
        One-hot encodes the document category labels in session.dataframe.

        Returns:
            A summary dictionary containing result_id, classes, and one_hot_labels.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "category_name" in df.columns:
            labels = df["category_name"].tolist()
        elif "category" in df.columns:
            labels = df["category"].tolist()
        else:
            return {"error": "No category column found in session dataframe."}

        lb = LabelBinarizer()
        one_hot = lb.fit_transform(labels)
        classes = [str(c) for c in lb.classes_]

        one_hot_list = one_hot.tolist()
        session.artifacts["binarized_labels"] = one_hot_list

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "one_hot_labels": one_hot_list,
            "n_samples": len(one_hot_list)
        }
        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
