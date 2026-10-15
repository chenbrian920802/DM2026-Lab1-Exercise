import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    """
    Returns the list of exploration tools initialized with session context.
    """

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> Dict[str, Any]:
        """
        Computes the cosine similarity between two documents in session.feature_matrix.

        Args:
            doc_index_1: Index of the first document (0-based row index).
            doc_index_2: Index of the second document (0-based row index).

        Returns:
            A summary dictionary containing result_id, doc_index_1, doc_index_2,
            and similarity_score.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix loaded in session. Build DTM first."}

        X = session.feature_matrix
        n_docs = X.shape[0]

        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"doc_index_1 ({doc_index_1}) out of range [0, {n_docs-1}]."}
        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"doc_index_2 ({doc_index_2}) out of range [0, {n_docs-1}]."}

        doc1_vec = X[doc_index_1]
        doc2_vec = X[doc_index_2]

        # Calculate cosine similarity using scikit-learn
        sim_matrix = cosine_similarity(doc1_vec, doc2_vec)
        score = float(sim_matrix[0, 0])

        # Handle NaN if vector is all zeros (e.g. empty text document)
        if np.isnan(score):
            score = 0.0

        result_id = session.next_result_id("similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity_score": score
        }
        session.store_result("cosine_similarity_tool", {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2
        }, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Computes Pearson correlation matrix for the top N terms by variance in session.feature_matrix,
        and generates an annotated heatmap set to session.pending_figure.

        Args:
            top_n: Number of top variance terms to select (default: 20).

        Returns:
            A summary dictionary containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names loaded in session. Build DTM first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.array(session.feature_matrix)
        feature_names = np.array(session.feature_names)

        if X.shape[1] == 0:
            return {"error": "Feature matrix has 0 columns."}

        # Compute term variances across all documents
        variances = np.var(X, axis=0)
        
        # Take top_n terms by variance (descending order)
        n_select = min(top_n, X.shape[1])
        top_indices = np.argsort(variances)[::-1][:n_select]

        top_terms = feature_names[top_indices].tolist()
        top_X = X[:, top_indices]

        # Compute Pearson correlation matrix across term columns
        corr_matrix = np.corrcoef(top_X, rowvar=False)
        # Handle 1x1 case or NaNs if any column has zero variance
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        corr_list = corr_matrix.tolist()

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {n_select} Terms by Variance)")
        fig.tight_layout()

        # Attach figure to session state
        session.pending_figure = fig

        result_id = session.next_result_id("correlation")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list
        }
        session.store_result("feature_correlation_matrix_tool", {
            "top_n": top_n
        }, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
