from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Flag near-constant terms by variance for feature selection / filtering.

        Calculates the variance of each term's count across all documents in
        session.feature_matrix and separates terms into kept and removed lists
        based on the specified threshold.

        Args:
            threshold: Minimum variance required for a term to be kept.
                Defaults to 0.0.

        Returns:
            Dict containing result_id, threshold, total_terms, kept_terms_count,
            removed_terms_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        X = session.feature_matrix

        # Calculate mean and mean of squares across rows (axis=0) for sparse matrix
        mean = np.array(X.mean(axis=0)).ravel()
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        terms = session.feature_names

        # Build full report DataFrame (sorted descending by variance)
        # Required format: 1st column = term, 2nd column = variance
        df = pd.DataFrame({"term": terms, "variance": variances})
        df_sorted = df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = df[df["variance"] >= threshold]
        removed_df = df[df["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(terms),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_sorted
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate Pearson correlation between term frequencies and a target class.

        Binarizes document labels against target_class (1 if matching, 0 otherwise)
        and computes the Pearson linear correlation coefficient for each term in
        session.feature_matrix.

        Args:
            target_class: The category name to compute correlation against (e.g., 'catB').
            threshold: Minimum absolute correlation required to keep a term.
                Defaults to 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms,
            correlations dict, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or lacks 'category_name'. Load a dataset first.")

        # Create binary target vector (1 for target_class, 0 otherwise)
        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        std_y = np.std(y)
        if std_y == 0:
            raise ValueError(f"Target class '{target_class}' has zero variance (all documents or no documents match).")

        X = session.feature_matrix.toarray()
        mean_y = np.mean(y)

        pearson_rs = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            std_xj = np.std(xj)
            if std_xj == 0:
                pearson_rs.append(0.0)
            else:
                cov = np.mean((xj - np.mean(xj)) * (y - mean_y))
                r = cov / (std_xj * std_y)
                pearson_rs.append(float(r))

        terms = session.feature_names

        # Required format for full_report: 1st column = term, 2nd column = metric name (pearson_r)
        report_df = pd.DataFrame({"term": terms, "pearson_r": pearson_rs})
        report_df_sorted = report_df.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        correlations = {term: round(r, 4) for term, r in zip(terms, pearson_rs)}

        kept_terms = [term for term, r in zip(terms, pearson_rs) if abs(r) >= threshold]
        removed_terms = [term for term, r in zip(terms, pearson_rs) if abs(r) < threshold]

        result_id = session.next_result_id("pearson")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "correlations": correlations,
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df_sorted
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate Spearman rank correlation between term frequencies and a target class.

        Binarizes document labels against target_class (1 if matching, 0 otherwise)
        and computes the Spearman monotonic rank correlation coefficient for each term in
        session.feature_matrix.

        Args:
            target_class: The category name to compute correlation against (e.g., 'catB').
            threshold: Minimum absolute correlation required to keep a term.
                Defaults to 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms,
            correlations dict, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or lacks 'category_name'. Load a dataset first.")

        # Create binary target vector (1 for target_class, 0 otherwise)
        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        if np.std(y) == 0:
            raise ValueError(f"Target class '{target_class}' has zero variance (all documents or no documents match).")

        X = session.feature_matrix.toarray()

        spearman_rhos = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            if np.std(xj) == 0:
                spearman_rhos.append(0.0)
            else:
                res = spearmanr(xj, y)
                rho = res.statistic if hasattr(res, "statistic") else res[0]
                spearman_rhos.append(0.0 if np.isnan(rho) else float(rho))

        terms = session.feature_names

        # Required format for full_report: 1st column = term, 2nd column = metric name (spearman_r)
        report_df = pd.DataFrame({"term": terms, "spearman_r": spearman_rhos})
        report_df_sorted = report_df.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        correlations = {term: round(rho, 4) for term, rho in zip(terms, spearman_rhos)}

        kept_terms = [term for term, rho in zip(terms, spearman_rhos) if abs(rho) >= threshold]
        removed_terms = [term for term, rho in zip(terms, spearman_rhos) if abs(rho) < threshold]

        result_id = session.next_result_id("spearman")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "correlations": correlations,
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df_sorted
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
