from typing import Optional, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from the current session dataframe.

        Fits CountVectorizer on the 'text' column of session.dataframe and builds
        the feature matrix, feature names, and category labels. Stores the count
        vectorizer artifact in session.artifacts['count_vectorizer'] and reports DTM
        sparsity statistics.

        Args:
            max_features: Maximum number of terms to include in the vocabulary.
                Defaults to None (unlimited).
            ngram_range: Lower and upper boundary of range of n-values for
                n-grams to be extracted, e.g. [1, 1] for unigrams, [1, 2] for
                unigrams and bigrams. Defaults to [1, 1].

        Returns:
            Dict containing summary statistics of the DTM including n_documents,
            n_terms, non_zero, total_elements, sparsity_pct, and result_id.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column. Load a dataset first.")

        # Convert list to tuple for CountVectorizer
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1])) if ngram_range and len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_tuple)
        X = vectorizer.fit_transform(session.dataframe["text"].fillna(""))

        feature_names = list(vectorizer.get_feature_names_out())

        # Store in session state
        session.feature_matrix = X
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Store vectorizer artifact for downstream tools like cosine_similarity_tool
        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={"max_features": max_features, "ngram_range": ngram_range},
            summary=summary
        )

        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the document-term matrix.

        Sums term counts across all documents in session.feature_matrix and returns
        the term frequency distribution, optionally restricted to the top N terms.

        Args:
            top_n: Optional integer specifying the top N most frequent terms to return.
                If None, returns frequencies for all terms in vocabulary.

        Returns:
            Dict containing result_id, total_terms, and term_frequencies dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        # Sum counts across all documents (columns of DTM)
        counts = session.feature_matrix.sum(axis=0).A1  # Convert to 1D array

        term_counts = list(zip(session.feature_names, counts))
        # Sort terms by count in descending order, secondary sort alphabetically by term name
        term_counts.sort(key=lambda x: (-x[1], x[0]))

        if top_n is not None and top_n > 0:
            term_counts = term_counts[:top_n]

        term_frequencies = {term: int(count) for term, count in term_counts}

        result_id = session.next_result_id("tf")

        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "returned_terms": len(term_frequencies),
            "term_frequencies": term_frequencies
        }

        report_df = pd.DataFrame(term_counts, columns=["term", "frequency"])

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generate a heatmap visualization of a raw positional slice of the document-term matrix.

        Slices the top n_documents rows and n_terms columns from session.feature_matrix,
        renders an annotated heatmap plot, attaches it to session.pending_figure, and
        returns the matrix slice and labels.

        Args:
            n_terms: Number of terms (columns) to include from the start of vocabulary.
                Defaults to 20.
            n_documents: Number of documents (rows) to include from the start of corpus.
                Defaults to 20.

        Returns:
            Dict containing result_id, n_documents, n_terms, terms, and matrix slice list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        actual_docs = min(n_documents, session.feature_matrix.shape[0])
        actual_terms = min(n_terms, session.feature_matrix.shape[1])

        # Slice submatrix and convert to dense 2D list
        submatrix = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        sliced_terms = session.feature_names[0:actual_terms]

        # Generate plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            submatrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in range(actual_docs)],
            ax=ax
        )
        ax.set_title(f"Document-Term Matrix Heatmap (Slice: {actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        matrix_list = [[int(val) for val in row] for row in submatrix]

        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "matrix": matrix_list
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
