import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Ensure punkt tokenizer data is available if nltk is used
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

def make_tools(session):
    """
    Returns the list of data utility tools initialized with session context.
    """

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes document text into lowercased unigrams using nltk.word_tokenize.

        Returns:
            A summary dictionary containing result_id, total_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "'text' column not found in session dataframe."}

        tokenized_list = []
        for text in df["text"]:
            if pd.isna(text) or not str(text).strip():
                tokenized_list.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                tokenized_list.append(tokens)

        session.dataframe["unigrams"] = tokenized_list
        session.artifacts["unigrams"] = tokenized_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(tokenized_list),
            "sample_tokens": tokenized_list[:3] if len(tokenized_list) >= 3 else tokenized_list
        }
        session.store_result("tokenize_tool", {}, summary)

        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and subdirectories in the specified subdirectory relative to repo root.

        Args:
            subdirectory: Relative directory path from repo root (default: root directory "").

        Returns:
            A summary dictionary containing result_id, subdirectory, directories, and files.
        """
        # Resolve path relative to current working directory (repo root)
        target_dir = os.path.abspath(subdirectory) if subdirectory else os.getcwd()

        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"'{subdirectory}' is not a directory."}

        entries = os.listdir(target_dir)
        directories = [e for e in entries if os.path.isdir(os.path.join(target_dir, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(target_dir, e))]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": sorted(directories),
            "files": sorted(files)
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)

        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """
        Inspects the top n_rows of the working DataFrame in session.

        Args:
            n_rows: Number of rows to peek at (default: 5).

        Returns:
            A summary dictionary containing result_id, columns, total_rows, and head_records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        head_df = df.head(n_rows).fillna("")
        records = head_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "columns": list(df.columns),
            "n_rows": len(records),
            "head": records
        }
        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks for missing or empty text values in session.dataframe["text"].

        Returns:
            A summary dictionary containing result_id, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "'text' column not found in session dataframe."}

        missing_indices = []
        for idx, val in enumerate(df["text"]):
            if pd.isna(val) or not str(val).strip():
                missing_indices.append(idx)

        result_id = session.next_result_id("missing")
        summary = {
            "result_id": result_id,
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices
        }
        session.store_result("check_missing_tool", {}, summary)

        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        Args:
            drop: If True, drops duplicate rows using keep=False (removing all duplicate copies).

        Returns:
            A summary dictionary containing result_id, duplicate_count, dropped, and remaining_count.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Identify duplicates on 'text' column
        duplicate_mask = df.duplicated(subset=["text"], keep=False) & df["text"].notna() & (df["text"] != "")
        duplicate_count = df.duplicated(subset=["text"], keep="first").sum()

        if drop:
            # Match Master's drop_duplicates(keep=False, inplace=True) logic
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        remaining_count = len(session.dataframe)

        result_id = session.next_result_id("duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": int(duplicate_count),
            "dropped": drop,
            "remaining_count": remaining_count
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Subsamples n rows from session.dataframe.

        Args:
            n: Number of samples to draw (default: 5).
            random_state: Seed for random sampling reproducibility.

        Returns:
            A summary dictionary containing result_id, sampled_indices, and sample_records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        sampled_df = df.sample(n=min(n, len(df)), random_state=random_state)
        sampled_indices = sampled_df.index.tolist()
        records = sampled_df.fillna("").to_dict(orient="records")

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "sample_size": len(sampled_df),
            "sampled_indices": sampled_indices,
            "sample": records
        }
        session.store_result("sample_data_tool", {
            "n": n,
            "random_state": random_state
        }, summary)

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes text length statistics and generates a boxplot grouped by category.

        Returns:
            A summary dictionary containing overall and per-category text_length statistics.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Calculate character text length
        df["text_length"] = df["text"].apply(lambda t: len(str(t)) if pd.notna(t) else 0)

        overall_stats = df["text_length"].describe().to_dict()

        cat_col = None
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "category" in df.columns:
            cat_col = "category"

        by_category_stats = {}
        if cat_col:
            for cat, group in df.groupby(cat_col):
                by_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Build Seaborn boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        if cat_col:
            sns.boxplot(x=cat_col, y="text_length", data=df, ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)

        ax.set_ylabel("Text Length (chars)")
        ax.set_title("Document Text Length Distribution")
        fig.tight_layout()

        # Set figure in session state
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": by_category_stats
        }
        session.store_result("describe_data_tool", {}, summary)

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
