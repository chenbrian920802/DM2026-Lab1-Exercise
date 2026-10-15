import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI converters and algorithms
from PAMI.extras.convert import DF2DB
import PAMI.frequentPattern.basic.FPGrowth as FPGrowthModule
import PAMI.frequentPattern.topk.FAE as FAEModule
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowthModule

def make_tools(session):
    """
    Returns the list of pattern-mining tools initialized with session context.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific category using PAMI algorithms.

        Args:
            category_name: The category to mine patterns for.
            filtering_method: Approach to filter vocabulary ('variance', 'tfidf', or 'term_frequency').
            algorithm: Algorithm to use ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count required for 'fpgrowth' and 'maxfpgrowth'.
            k: Number of top patterns to return when using 'topk'.

        Returns:
            A summary dictionary containing result_id, category_name, filtering_method,
            algorithm, and extracted patterns.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "category_name" not in df.columns and "category" not in df.columns:
            return {"error": "No category column found in session dataframe."}

        # Filter dataframe by category_name
        cat_col = "category_name" if "category_name" in df.columns else "category"
        cat_df = df[df[cat_col] == category_name]

        if len(cat_df) == 0:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].dropna().astype(str).tolist()
        if not texts:
            return {"error": f"No valid text found for category '{category_name}'."}

        # Step 1: Category-local CountVectorizer with English stopwords removed
        vec = CountVectorizer(stop_words="english")
        try:
            dtm = vec.fit_transform(texts)
        except ValueError:
            # e.g., empty vocabulary after stop words removal
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vec.get_feature_names_out())
        if dtm.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Vocabulary filtering per method
        dtm_dense = dtm.toarray()
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0)
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm)
            scores = np.mean(tfidf_matrix.toarray(), axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        surviving_indices = np.where(mask)[0]
        if len(surviving_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        surviving_terms = feature_names[surviving_indices]
        filtered_dtm = dtm_dense[:, surviving_indices]

        # Step 3: Convert to pandas DataFrame for PAMI DF2DB
        transaction_df = pd.DataFrame(filtered_dtm, columns=surviving_terms)
        
        # Sanitize column names and handle potential duplicate names resulting from non-ASCII stripping
        sanitized_cols = []
        counts = {}
        for col in transaction_df.columns:
            clean_col = str(col).encode('ascii', 'ignore').decode('ascii')
            if not clean_col:
                clean_col = "term"
            if clean_col in counts:
                counts[clean_col] += 1
                sanitized_cols.append(f"{clean_col}_{counts[clean_col]}")
            else:
                counts[clean_col] = 0
                sanitized_cols.append(clean_col)
        transaction_df.columns = sanitized_cols

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactional.db")
            output_file = os.path.join(tmpdir, "patterns.txt")

            # DF2DB converts binary/count df to transactional file
            converter = DF2DB.DF2DB(transaction_df)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            # Step 4: Mine patterns using PAMI algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowthModule.FPGrowth(db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                miner.save(output_file)
            elif algorithm == "topk":
                miner = FAEModule.FAE(db_file, k=k, sep="\t")
                miner.startMine()
                miner.save(output_file)
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowthModule.MaxFPGrowth(db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                miner.save(output_file)
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            patterns = []
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(":")
                        if len(parts) == 2:
                            itemset_str = parts[0].strip()
                            try:
                                sup = float(parts[1].strip())
                            except ValueError:
                                sup = parts[1].strip()
                            items = itemset_str.split("\t")
                            if len(items) == 1:
                                items = itemset_str.split()
                            patterns.append({"pattern": items, "support": sup})
                        else:
                            tokens = line.split("\t")
                            if len(tokens) >= 2:
                                try:
                                    sup = float(tokens[-1])
                                    items = tokens[:-1]
                                except ValueError:
                                    items = tokens
                                    sup = None
                                patterns.append({"pattern": items, "support": sup})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns),
            "patterns": patterns
        }
        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)

        return summary

    return [mine_patterns_tool]
