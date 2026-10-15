import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_cosine_similarity_tool():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 assertions
    # 0 vs 1 -> 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert pytest.approx(res_0_1["similarity_score"], abs=1e-3) == 0.9847

    # 0 vs 3 -> 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert pytest.approx(res_0_3["similarity_score"], abs=1e-3) == 0.0909

    # 3 vs 4 -> 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert pytest.approx(res_3_4["similarity_score"], abs=1e-3) == 0.9847

    # 0 vs 7 (empty row) -> 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert pytest.approx(res_0_7["similarity_score"], abs=1e-3) == 0.0

def test_feature_correlation_matrix_tool():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({"top_n": 20})

    terms = res.get("terms", [])
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert terms == expected_terms

    corr_matrix = np.array(res.get("correlation_matrix", []))

    # Known Answer #19 assertions
    # alpha vs beta (indices 0 and 3) -> 0.8885
    np.testing.assert_allclose(corr_matrix[0, 3], 0.8885, atol=1e-3)

    # gamma vs delta (indices 1 and 2) -> 0.8307
    np.testing.assert_allclose(corr_matrix[1, 2], 0.8307, atol=1e-3)

    # alpha vs gamma (indices 0 and 1) -> -0.5718
    np.testing.assert_allclose(corr_matrix[0, 1], -0.5718, atol=1e-3)

    # Verify pending figure was created
    assert session.pending_figure is not None
