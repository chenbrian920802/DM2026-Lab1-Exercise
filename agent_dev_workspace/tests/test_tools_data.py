import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools

def setup_session_with_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_tokenize_tool():
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert res.get("total_documents") == 8

    # Check session unigrams artifact against Known Answer #10
    unigrams = session.artifacts.get("unigrams", [])
    assert len(unigrams) == 8
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[1] == ["always", "alpha", "alpha", "beta"]
    assert unigrams[2] == ["always", "alpha", "beta"]
    assert unigrams[3] == ["always", "gamma", "gamma", "gamma", "delta"]
    assert unigrams[4] == ["always", "gamma", "gamma", "delta"]
    assert unigrams[5] == ["always", "gamma", "delta"]
    assert unigrams[6] == ["always", "gamma", "delta"]
    assert unigrams[7] == []

def test_list_files_tool():
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    # Known Answer #11 assertion
    assert res.get("directories") == []
    assert res.get("files") == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool():
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 5})

    assert res.get("total_rows") == 8
    assert "text" in res.get("columns", [])
    assert res.get("n_rows") == 5

def test_check_missing_tool():
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known Answer #2 assertion
    assert res.get("missing_count") == 1
    assert res.get("missing_indices") == [7]

def test_check_duplicates_tool():
    # 1. Test check without drop
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check.get("duplicate_count") == 1
    assert res_check.get("remaining_count") == 8

    # 2. Test check with drop=True (keep=False drops all copies -> 6 rows remaining)
    session_drop = setup_session_with_data()
    data_tools_drop = make_data_tools(session_drop)
    check_duplicates_drop = next(t for t in data_tools_drop if t.name == "check_duplicates_tool")

    res_drop = check_duplicates_drop.invoke({"drop": True})
    assert res_drop.get("remaining_count") == 6

def test_sample_data_tool():
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Known Answer #13 assertion
    assert res.get("sampled_indices") == [1, 5, 0, 7]

def test_describe_data_tool():
    session = setup_session_with_data()
    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Known Answer #18 assertions
    overall = res.get("overall", {})
    assert overall.get("count") == 8
    np.testing.assert_allclose(overall.get("mean"), 19.875, atol=1e-3)
    np.testing.assert_allclose(overall.get("std"), 9.4330, atol=1e-3)
    assert overall.get("min") == 0
    assert overall.get("max") == 30

    by_cat = res.get("by_category", {})
    np.testing.assert_allclose(by_cat.get("catA", {}).get("mean"), 17.25, atol=1e-3)
    np.testing.assert_allclose(by_cat.get("catB", {}).get("mean"), 22.50, atol=1e-3)

    # Verify pending figure was created
    assert session.pending_figure is not None
