import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # Each test function creates its own fresh session
    session = SessionState()

    # Load premade load_dataset_tool to load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert known answer #4 from TEST_FIXTURE.md
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["n_terms"] == 5
    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # Each test function creates its own fresh session
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # First build the DTM
    build_dtm_tool.invoke({})

    # Now run term_frequency_tool
    result = term_frequency_tool.invoke({})

    tf = result["term_frequencies"]

    # Assert known answer #14 in TEST_FIXTURE.md:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    assert tf["alpha"] == 6
    assert tf["always"] == 7
    assert tf["beta"] == 3
    assert tf["delta"] == 4
    assert tf["gamma"] == 7


def test_dtm_heatmap_tool():
    # Each test function creates its own fresh session
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Test full matrix slice (known answer #21)
    result = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert result["matrix"] == expected_matrix
    assert session.pending_figure is not None

    # Test partial slice (top-left 3x2 corner)
    partial_result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert partial_result["matrix"] == [[3, 1], [2, 1], [1, 1]]
