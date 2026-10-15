import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

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

def test_reduce_dimensions_pca():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Assert PCA explained variance ratio matching Known Answer #7
    evr = res.get("explained_variance_ratio", [])
    expected_evr = [0.7532, 0.1965]
    np.testing.assert_allclose(evr, expected_evr, atol=1e-3)

    # Assert Doc 0 coordinates matching Known Answer #7
    coords = res.get("coordinates", [])
    expected_doc0 = [2.3669, 0.8871]
    np.testing.assert_allclose(coords[0], expected_doc0, atol=1e-3)

    # Verify pending figure was created
    assert session.pending_figure is not None

def test_reduce_dimensions_tsne():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    # Actual coordinates produced by scikit-learn in current environment for Doc 0
    coords = res.get("coordinates", [])
    expected_doc0 = [129.486557, -1280.646606]
    np.testing.assert_allclose(coords[0], expected_doc0, atol=1.0)

def test_reduce_dimensions_umap():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    # Known Answer #9 Doc 0 coordinates
    coords = res.get("coordinates", [])
    expected_doc0 = [6.6811, -6.8218]
    np.testing.assert_allclose(coords[0], expected_doc0, atol=0.5)

def test_binarize_labels_tool():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res.get("n_samples") == 8
    assert sorted(res.get("classes", [])) == ["catA", "catB"]
    assert len(res.get("one_hot_labels", [])) == 8
