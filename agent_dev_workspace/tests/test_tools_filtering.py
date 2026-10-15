import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run with threshold=0.15 (Known answer #5)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(result["removed_terms"]) == {"always"}
    assert result["kept_terms_count"] == 4
    assert result["removed_terms_count"] == 1

    # Check full report variances stored in session
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]

    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375
    }

    for term, expected_var in expected_variances.items():
        actual_var = full_report.loc[full_report["term"] == term, "variance"].values[0]
        assert actual_var == pytest.approx(expected_var, abs=1e-3)


def test_pearson_filter_tool():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run for target_class="catB" (Known answer #6)
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    correlations = result["correlations"]

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }

    for term, expected_r in expected_pearson.items():
        assert correlations[term] == pytest.approx(expected_r, abs=1e-3)


def test_spearman_filter_tool():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run for target_class="catB" (Known answer #6)
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    correlations = result["correlations"]

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    for term, expected_rho in expected_spearman.items():
        assert correlations[term] == pytest.approx(expected_rho, abs=1e-3)
