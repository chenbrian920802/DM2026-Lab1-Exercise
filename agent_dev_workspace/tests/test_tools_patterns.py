import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_known_answers():
    # Set up session
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's mine_patterns_tool
    student_tools = make_tools(session)
    mine_patterns_tool = next(t for t in student_tools if t.name == "mine_patterns_tool")

    # 1. Test filtering_method="variance" -> 0 terms survive
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth"
    })
    assert res_var.get("patterns") == []
    assert res_var.get("note") == "no terms survived filtering"

    # 2. Known Answer #17: filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catA_fp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_catA = res_catA_fp.get("patterns", [])
    assert len(patterns_catA) == 1
    assert patterns_catA[0]["pattern"] == ["alpha"]
    assert pytest.approx(patterns_catA[0]["support"]) == 3

    res_catB_fp = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_catB = res_catB_fp.get("patterns", [])
    assert len(patterns_catB) == 1
    assert patterns_catB[0]["pattern"] == ["gamma"]
    assert pytest.approx(patterns_catB[0]["support"]) == 4

    # 3. Known Answer #20: algorithm="topk" (k=1)
    res_catA_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_topk_a = res_catA_topk.get("patterns", [])
    assert len(patterns_topk_a) == 1
    assert patterns_topk_a[0]["pattern"] == ["alpha"]
    assert pytest.approx(patterns_topk_a[0]["support"]) == 3

    # 4. Known Answer #20: algorithm="maxfpgrowth" -> returns 0 patterns
    res_catA_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catA_max.get("patterns") == []
