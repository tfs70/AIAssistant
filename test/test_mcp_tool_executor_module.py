
import pytest

from module.mcp_tool_executor_module import (
    _build_tool_catalog,
    _validate_tool_call,
)


class FakeTool:
    def __init__(self, name, input_schema):
        self.name = name
        self.input_schema = input_schema


def make_tool():
    return FakeTool(
        name="example_tool",
        input_schema={
            "type": "object",
            "properties": {
                "topCount": {"type": "integer"},
                "vendorNames": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["topCount"],
        },
    )


def test_catalog_is_built_dynamically():
    tool = make_tool()

    catalog = _build_tool_catalog([tool])

    assert list(catalog) == ["example_tool"]
    assert catalog["example_tool"] is tool


def test_catalog_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        _build_tool_catalog([make_tool(), make_tool()])


def test_valid_arguments_are_accepted():
    tool = make_tool()
    catalog = _build_tool_catalog([tool])

    name, arguments = _validate_tool_call(
        "example_tool",
        {"topCount": 10, "vendorNames": ["تأمین‌کننده الف"]},
        catalog,
    )

    assert name == "example_tool"
    assert arguments["topCount"] == 10


def test_unknown_tool_is_rejected():
    catalog = _build_tool_catalog([make_tool()])

    with pytest.raises(ValueError, match="not available"):
        _validate_tool_call("unknown_tool", {}, catalog)


def test_unknown_argument_is_rejected():
    catalog = _build_tool_catalog([make_tool()])

    with pytest.raises(ValueError, match="Unknown arguments"):
        _validate_tool_call(
            "example_tool",
            {"topCount": 10, "unexpected": True},
            catalog,
        )


def test_missing_required_argument_is_rejected():
    catalog = _build_tool_catalog([make_tool()])

    with pytest.raises(ValueError, match="Missing required"):
        _validate_tool_call(
            "example_tool",
            {},
            catalog,
        )


def test_wrong_argument_type_is_rejected():
    catalog = _build_tool_catalog([make_tool()])

    with pytest.raises(ValueError, match="Invalid type"):
        _validate_tool_call(
            "example_tool",
            {"topCount": "ten"},
            catalog,
        )


def test_boolean_is_not_accepted_as_integer():
    catalog = _build_tool_catalog([make_tool()])

    with pytest.raises(ValueError, match="Invalid type"):
        _validate_tool_call(
            "example_tool",
            {"topCount": True},
            catalog,
        )
