from cafl.utils.schema import schema_spec_to_json_schema, validate_output_text


def test_shorthand_schema_preserves_nested_json_schema_enum():
    schema = {
        "answer": {"type": "string", "enum": ["True", "False"]},
        "rationale": "a string",
    }

    json_schema = schema_spec_to_json_schema(schema)

    assert json_schema["properties"]["answer"] == {"type": "string", "enum": ["True", "False"]}
    assert validate_output_text('{"answer": "True", "rationale": "x"}', schema) == (
        {"answer": "True", "rationale": "x"},
        None,
    )
    parsed, error = validate_output_text('{"answer": "Maybe", "rationale": "x"}', schema)
    assert parsed == {"answer": "Maybe", "rationale": "x"}
    assert error == "'Maybe' is not one of ['True', 'False']"


def test_full_json_schema_can_require_boolean_answer():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "boolean"}},
        "required": ["answer"],
    }

    assert validate_output_text('{"answer": true}', schema) == ({"answer": True}, None)
    parsed, error = validate_output_text('{"answer": "True"}', schema)
    assert parsed == {"answer": "True"}
    assert error == "'True' is not of type 'boolean'"
