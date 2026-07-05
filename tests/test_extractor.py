from modelwatch.extractor import parse_json_object


def test_parse_json_object_accepts_fenced_ollama_text():
    payload = parse_json_object(
        """
        ```json
        {"model_name": "Qwen3 4B Instruct", "confidence": 0.9}
        ```
        """
    )

    assert payload["model_name"] == "Qwen3 4B Instruct"
    assert payload["confidence"] == 0.9
