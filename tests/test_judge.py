from modelwatch.judge import decision_from_payload


def test_decision_from_payload_normalizes_keep_and_reason():
    decision = decision_from_payload(
        {
            "keep": "yes",
            "reason": "Official foundation model",
            "confidence": "high",
        }
    )

    assert decision.keep is True
    assert decision.reason == "Official foundation model"
    assert decision.confidence == 0.85


def test_decision_from_payload_rejects_by_default():
    decision = decision_from_payload({})

    assert decision.keep is False
    assert decision.reason == "No benchmark-worthy release signal"
