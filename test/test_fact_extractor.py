from module.fact_extractor_module import extract_facts


def test_fact_extractor() -> None:
    """تست استخراج Fact"""

    user_id: str = "user_001"

    message: str = "اسم من علی است و برنامه‌نویس هستم."

    facts = extract_facts(
        user_id=user_id,
        message=message,
    )

    print()
    print("===== FACTS =====")

    for fact in facts:
        print(f"{fact.key}: {fact.value}")

    print("=================")

    assert len(facts) >= 2

    fact_values = {
        fact.key: fact.value
        for fact in facts
    }

    assert fact_values["name"] == "علی"
    assert fact_values["job"] == "برنامه‌نویس"

    print()
    print("test_fact_extractor: PASSED")


if __name__ == "__main__":
    test_fact_extractor()