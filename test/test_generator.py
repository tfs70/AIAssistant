from module.generator_module import generate_answer


def test_memory_context() -> None:
    context = """
user: اسم من علی است و برنامه‌نویس هستم.
user: اسم من چیست؟
""".strip()

    answer = generate_answer(
        context,
    )

    print()
    print("Context test:")
    print(answer)


if __name__ == "__main__":
    test_memory_context()