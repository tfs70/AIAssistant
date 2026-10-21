
import json

from module.mcp_tool_executor_module import route_and_execute_question


QUESTIONS = [
    "سلام، حالت چطوره؟",
    "کدام کالاها واقعاً بیشترین خرید را داشته‌اند؟",
    "کدام کالاها بیشترین سفارش خرید را داشته‌اند؟",
]


def main() -> None:
    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Question: {question}")

        try:
            result = route_and_execute_question(question)

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
