from module.generator_module import generate_answer

query: str = "سلام، خودت را معرفی کن."

answer: str = generate_answer(
    query=query,
)

print(answer)