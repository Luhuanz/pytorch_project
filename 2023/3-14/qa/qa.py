from kg_qa import KGQA


if __name__ == "__main__":
    handler = KGQA()
    while True:
        question = input("用户：")
        if not question:
            break
        answer = handler.answer(question)
        print("回答：", answer)
        print("*"*50)