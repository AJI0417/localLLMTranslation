# 套件載入
import chainlit as cl
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent


# 載入llama3.1模型
llama_Model = ChatOllama(
    model="gemma3:12b",
)

# 載入translategemma模型
Translate_Model = ChatOllama(model="translategemma:4b")


# 透過Lanchain建立Agent 並建立prompt
llama_Agent = create_agent(
    llama_Model,
    system_prompt="""你是一位專業的台灣旅遊顧問，擁有豐富的台灣各地美食、文化、景點的知識。
# 回答規則
1. 使用繁體中文(zh-TW)回答
2. 採用條列式格式，每個要點需包含：
   - 名稱/地點
   - 簡短描述(1-2句話)
   - 特色亮點
3. 每次回答提供 3-5個建議即可
4. 語氣親切友善，但保持專業
5. 如果不確定或超出專業範圍，請誠實說明
6. 回答範例:
    **(回答範例)**
    - 地點：
    - 特色：
    - 推薦：
請根據使用者的問題，提供類似格式的專業建議。""",
)


# chainlit UI
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="多國語系介紹台灣-聊天機器人🇹🇼✨").send()


@cl.on_message
async def on_message(message: cl.Message):
    # 將使用者訊息送給 agent
    response = await llama_Agent.ainvoke(
        {"messages": [{"role": "user", "content": message.content}]}
    )

    msg = response["messages"][-1].content  # 整理Agent response

    Translate_response = await Translate_Model.ainvoke(
        [
            {
                "role": "user",
                "content": f"將以下文字從繁體中文(zh-TW)翻譯成英文(en)。只輸出翻譯結果,不要任何解釋或額外文字:\n{msg}",
                # content這裡可以切換語言 ex:繁體中文(zh-TW)翻譯成日文(ja)
                # 英文代碼表請查閱 https://ollama.com/library/translategemma
            }
        ]
    )

    # 整理Translate_response
    translated_text = Translate_response.content

    # chainlit輸出
    await cl.Message(content=msg).send()
    await cl.Message(content=f"翻譯結果：\n {translated_text}").send()
