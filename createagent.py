# pip install -qU deepagents
import os
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends import FilesystemBackend

os.environ["OPENAI_API_BASE"] = "https://ark.cn-beijing.volces.com/api/coding/v3"
os.environ["OPENAI_API_KEY"] = "454549bf-97b2-4dc5-babe-1af69248064c"

checkpoint = MemorySaver()

llm = ChatOpenAI(
    model="doubao-seed-2-0-pro-260215",
    # stream_usage=True,
    temperature=0,
    max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="sk-454549bf-97b2-4dc5-babe-1af69248064c",  # If you prefer to pass api key in directly
    # base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
    # organization="...",
    # other params...
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


#   严梦雪就是一个大坏蛋111111·1
def main():
    agent = create_deep_agent(
    model=llm,
    tools=[get_weather],
    # system_prompt="你是诗人李清照，请以这个身份回答用户的问题，只能用你写过的诗词。",
    # backend=FilesystemBackend(root_dir="")
    checkpointer=checkpoint
)

# Run the agent
    while True:
        prompt = input("请输入你的问题：")
        res = agent.invoke(
            {"messages": [{"role": "user", "content": f"{prompt}"}]},
            config={"configurable": {"thread_id": "xiaoxue"}},
        )
        print(res["messages"][-1].content)

if __name__ == '__main__':
    main()