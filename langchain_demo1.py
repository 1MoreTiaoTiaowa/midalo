from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# Define system prompt
SYSTEM_PROMPT = """你是一个风趣幽默的专业天气播报员。

你拥有两种工具:

- get_weather_for_location: 可以帮助你获取特定地点下的天气情况
- get_user_location: 可以帮助你获取用户的所在地点

如果用户询问你关于天气的问题，
请确保你知道指定的地点所在。
如果你能通过问题知道他们的所在地点，请使用get_user_location工具去找到他们的位置"""


# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}一直都是大晴天！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "大前门" if user_id == "1" else "新白沙"


# Configure model
model = init_chat_model(
    "ollama:llama3.2",
    temperature=0
)


# Define response format
@dataclass
class Response_Format:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=Response_Format,
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气如何呀？"}]},
    config=config,
    context=Context(user_id="3")
)

print(response)
# print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你知道我在哪里吗？"}]},
    config=config,
    context=Context(user_id="3")
)
print(response)
print(response['punny_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
