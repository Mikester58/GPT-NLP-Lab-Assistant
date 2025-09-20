from ai_sdk.openai import openai
from mastra.core.agent import Agent

# To create a requirements file:
# pip freeze > requirements.txt

my_agent = Agent(
    name="My Agent",
    instructions="You are a helpful assistant.",
    model=openai("gpt-4o-mini")
)

