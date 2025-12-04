import asyncio
from llama_index.llms.openai_like import OpenAILike
from droidrun import DroidAgent, AdbTools
import os
import openai
from openai import OpenAI
import pandas as pd

async def single_app():
    tools = AdbTools(use_tcp=True)
    # tools.start_app("org.tasks")


    with open("key",'r') as f:
        forge_key=f.read().strip()
        llm = OpenAILike(
        model="gpt-5",  # or "gpt-4o", "gpt-4", etc.
        api_base="https://api.openai.com/v",  # For local endpoints
        is_chat_model=True, # droidrun requires chat model support
        api_key=forge_key,
    )
    agent = DroidAgent(
                goal='Go to the privacy policy page of the app org.tasks',
                llm=llm,
                tools=tools,
                vision=False,        # Set to True if your model supports vision
                # reasoning=True,     # Optional: enable planning/reasoning
                save_trajectories="step",
            )

    # Run the agent
    result = await agent.run()
    print(f"Success: {result['success']}")
    if result.get('output'):
        print(f"Output: {result['output']}")





if __name__ == "__main__":
    asyncio.run(single_app())