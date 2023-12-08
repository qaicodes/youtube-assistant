from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from dotenv import load_dotenv

load_dotenv()

def langchain_agents():
    llm = OpenAI(temperature= 0.5)
    tools = load_tools(['wikipedia'], llm = llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.run("How many poeple died on Mount Everest till now?")
    print(result)


if __name__ == '__main__':
    print(langchain_agents())
