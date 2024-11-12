from utils.api_keys import fetch_api_key
from utils.retriever import ArXivRetriever
from langchain.agents import Tool
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os


class Model:
    def __init__(self, username):
        self.configure_environment()
        self.llm = ChatMistralAI(model="mistral-large-2402")
        self.retriever = ArXivRetriever()
        self.search_tool = Tool(
            name="ArXiv_Retriever",
            func=self.retriever.fetch_papers,
            description="Use this tool to search for research papers related to a topic."
        )
        self.memory = MemorySaver()
        self.agent_executor = create_react_agent(self.llm, [self.search_tool], checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": username}}
        self.parser = StrOutputParser()

    def response(self, prompt):
        res = self.agent_executor.invoke({
            "messages": [HumanMessage(prompt)],
        }, self.config)
        return self.parser.invoke(input=res['messages'][-1], config=self.config)

    @staticmethod
    def configure_environment():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = fetch_api_key("langsmith", requires_pass=False)
        os.environ["MISTRAL_API_KEY"] = fetch_api_key("mistral", requires_pass=False)