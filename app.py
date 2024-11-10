import streamlit as st
from api_keys import fetch_api_key
import feedparser, requests
from langchain.agents import Tool
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import os


class ArXivRetriever:
    def __init__(self, api_url="http://export.arxiv.org/api/query?"):
        self.api_url = api_url

    def fetch_papers(self, query, max_results=5):
        query_url = f"{self.api_url}search_query=all:{query}&max_results={max_results}"
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            papers = []
            for entry in feed.entries:
                paper_id = entry.id.split("/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                paper = {
                    "id": paper_id,
                    "title": entry.title,
                    "category": entry.category,
                    "authors": [author.name for author in entry.authors],
                    "author_comment": entry.get("arxiv_comment", "No comments available"),
                    "published": entry.published,
                    "summary": entry.summary,
                    "link": entry.link,
                    "pdf_url": pdf_url,
                    "journal_ref": entry.get("arxiv_journal_ref", "No journal reference")
                }
                papers.append(paper)
            return papers
        except requests.exceptions.RequestException as e:
            st.error("Error fetching papers. Please try again later.")
            return []


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

    def response(self, prompt):
        res = self.agent_executor.invoke({
            "messages": [HumanMessage(prompt)],
        }, self.config)
        return res['messages'][-1]

    @staticmethod
    def configure_environment():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = fetch_api_key("langsmith", requires_pass=False)
        os.environ["MISTRAL_API_KEY"] = fetch_api_key("mistral", requires_pass=False)


def chat(model):
    st.title("Research Assistant Chatbot")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your message:", "")

    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")

        bot_response = model.response(user_input)
        st.session_state.chat_history.append(f"Bot: {bot_response}")

    for message in st.session_state.chat_history:
        st.write(message)


if __name__ == "__main__":
    model = Model("Parag")
    chat(model)
