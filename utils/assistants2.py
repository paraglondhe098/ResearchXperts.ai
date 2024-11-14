from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Any, Optional, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ai21 import ChatAI21
from langchain.tools.retriever import create_retriever_tool
from utils.core import ResearchPaper, ArXivRetriever
from datetime import datetime


class PaperSearchAssistant:
    """Assistant that uses LLM to optimize search queries and fetch papers"""

    def __init__(self, llm: Any):
        self.llm = llm
        self.arxiv_retriever = ArXivRetriever()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research paper search assistant. Your task is to:
             1. Analyze the user's query
             2. Correct any spelling mistakes
             3. Format it into an optimal search query for ArXiv
             4. Return only the reformulated query without any additional text or explanations.

             Make sure to:
             - Preserve technical terms and acronyms
             - Use Boolean operators (AND, OR) when appropriate
             - Include relevant synonyms for important terms"""),
            ("user", "{query}")
        ])

    def search_papers(self, query: str,
                      max_results: int = 5) -> List[ResearchPaper]:
        """Search for papers on ArXiv and return List of ResearchPaper objects"""
        chain = self.prompt | self.llm
        cleaned_query = chain.invoke({"query": query}).content
        print(cleaned_query)
        papers = self.arxiv_retriever.fetch_papers(cleaned_query, max_results=max_results)
        return papers


class PaperAssistant:
    """
    A class for reading PDFs and answering questions using RAG (Retrieval Augmented Generation).
    """

    def __init__(
            self,
            paper: ResearchPaper,
            llm: Any,
            embedding_model: Optional[Any] = None,
            k: int = 10,
            cache_dir: Optional[str] = None,
            username: str = "abc"
    ):
        """
        Initialize the PDF reader with a language model and document source.
        """
        self.paper = paper
        self.llm = llm
        self.embedding_model = embedding_model
        self.k = k
        self.cache_dir = cache_dir
        self.memory = MemorySaver()
        self.docs: List[Document] = []
        self.config = {"configurable": {"thread_id": username}}
        self.system_prompt = f"""You are Dr. Inform, a research paper assistant.
        - Your primary role is to retrieve precise, relevant content from the research paper using the tool `research_paper_search`.
        - Avoid tool usage only if metadata alone can provide a complete answer or if the question is unrelated.
        Current Paper:
        Title: "{paper.title}"
        Category: {paper.category}
        Authors: {', '.join(paper.authors)}
        Published: {paper.published.strftime('%Y-%m-%d')}
        Journal Ref: {paper.journal_ref if paper.journal_ref else 'N/A'}
        """

        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """
        Initialize the vector store with document embeddings.
        """
        try:
            self.docs = self._pdf_to_documents(self.paper.pdf_url)
            if not self.docs:
                raise ValueError("No content extracted from PDF")
            if not self.embedding_model:
                self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = InMemoryVectorStore.from_documents(documents=self.docs, embedding=self.embedding_model)
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
            self.tool = create_retriever_tool(
                self.retriever,
                "research_paper_search",
                f"Use this tool to search the research paper titled {self.paper.title} for relevant information. This tool should be your primary "
                "method for answering questions about the paper's content, methodology, results, and conclusions. Only skip "
                "using this tool if the question is clearly about metadata (like page count) or is completely unrelated to "
                "the paper's content."
            )
            self.agent_executor = create_react_agent(self.llm, tools=[self.tool], checkpointer=self.memory,
                                                     state_modifier=self.system_prompt)

        except Exception as e:
            raise ValueError(f"Failed to initialize vector store: {str(e)}")

    @staticmethod
    def _pdf_to_documents(url) -> List[Document]:
        loader = PyPDFLoader(url)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        return pages

    def answer(self, question: str) -> Optional[dict]:
        try:
            response = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=self.config)
            tool_used = response['messages'][-2].type == 'tool'
            res = {"answer": (response['messages'][-1].content, tool_used),
                   "tool_used": tool_used}
            if tool_used:
                res["tool_message"] = response['messages'][-2]
            return res
        except Exception as e:
            raise ValueError(f"Failed to process question: {str(e)}")

    def pages(self) -> int:
        """Return the number of pages in the pdf file."""
        return len(self.docs)

    def get_average_chunk_size(self) -> float:
        """Return the average chunk size in characters."""
        if not self.docs:
            return 0
        return sum(len(doc.page_content) for doc in self.docs) / len(self.docs)

    def get_paper_stats(self) -> dict:
        """
        Get comprehensive statistics about the paper.

        Returns:
            Dictionary containing paper statistics and metadata
        """
        word_counts = [len(doc.page_content.split()) for doc in self.docs]
        return {
            "paper_info": {
                "title": self.paper.title,
                "authors": self.paper.authors,
                "category": self.paper.category,
                "published_date": self.paper.published.strftime('%Y-%m-%d'),
                "journal_ref": self.paper.journal_ref
            },
            "content_stats": {
                "total_pages": self.pages(),
                "total_words": sum(word_counts),
                "average_words_per_page": round(sum(word_counts) / len(word_counts) if word_counts else 0, 2),
                "average_chunk_size": round(self.get_average_chunk_size(), 2)
            },
            "processing_info": {
                "embedding_model": self.embedding_model,
                "llm": self.llm,
                "retrieval_chunks": self.k,
                "vectorstore_documents": len(self.docs)
            }
        }
