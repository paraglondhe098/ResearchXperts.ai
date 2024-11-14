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
from utils.core import ResearchPaper, ArXivRetriever
from datetime import datetime
from utils.prompts import get_querry_search_prompt, get_query_optimization_prompt, get_response_gen_prompt, \
    get_response_gen_prompt_casual


class QueryHistory:
    """Manages history of previously asked questions for query optimization"""

    def __init__(self, max_questions: int = 5):
        """
        Initialize query history tracker.

        Args:
            max_questions (int): Maximum number of previous questions to store
        """
        if not isinstance(max_questions, int) or max_questions <= 0:
            raise ValueError("max_questions must be a positive integer")

        self.questions: List[str] = []
        self.max_questions = max_questions

    def add_question(self, question: str) -> None:
        """
        Add a new question to the history.

        Args:
            question (str): The question to add
        """
        if not isinstance(question, str) or not question.strip():
            return

        self.questions.append(question.strip())

        # Keep only the most recent questions
        if len(self.questions) > self.max_questions:
            self.questions = self.questions[-self.max_questions:]

    def get_history(self) -> str:
        """
        Get formatted question history.

        Returns:
            str: Numbered list of previous questions
        """
        if not self.questions:
            return "No previous questions"

        return "\n".join([
            f"Previous Question {i + 1}: {q}"
            for i, q in enumerate(self.questions)
        ])

    def clear(self) -> None:
        """Clear the question history"""
        self.questions = []


class QueryOptimizer:
    """Handles query optimization using AI21 models through langchain"""

    def __init__(self, llm=None, max_history: int = 5):
        """
        Initialize with an AI21 language model.

        Args:
            max_history (int): Maximum number of previous questions to store

        Raises:
            ValueError: If initialization fails
        """
        try:
            self.llm = llm if llm else ChatAI21(model="jamba-1.5-large")
            self.query_history = QueryHistory(max_questions=max_history)

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", get_query_optimization_prompt()),
                ("user", "{query}")
            ])

            # Create optimization chain
            self.chain = self.prompt | self.llm

        except Exception as e:
            raise ValueError(f"Failed to initialize query optimizer: {str(e)}")

    def optimize(self, query: str) -> str:
        """
        Optimize the query for better retrieval using previous questions as context.

        Args:
            query (str): The query to optimize

        Returns:
            str: The optimized query

        Raises:
            ValueError: If query optimization fails
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Get optimization with question history context
            optimized = self.chain.invoke({
                "query": query,
                "question_history": self.query_history.get_history()
            }).content

            # Add the current question to history after successful optimization
            self.query_history.add_question(query)

            return optimized.strip()

        except Exception as e:
            print(f"Query optimization failed: {str(e)}")
            return query  # Fallback to original query if optimization fails

    def clear_history(self) -> None:
        """Clear the question history"""
        self.query_history.clear()

    def get_question_history(self) -> List[str]:
        """
        Get list of previous questions.

        Returns:
            List[str]: List of previous questions in chronological order
        """
        return self.query_history.questions.copy()


class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: datetime = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()


class ChatMemory:
    def __init__(self, max_messages: int = 10, max_message_length: int = 500):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.max_message_length = max_message_length

    def add_message(self, role: str, content: str):
        """Add a new message to the chat history"""
        # Trim content if it exceeds max length
        trimmed_content = self._trim_content(content)
        self.messages.append(ChatMessage(role, trimmed_content))

        # Remove oldest messages if exceeding max messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def _trim_content(self, content: str) -> str:
        """Trim message content to max length while preserving word boundaries"""
        if len(content) <= self.max_message_length:
            return content

        # Find the last space before max length
        trimmed = content[:self.max_message_length]
        last_space = trimmed.rfind(' ')
        if last_space > 0:
            trimmed = trimmed[:last_space]

        return trimmed + "..."

    def get_chat_history(self) -> str:
        """Get formatted chat history for context"""
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.messages
        ])

    def clear(self):
        """Clear chat history"""
        self.messages = []


class PaperSearchAssistant:
    """Assistant that uses LLM to optimize search queries and fetch papers"""

    def __init__(self, llm: Any):
        self.llm = llm
        self.arxiv_retriever = ArXivRetriever()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", get_querry_search_prompt()),
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
            qo_llm: Any = None,
            embedding_model: Optional[Any] = None,
            k: int = 10,
            cache_dir: Optional[str] = None,
            max_chat_messages: int = 10,
            max_message_length: int = 500
    ):
        """
        Initialize the PDF reader with a language model and document source.
        """
        self.paper = paper
        self.llm = llm
        self.embedding_model = embedding_model
        self.k = k
        self.cache_dir = cache_dir
        self.docs: List[Document] = []
        self.chat_memory = ChatMemory(
            max_messages=max_chat_messages,
            max_message_length=max_message_length
        )

        self.query_optimizer = QueryOptimizer(llm=self.llm if qo_llm == 'same' else qo_llm)
        self.prompt2 = ChatPromptTemplate.from_messages([
            ("system", get_response_gen_prompt_casual(self.paper)),
            ("human", "{input}")
        ])
        self._initialize_vectorstore()
        self._setup_rag_chain()

    def _initialize_vectorstore(self) -> None:
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

        except Exception as e:
            raise ValueError(f"Failed to initialize vector store: {str(e)}")

    def _optimize_query(self, query: str) -> str:
        """Optimize the query using LLM before retrieval"""
        return self.query_optimizer.optimize(query)

    def _setup_rag_chain(self) -> None:
        """
        Set up the RAG chain for question answering.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", get_response_gen_prompt(self.paper)),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    @staticmethod
    def _pdf_to_documents(url) -> List[Document]:
        loader = PyPDFLoader(url)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        return pages

    async def a_answer(self, question: str) -> Optional[dict]:
        try:

            self.chat_memory.add_message("user", question)
            optimized_query = self._optimize_query(question)

            response = await self.rag_chain.ainvoke({
                "input": optimized_query,
                "chat_history": self.chat_memory.get_chat_history()
            })

            self.chat_memory.add_message("assistant", response["answer"])
            return response
        except Exception as e:
            raise ValueError(f"Failed to process question: {str(e)}")

    def answer(self, question: str) -> Optional[dict]:
        try:
            self.chat_memory.add_message("user", question)

            optimized_query = self._optimize_query(question)
            print(optimized_query)
            if "No need of retrieval" in optimized_query:
                chain = self.prompt2 | self.llm
                res = chain.invoke({
                    "input": question,
                    "chat_history": self.chat_memory.get_chat_history()
                })
                response = {'answer': res.content,
                            'metadata': None}
            else:
                res = self.rag_chain.invoke({
                    "input": optimized_query,
                    "chat_history": self.chat_memory.get_chat_history()
                })

                metadata = [(int(context.metadata.get('page', 0)), context.page_content) for context in res['context']]
                response = {
                    'answer': res['answer'],
                    'metadata.': metadata
                }

            self.chat_memory.add_message("assistant", response["answer"])
            return response
        except Exception as e:
            raise ValueError(f"Failed to process question: {str(e)}")

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_memory.clear()

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history as a list of messages"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.chat_memory.messages
        ]

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
