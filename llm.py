from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import os
from langchain_community.document_loaders import TextLoader
store = {}


def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def text_split():
    loader = TextLoader("운수.txt", encoding="utf-8")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(pages)
    return texts


def get_llm(model="solar-pro"):
    llm =ChatUpstage(model=model)
    return llm


def get_retriever():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    texts = text_split()

    if os.path.exists("chroma_db"):
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name='chroma-db')
    else:
        db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name='chroma-db',persist_directory="chroma_db")
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_rag_chain():
    llm = get_llm()
    history_aware_retriever = get_history_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        - 다음 문맥(context)을 참고하여 질문에 답변하세요.
        - 문맥에 직접적으로 명시되지 않았더라도, 문맥에 있는 정보로부터 합리적으로 추론할 수 있다면 추론해서 답변하세요.
        - 예를 들어 인물이 특정 대상을 간절히 원하거나 반복적으로 언급한다면, 이는 그 인물이 그것을 좋아하거나 원한다는 근거로 볼 수 있습니다.
        - 문맥과 전혀 관련이 없거나 추론조차 불가능한 질문일 때만 "모른다"고 답변하세요.
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain


def get_ai_message(user_question):
    rag_chain = get_rag_chain()
    ai_message = rag_chain.stream({"input": user_question}, config={"configurable": {"session_id": "abc123"}})
    return ai_message
