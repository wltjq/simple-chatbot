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

#chat history를 저장 (session_id -> ChatMessageHistory)
store = {}


def get_session_history(session_id:str) -> BaseChatMessageHistory:
    #해당 세션의 기록이 없으면 새로 생성 
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def text_split():
    #txt 파일 로드 
    loader = TextLoader("운수.txt", encoding="utf-8")
    pages = loader.load()

    #chunk 단위로 txt 분할 (chunk_size : 청크 크기, chunk_overlap : 겹치는 길이)
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

    #기존에 저장된 vector DB(Chroma)가 있으면 불러오고, 없으면 새로 생성 
    if os.path.exists("chroma_db"):
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name='chroma-db')
    else:
        db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name='chroma-db',persist_directory="chroma_db")
    
    #상위 4개의 문서를 반환하는 Retriever 생성
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    #chat history(이전 질문)를 참고해 질문을 재구성하는 system prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    #system prompt, chat history, user question을 입력받아, LLM이 대화 맥락을 반영한 독립적인 질문으로 재구성하도록 지시하는 prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    #대화 맥락 반영해서 검색까지 해주는 업그레이드된 retriever
    #contextualize_q_prompt -> llm이 질문 재구성 -> retriever -> 문서 검색
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_rag_chain():
    llm = get_llm()
    history_aware_retriever = get_history_retriever()

    #문맥(context)과 질문(input)을 입력받아 답변을 생성하도록 지시하는 prompt
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

    #문서들을 하나로 합쳐 LLM에 전달하는 체인
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    #retriever + 답변 생성 체인을 결합한 RAG 체인
    #history_aware_retriever -> 문서 검색 -> combine_docs_chain -> 답변 생성
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    
    #대화 기록을 관리해주는 체인
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') #답변 부분만 추출
    return conversational_rag_chain


def get_ai_message(user_question):
    #사용자의 질문을 받아 RAG 체인을 통해 답변 반환 
    rag_chain = get_rag_chain()
    ai_message = rag_chain.stream({"input": user_question}, config={"configurable": {"session_id": "abc123"}})
    return ai_message
