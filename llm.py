from dotenv import load_dotenv
load_dotenv()  

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

#PDF loader

loader = PyPDFLoader("운수.pdf")
pages = loader.load_and_split()


#PDF split

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

texts = text_splitter.split_documents(pages)


#embedding 

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

#vectorDB
db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name='chroma-db',persist_directory="chroma_db")

#Question
query = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatUpstage(model="solar-pro")
retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 문맥(context)을 참고하여 질문에 답변하세요. 답을 모르면 모른다고 말하세요.\n\n{context}"),
    ("human", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#Generation
ai_message = retrieval_chain.invoke({"input": query})
print(ai_message["answer"])