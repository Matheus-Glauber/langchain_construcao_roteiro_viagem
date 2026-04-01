from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview") 
documento = TextLoader("documentos/GTB_gold_Nov23.txt", encoding="utf-8").load()

pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documento)

dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | model | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join([trecho.page_content for trecho in trechos])
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print(responder("Como devo proceder caso tenha um item roubado?"))
