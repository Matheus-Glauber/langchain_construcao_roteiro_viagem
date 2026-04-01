from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from typing import TypedDict, Literal
import asyncio
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praias."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha. Você é um especialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

chain_praia = prompt_consultor_praia | model | StrOutputParser()
chain_montanha = prompt_consultor_montanha | model | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | model.with_structured_output(Rota) # type: ignore

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {
        "destino": await roteador.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

async def no_praia(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_praia.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_montanha.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

def escolher_chain(estado: Estado)->Literal["praia", "montanha"]:
    return estado["destino"]["destino"]

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador) # type: ignore
grafo.add_node("praia", no_praia) # type: ignore
grafo.add_node("montanha", no_montanha) # type: ignore

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_chain)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    query = "Quero visitar um lugar no Brasil, famoso por praias e cultura."
    resposta = await app.ainvoke({"query": query}) # type: ignore
    print(resposta)

asyncio.run(main())
