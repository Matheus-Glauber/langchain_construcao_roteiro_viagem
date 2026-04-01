from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Literal
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

roteador = prompt_roteador | model.with_structured_output(Rota)

def responda(pergunta: str):
    rota = roteador.invoke({"query": pergunta})
    if rota["destino"] == "praia":
        return chain_praia.invoke({"query": pergunta})
    elif rota == "montanha":
        return chain_montanha.invoke({"query": pergunta})
    else:
        return "Desculpe, não entendi sua preferência."
    
print(responda("Quero escalar uma montanha no Chile."))
