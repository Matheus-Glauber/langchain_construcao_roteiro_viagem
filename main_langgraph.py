# =============================================================================
# CONCEITO: LangGraph — Grafos de Agentes com Estado
# =============================================================================
# LangGraph é uma extensão do LangChain para construir fluxos de trabalho complexos
# com LLMs usando a estrutura de grafos dirigidos (DAG — Directed Acyclic Graph).
#
# Enquanto chains LangChain são lineares (A → B → C), o LangGraph permite:
#   - Ramificações condicionais (if/else baseado em output do LLM)
#   - Ciclos (loops, retentativas)
#   - Múltiplos nós independentes com estado compartilhado
#
# Conceitos fundamentais:
#   - STATE: dicionário compartilhado entre todos os nós do grafo.
#   - NODE: função assíncrona que lê e atualiza o estado.
#   - EDGE: conexão entre nós (pode ser fixa ou condicional).
#   - START/END: nós especiais que marcam o início e o fim do grafo.
#
# Fluxo deste exemplo:
#   START → [rotear] → (praia ou montanha) → END
#
# O nó "rotear" usa o LLM para classificar a pergunta e decidir para qual
# especialista enviar: "praia" (Sra. Praia) ou "montanha" (Sr. Montanha).
# =============================================================================

# ChatGoogleGenerativeAI: LLM do Google Gemini para geração de respostas.
from langchain_google_genai import ChatGoogleGenerativeAI

# ChatPromptTemplate e StrOutputParser: para montar prompts e parsear respostas.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# StateGraph: classe principal do LangGraph para definir o grafo.
# START e END: nós especiais que marcam o início e o fim do fluxo.
from langgraph.graph import StateGraph, START, END

# RunnableConfig: configuração de execução passada para chains (timeouts, callbacks, etc.).
from langchain_core.runnables import RunnableConfig

# python-dotenv: carrega variáveis de ambiente do .env.
from dotenv import load_dotenv

# TypedDict: cria dicionários tipados em Python — usado para definir o Estado do grafo.
# Literal: restringe um tipo a valores específicos (ex.: apenas "praia" ou "montanha").
from typing import TypedDict, Literal

# asyncio: biblioteca padrão Python para programação assíncrona (async/await).
# LangGraph suporta execução assíncrona nativamente, melhorando a performance.
import asyncio
import os

load_dotenv()

# Modelo LLM compartilhado entre todos os nós do grafo.
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# =============================================================================
# CONCEITO: Especialistas (Personas)
# =============================================================================
# Cada especialista é uma chain independente com um prompt de sistema diferente.
# O prompt de sistema ("system") define a personalidade e expertise do assistente.
# Aqui criamos dois especialistas:
#   - Sra. Praia: responde sobre destinos de praia.
#   - Sr. Montanha: responde sobre montanhas e atividades radicais.
# =============================================================================
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

# Chains dos especialistas: prompt → modelo → texto.
chain_praia = prompt_consultor_praia | model | StrOutputParser()
chain_montanha = prompt_consultor_montanha | model | StrOutputParser()

# =============================================================================
# CONCEITO: Roteamento com Structured Output
# =============================================================================
# O roteador é um LLM que classifica a intenção da pergunta e retorna um objeto
# estruturado (Rota) em vez de texto livre.
#
# TypedDict Rota: define que o roteador deve retornar {"destino": "praia"} ou
#                 {"destino": "montanha"} — nada mais.
#
# model.with_structured_output(Rota): instrui o Gemini a retornar sempre um
# JSON que corresponda ao esquema de Rota (usando function calling internamente).
# =============================================================================
class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

# Chain do roteador: analisa a pergunta e retorna {"destino": "praia"} ou {"destino": "montanha"}.
roteador = prompt_roteador | model.with_structured_output(Rota) # type: ignore

# =============================================================================
# CONCEITO: Estado do Grafo (State)
# =============================================================================
# O Estado é o "banco de dados" compartilhado entre todos os nós do grafo.
# Cada nó pode ler qualquer campo do estado e retornar um dicionário parcial
# para atualizar apenas os campos que ele modificou.
#
# - query: a pergunta original do usuário (nunca muda).
# - destino: preenchido pelo nó roteador com a classificação do LLM.
# - resposta: preenchida pelo nó especialista (praia ou montanha).
# =============================================================================
class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

# =============================================================================
# CONCEITO: Nós do Grafo (Nodes)
# =============================================================================
# Cada nó é uma função assíncrona que:
#   1. Recebe o estado atual do grafo.
#   2. Executa alguma lógica (chamada ao LLM, transformação de dados, etc.).
#   3. Retorna um dicionário com os campos do estado que devem ser atualizados.
#
# O uso de async/await garante que as chamadas ao LLM não bloqueiem o event loop,
# permitindo execução eficiente de múltiplos nós.
# =============================================================================

# Nó roteador: classifica a pergunta e salva o destino no estado.
async def no_roteador(estado: Estado, config=RunnableConfig):
    return {
        "destino": await roteador.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

# Nó especialista praia: responde usando o persona de Sra. Praia.
async def no_praia(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_praia.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

# Nó especialista montanha: responde usando o persona de Sr. Montanha.
async def no_montanha(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_montanha.ainvoke({"query": estado["query"]}, config) # type: ignore
    }

# =============================================================================
# CONCEITO: Arestas Condicionais (Conditional Edges)
# =============================================================================
# Esta função é chamada após o nó "rotear" e decide qual nó executar a seguir.
# Ela lê o campo "destino" do estado (preenchido pelo roteador) e retorna o
# nome do próximo nó como string. O LangGraph usa esse valor para navegar no grafo.
# =============================================================================
def escolher_chain(estado: Estado)->Literal["praia", "montanha"]:
    return estado["destino"]["destino"]

# =============================================================================
# CONCEITO: Construção do Grafo
# =============================================================================
# StateGraph(Estado): cria um grafo que usa Estado como esquema de estado compartilhado.
# add_node: registra uma função como nó do grafo com um nome identificador.
# add_edge: cria uma conexão fixa entre dois nós (sempre executada).
# add_conditional_edges: cria conexões condicionais baseadas no retorno de uma função.
#
# Estrutura final do grafo:
#   START → "rotear" → (escolher_chain) → "praia" → END
#                                       → "montanha" → END
# =============================================================================
grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador) # type: ignore
grafo.add_node("praia", no_praia) # type: ignore
grafo.add_node("montanha", no_montanha) # type: ignore

# Aresta fixa: o grafo começa sempre pelo nó "rotear".
grafo.add_edge(START, "rotear")

# Aresta condicional: após "rotear", a função escolher_chain decide o próximo nó.
grafo.add_conditional_edges("rotear", escolher_chain)

# Arestas fixas: após responder, o grafo termina.
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

# compile(): valida o grafo, otimiza o fluxo e retorna um objeto executável.
# Após compilado, o grafo não pode mais ser modificado.
app = grafo.compile()

async def main():
    query = "Quero visitar um lugar no Brasil, famoso por praias e cultura."
    # ainvoke: executa o grafo de forma assíncrona, passando o estado inicial.
    # O grafo processa START → rotear → (praia ou montanha) → END e retorna o estado final.
    resposta = await app.ainvoke({"query": query}) # type: ignore
    print(resposta)

# asyncio.run: ponto de entrada para executar código assíncrono em Python.
# Cria e gerencia o event loop automaticamente.
asyncio.run(main())
