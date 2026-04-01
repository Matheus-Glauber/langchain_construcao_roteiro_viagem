# =============================================================================
# CONCEITO: RAG — Retrieval-Augmented Generation (Geração Aumentada por Recuperação)
# =============================================================================
# RAG é uma técnica que combina dois componentes principais:
#
#   1. RETRIEVAL (Recuperação): busca trechos relevantes de documentos em uma base
#      vetorial usando similaridade semântica (embeddings).
#
#   2. GENERATION (Geração): usa um LLM para gerar uma resposta com base APENAS
#      nos trechos recuperados, evitando alucinações (respostas inventadas).
#
# Fluxo do RAG:
#   Documentos → Chunking → Embeddings → VectorStore
#                                              ↓
#   Pergunta → Embedding da pergunta → Busca por similaridade → Trechos relevantes
#                                              ↓
#   Prompt com contexto → LLM → Resposta fundamentada
#
# Caso de uso aqui: responder perguntas sobre apólices de seguro com base nos
# documentos oficiais, sem que o modelo invente informações.
# =============================================================================

# python-dotenv: carrega variáveis de ambiente do arquivo .env.
from dotenv import load_dotenv

# ChatGoogleGenerativeAI: LLM do Google Gemini para geração de respostas.
# GoogleGenerativeAIEmbeddings: modelo de embeddings do Google para vetorizar textos.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# PyPDFLoader: carrega arquivos PDF e os converte em documentos LangChain.
# Cada página do PDF vira um objeto Document com page_content e metadata.
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# FAISS (Facebook AI Similarity Search): biblioteca de busca vetorial eficiente.
# Armazena os embeddings dos documentos e permite busca rápida por similaridade.
# É executado localmente (em memória), sem necessidade de servidor externo.
from langchain_community.vectorstores import FAISS

# RecursiveCharacterTextSplitter: divide documentos grandes em pedaços menores (chunks).
# Tenta dividir por parágrafos, depois frases, depois caracteres, preservando o contexto.
# chunk_size: tamanho máximo de cada pedaço (em caracteres).
# chunk_overlap: sobreposição entre pedaços consecutivos para não perder contexto nas bordas.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChatPromptTemplate e StrOutputParser: para montar o prompt e parsear a resposta.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# LLM responsável por gerar a resposta final com base no contexto recuperado.
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# =============================================================================
# CONCEITO: Embeddings
# =============================================================================
# Embeddings são representações numéricas (vetores) de textos.
# Textos com significados semelhantes terão vetores próximos no espaço vetorial.
# Ex.: "como acionar seguro" e "procedimento para sinistro" gerarão vetores similares.
#
# O modelo gemini-embedding-2-preview converte textos em vetores de alta dimensão.
# Esses vetores são usados tanto para indexar os documentos quanto para
# transformar a pergunta do usuário antes da busca por similaridade.
# =============================================================================
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Lista de arquivos PDF com as apólices de seguro em diferentes categorias.
# Cada arquivo contém as condições gerais do seguro para uma categoria diferente.
arquivos = ["documentos/GTB_gold_Nov23.pdf", "documentos/GTB_platinum_Nov23.pdf", "documentos/GTB_standard_Nov23.pdf"]

# =============================================================================
# CONCEITO: Document Loaders
# =============================================================================
# Document Loaders são responsáveis por carregar e converter diferentes fontes de dados
# (PDFs, TXTs, sites, bancos de dados, etc.) em objetos Document do LangChain.
# PyPDFLoader lê cada página do PDF e retorna uma lista de Documents.
# sum([...], []) é um truque Python para "achatar" uma lista de listas em uma só lista.
# =============================================================================
documentos = sum([PyPDFLoader(arquivo).load() for arquivo in arquivos], [])

# =============================================================================
# CONCEITO: Text Splitting (Chunking)
# =============================================================================
# LLMs têm um limite de tokens que podem processar de uma vez (context window).
# Além disso, embeddings de textos muito longos perdem precisão semântica.
# Por isso, dividimos os documentos em pedaços menores (chunks) antes de vetorizá-los.
#
# chunk_size=1000: cada chunk terá no máximo 1000 caracteres.
# chunk_overlap=200: os últimos 200 caracteres de um chunk se repetem no início do próximo,
#                    evitando que informações importantes sejam "cortadas" na divisão.
# =============================================================================
pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documentos)

# =============================================================================
# CONCEITO: VectorStore e Retriever
# =============================================================================
# FAISS.from_documents:
#   1. Chama embeddings.embed_documents() para vetorizar cada chunk.
#   2. Armazena os vetores e os textos originais no índice FAISS.
#
# as_retriever: converte o VectorStore em um Retriever (interface padrão LangChain).
#   search_kwargs={"k": 2}: ao receber uma pergunta, retorna os 2 chunks mais similares.
#
# Na busca, a pergunta do usuário também é vetorizada e comparada com todos os
# chunks armazenados usando distância cosseno (ou L2), retornando os mais próximos.
# =============================================================================
dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

# =============================================================================
# CONCEITO: Prompt com Contexto (Grounded Generation)
# =============================================================================
# A instrução "Responda usando exclusivamente o conteúdo fornecido" é fundamental
# para evitar alucinações: impede que o modelo use seu conhecimento interno e
# o força a basear a resposta apenas nos trechos recuperados dos documentos.
# {query}: a pergunta do usuário.
# {contexto}: os trechos relevantes recuperados do VectorStore.
# =============================================================================
prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:")
    ]
)

# Chain de geração: recebe prompt com pergunta + contexto → LLM → resposta em texto.
cadeia = prompt_consulta_seguro | model | StrOutputParser()

def responder(pergunta: str):
    # Etapa 1 — RETRIEVAL: vetoriza a pergunta e busca os chunks mais relevantes.
    trechos = dados_recuperados.invoke(pergunta)

    # Etapa 2 — Monta o contexto concatenando os textos dos chunks recuperados.
    contexto = "\n\n".join([trecho.page_content for trecho in trechos])

    # Etapa 3 — GENERATION: envia pergunta + contexto ao LLM e retorna a resposta.
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print(responder("Como devo proceder caso tenha um item comprado roubado e sabendo que tenho o cartão gold?"))
