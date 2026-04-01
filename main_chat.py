# =============================================================================
# CONCEITO: Memória em Conversas com LangChain (Chat History)
# =============================================================================
# Por padrão, os LLMs são stateless: cada chamada é independente e o modelo não
# lembra de mensagens anteriores. Para criar um chatbot com contexto de conversa,
# precisamos manualmente armazenar e reenviar o histórico de mensagens a cada turno.
#
# O LangChain oferece abstrações para isso:
#   - InMemoryChatMessageHistory: armazena o histórico em memória RAM (não persiste).
#   - RunnableWithMessageHistory: envolve uma chain e injeta o histórico automaticamente.
# =============================================================================
import os
from dotenv import load_dotenv

# ChatGoogleGenerativeAI: classe do LangChain para usar o modelo Gemini do Google.
from langchain_google_genai import ChatGoogleGenerativeAI

# ChatPromptTemplate: versão do PromptTemplate otimizada para modelos de chat.
# Suporta múltiplos papéis de mensagem: "system", "human", "ai", "placeholder".
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser: converte a resposta do modelo (objeto AIMessage) em uma string simples.
from langchain_core.output_parsers import StrOutputParser

# InMemoryChatMessageHistory: implementação de histórico de chat armazenado em memória.
# Mantém uma lista de mensagens (HumanMessage, AIMessage) para uma sessão específica.
from langchain_core.chat_history import InMemoryChatMessageHistory

# RunnableWithMessageHistory: wrapper que adiciona suporte a histórico em qualquer chain.
# A cada invocação, ele:
#   1. Recupera o histórico da sessão atual.
#   2. Injeta as mensagens anteriores no prompt.
#   3. Executa a chain.
#   4. Salva a nova mensagem e resposta no histórico.
from langchain_core.runnables.history import RunnableWithMessageHistory

# Carrega as variáveis de ambiente do arquivo .env (ex.: GEMINI_API_KEY).
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Instancia o modelo LLM do Google Gemini.
# gemini-2.5-flash-lite: versão leve e rápida, ideal para chatbots.
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

# =============================================================================
# CONCEITO: ChatPromptTemplate com Placeholder de Histórico
# =============================================================================
# O ChatPromptTemplate organiza as mensagens em papéis:
#   - "system": instrução de comportamento para o modelo (não visível ao usuário).
#   - "placeholder": posição reservada onde o histórico de mensagens será injetado.
#     O nome {historico} deve coincidir com o parâmetro history_messages_key do wrapper.
#   - "human": a mensagem atual do usuário.
#
# Dessa forma, a cada nova pergunta, o modelo recebe todo o contexto anterior.
# =============================================================================
prompt_sugestao = ChatPromptTemplate(
    [
        ("system", "Você é um assistente de viagem especializado em destinos brasileiros, sugirá apenas um destino por vez."),
        # O placeholder {historico} será substituído pelas mensagens anteriores da conversa.
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

# Monta a chain básica: prompt → modelo → parser de texto.
chain = prompt_sugestao | modelo | StrOutputParser()

# =============================================================================
# CONCEITO: Gerenciamento de Sessões
# =============================================================================
# Em aplicações reais, múltiplos usuários podem conversar simultaneamente.
# Cada usuário precisa do seu próprio histórico, identificado por um session_id.
# O dicionário `memoria` mapeia session_id → InMemoryChatMessageHistory.
# A função historico_por_sessao é chamada pelo RunnableWithMessageHistory para
# recuperar (ou criar) o histórico de uma sessão específica.
# =============================================================================
memoria = {}
sessao = "aula_langchain"

def historico_por_sessao(sessao: str):
    # Se ainda não existe histórico para esta sessão, cria um novo.
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

# Perguntas simulando uma conversa com múltiplos turnos.
# Na segunda pergunta, "esse lugar" só faz sentido se o modelo lembrar da primeira.
perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época para visitar esse lugar?",
]

# =============================================================================
# CONCEITO: RunnableWithMessageHistory
# =============================================================================
# Envolve a chain com suporte automático ao histórico de mensagens.
# Parâmetros:
#   - runnable: a chain que será executada.
#   - get_session_history: função que retorna o histórico para um dado session_id.
#   - input_messages_key: chave do dicionário de entrada que contém a mensagem atual.
#   - history_messages_key: chave usada no prompt para injetar o histórico ({historico}).
# =============================================================================
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

# Itera sobre as perguntas simulando uma conversa sequencial.
# O session_id garante que o histórico correto seja usado a cada chamada.
for pergunta in perguntas:
    resposta = chain_with_memory.invoke(
        {
            "query": pergunta,
        },
        # O config com session_id identifica qual histórico usar para esta conversa.
        config={"session_id": sessao} # type: ignore
    )
    print(f"Usuário: {pergunta}")
    print(f"IA: {resposta}")
    print("-" * 50)
