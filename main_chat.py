import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

prompt_sugestao = ChatPromptTemplate(
    [
        ("system", "Você é um assistente de viagem especializado em destinos brasileiros, sugirá apenas um destino por vez."),
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

chain = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época para visitar esse lugar?",
]

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

for pergunta in perguntas:
    resposta = chain_with_memory.invoke(
        {
            "query": pergunta,
        },
        config={"session_id": sessao} # type: ignore
    )
    print(f"Usuário: {pergunta}")
    print(f"IA: {resposta}")
    print("-" * 50)
