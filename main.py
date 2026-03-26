from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "parques temáticos"

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

template = PromptTemplate.from_template(
    """Crie um roteiro de viagem de {numero_dias} dias, 
    para uma familia com {numero_criancas} crianças, 
    que gosta de {atividade}."""
)

chain = template | modelo

resposta = chain.invoke({
    "numero_dias": numero_dias,
    "numero_criancas": numero_criancas,
    "atividade": atividade,
})

print(resposta.content)
