from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

atividade = "parques temáticos"

template_cidade = PromptTemplate(
    template=""""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"],
)

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

chain = template_cidade | modelo | StrOutputParser()

response = chain.invoke({
    "interesse": atividade,
})

print(response)
