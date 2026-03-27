from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

set_debug(True)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

atividade = "parques temáticos"

class Destino(BaseModel):
    cidade: str = Field(..., description="A cidade sugerida para visitar")
    motivo: str = Field(..., description="O motivo pelo qual a cidade é recomendada para visitar")

parseador = JsonOutputParser(pydantic_object=Destino)

template_cidade = PromptTemplate(
    template=""""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()},
)

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

chain = template_cidade | modelo | parseador

response = chain.invoke({
    "interesse": atividade,
})

print(response)
