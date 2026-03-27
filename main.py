from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

set_debug(False)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

atividade = "Pontos turisticos no Chile"

class Destino(BaseModel):
    cidade: str = Field(..., description="A cidade sugerida para visitar")
    motivo: str = Field(..., description="O motivo pelo qual a cidade é recomendada para visitar")

class Restaurantes(BaseModel):
    cidade: str = Field(..., description="A cidade sugerida para visitar")
    restaurantes:str = Field(..., description="Uma lista de restaurantes recomendados na cidade sugerida")
    motivo: str = Field(..., description="O motivo pelo esses restaurantes são recomendados")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

template_cidade = PromptTemplate(
    template=""""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},
)

template_restaurantes = PromptTemplate(
    template=""""
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},
)

template_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
)

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

chain_1 = template_cidade | modelo | parseador_destino
chain_2 = template_restaurantes | modelo | parseador_restaurantes
chain_3 = template_cultural | modelo | StrOutputParser()

chain = chain_1 | { 
    "destino": RunnablePassthrough(),
    "restaurantes": chain_2, 
    "cultura": chain_3
}

response = chain.invoke({
    "interesse": atividade,
})

print("=== Destino ===")
print(response["destino"])
print("\n=== Restaurantes ===")
print(response["restaurantes"])
print("\n=== Cultural ===")
print(response["cultura"])
