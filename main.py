# =============================================================================
# CONCEITO: LangChain — Chains e Output Parsers
# =============================================================================
# LangChain é um framework para construção de aplicações com LLMs (Large Language Models).
# Seu principal conceito é a "chain" (cadeia): uma sequência de componentes encadeados
# usando o operador | (pipe), inspirado no padrão de design "pipeline".
#
# Fluxo básico de uma chain:
#   PromptTemplate --> LLM --> OutputParser
#
# Cada componente recebe uma entrada, processa e passa para o próximo.
# =============================================================================

# langchain_google_genai: integração do LangChain com os modelos Gemini do Google.
# ChatGoogleGenerativeAI é a classe que representa o modelo de chat (LLM).
from langchain_google_genai import ChatGoogleGenerativeAI

# PromptTemplate: permite criar templates de prompts reutilizáveis com variáveis dinâmicas.
# Ex.: "Sugira uma cidade para {interesse}" — {interesse} será preenchido em tempo de execução.
from langchain_core.prompts import PromptTemplate

# JsonOutputParser: parseia a resposta do modelo e extrai um objeto JSON estruturado.
# StrOutputParser: parseia a resposta do modelo e retorna uma string simples.
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# set_debug: ativa/desativa logs detalhados do LangChain para fins de depuração.
# Quando True, imprime cada etapa da chain no console (útil para desenvolvimento).
from langchain_core.globals import set_debug

# RunnablePassthrough: componente que passa os dados recebidos sem modificá-los.
# Usado para preservar o resultado de uma etapa ao ramificar a chain.
from langchain_core.runnables import RunnablePassthrough

# Pydantic: biblioteca de validação de dados em Python.
# BaseModel: classe base para criar modelos de dados com tipagem e validação automática.
# Field: permite descrever cada campo do modelo, incluindo metadados como descrição.
# O LangChain usa esses modelos Pydantic para instruir o LLM a retornar JSON estruturado.
from pydantic import BaseModel, Field

# python-dotenv: carrega variáveis de ambiente de um arquivo .env para o os.environ.
# Evita expor credenciais diretamente no código-fonte.
from dotenv import load_dotenv
import os

# Desativa o modo de depuração do LangChain (não imprime logs intermediários).
set_debug(False)

# Carrega as variáveis definidas no arquivo .env (ex.: GEMINI_API_KEY=...).
load_dotenv()

# Lê a chave de API do Google Gemini a partir das variáveis de ambiente.
# Boa prática de segurança: nunca colocar credenciais diretamente no código.
api_key = os.getenv("GEMINI_API_KEY")

# Interesse do usuário que guiará as sugestões do roteiro de viagem.
atividade = "Pontos turisticos no Chile"

# =============================================================================
# CONCEITO: Modelos Pydantic como Esquema de Saída
# =============================================================================
# Ao definir classes Pydantic, instruímos o LLM a retornar respostas no formato
# esperado. O JsonOutputParser usa essas classes para gerar instruções de formatação
# automáticas (get_format_instructions) que são injetadas no prompt, guiando o modelo
# a responder sempre com a estrutura correta.
# =============================================================================

# Modelo que representa o destino sugerido pelo LLM.
class Destino(BaseModel):
    cidade: str = Field(..., description="A cidade sugerida para visitar")
    motivo: str = Field(..., description="O motivo pelo qual a cidade é recomendada para visitar")

# Modelo que representa as sugestões de restaurantes para uma cidade.
class Restaurantes(BaseModel):
    cidade: str = Field(..., description="A cidade sugerida para visitar")
    restaurantes:str = Field(..., description="Uma lista de restaurantes recomendados na cidade sugerida")
    motivo: str = Field(..., description="O motivo pelo esses restaurantes são recomendados")

# Instância do parser que converte a resposta textual do LLM em um objeto Destino.
parseador_destino = JsonOutputParser(pydantic_object=Destino)

# Instância do parser que converte a resposta textual do LLM em um objeto Restaurantes.
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

# =============================================================================
# CONCEITO: PromptTemplate com Variáveis Parciais
# =============================================================================
# PromptTemplate aceita dois tipos de variáveis:
#   - input_variables: preenchidas em tempo de execução (pelo usuário ou pela chain).
#   - partial_variables: preenchidas antecipadamente (valores fixos conhecidos de antemão).
# Aqui, as instruções de formatação JSON são injetadas como partial_variables, pois
# são sempre as mesmas e não precisam ser fornecidas a cada invocação.
# =============================================================================

# Template para solicitar ao LLM a sugestão de uma cidade.
# {interesse} será fornecido em tempo de execução.
# {formato_de_saida} já está preenchido com as instruções JSON geradas pelo parser.
template_cidade = PromptTemplate(
    template=""""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},
)

# Template para solicitar sugestões de restaurantes.
# {cidade} será preenchido automaticamente pela chain com o resultado da etapa anterior.
template_restaurantes = PromptTemplate(
    template=""""
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},
) # type: ignore

# Template para solicitar atividades culturais.
# Usa StrOutputParser (texto livre), pois não precisamos de estrutura JSON aqui.
template_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
) # type: ignore

# =============================================================================
# CONCEITO: LLM — ChatGoogleGenerativeAI (Gemini)
# =============================================================================
# O modelo gemini-2.5-flash-lite é um LLM (Large Language Model) do Google.
# "Flash" indica que é otimizado para velocidade e eficiência, sendo ideal para
# aplicações que precisam de respostas rápidas sem sacrificar muita qualidade.
# =============================================================================
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

# =============================================================================
# CONCEITO: Chains Sequenciais e Paralelas com o operador |
# =============================================================================
# O operador | (pipe) encadeia componentes: a saída de um vira entrada do próximo.
#
# chain_1: template_cidade → modelo → parseador_destino
#   Recebe {interesse}, monta o prompt, envia ao LLM, retorna objeto Destino.
#
# chain_2: template_restaurantes → modelo → parseador_restaurantes
#   Recebe {cidade}, monta o prompt, envia ao LLM, retorna objeto Restaurantes.
#
# chain_3: template_cultural → modelo → StrOutputParser
#   Recebe {cidade}, monta o prompt, envia ao LLM, retorna texto livre.
# =============================================================================
chain_1 = template_cidade | modelo | parseador_destino
chain_2 = template_restaurantes | modelo | parseador_restaurantes
chain_3 = template_cultural | modelo | StrOutputParser()

# =============================================================================
# CONCEITO: Fan-out (ramificação paralela) com RunnablePassthrough
# =============================================================================
# Após chain_1 retornar o objeto Destino, a chain principal se ramifica em 3 caminhos:
#
#   - "destino": RunnablePassthrough() — apenas repassa o objeto Destino sem modificar.
#   - "restaurantes": chain_2 — usa {cidade} do Destino para buscar restaurantes.
#   - "cultura": chain_3 — usa {cidade} do Destino para buscar atividades culturais.
#
# O resultado final é um dicionário com as três chaves preenchidas.
# Isso é chamado de "fan-out" (expansão): um resultado alimenta múltiplas sub-chains.
# =============================================================================
chain = chain_1 | { 
    "destino": RunnablePassthrough(),
    "restaurantes": chain_2, 
    "cultura": chain_3
}

# Executa toda a chain passando o interesse do usuário como entrada inicial.
# O LangChain cuida de encadear todos os passos automaticamente.
response = chain.invoke({
    "interesse": atividade,
})

print("=== Destino ===")
print(response["destino"])
print("\n=== Restaurantes ===")
print(response["restaurantes"])
print("\n=== Cultural ===")
print(response["cultura"])
