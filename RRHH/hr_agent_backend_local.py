# cargar módulos principales
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
# cargar módulos de agentes y herramientas
import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
from Repositorio_API_KEY import API_KEY_OPENAI
from Repositorio_API_KEY import API_KEY_PINECONE
from Repositorio_API_KEY import ENVIRONMENT

# inicializar cliente de pinecone y conectar con el índice de pinecone
pinecone.init(
        api_key= API_KEY_PINECONE ,  
        environment= ENVIRONMENT
) 

index_name = 'tk-policy'
index = pinecone.Index(index_name) # connect to pinecone index

# inicializar objeto de embeddings; para usar con la consulta/entrada del usuario
embed = OpenAIEmbeddings(
                model = 'text-embedding-ada-002',
                openai_api_key= API_KEY_OPENAI ,
            )

# inicializar objeto de vectorstore(pinecone)
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(    
    openai_api_key= API_KEY_OPENAI , 
    model_name="gpt-3.5-turbo-0301", 
    temperature=0.2
    )

# inicializar objeto retriever de vectorstore
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("C:\soporte\RRHH\employee_data.csv") #  cargar employee_data.csv como dataframe
python = PythonAstREPLTool(locals={"df": df}) # establecer acceso de la herramienta de python_repl al dataframe

# crear herramienta de calculadora
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# crear variables para las cadenas embebidas en los prompt
# = 'Alexander Verdad' # establecer usuario
df_columns = df.columns.to_list() # imprimir nombres de columnas de df

# preparar el recuperador de vectores (policies de tk), el python_repl (con acceso a df) y la calculadora de langchain como herramientas para el agente
tools = [

    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Útil para cuando necesite responder preguntas sobre los datos de los empleados almacenados en el
        marco de datos de pandas'df'. 
        Ejecute las operaciones de Python Pandas en 'df' para ayudarlo a obtener la respuesta correcta.
        'df' tiene lo siguiente columns: {df_columns}
        """
    ),
    Tool(
    name = "Calculator",
    func=calculator.run,
    description = f"""
    Útil cuando necesitas hacer operaciones matemáticas o aritméticas.
    """
    )
]


# cambiar el valor del argumento prefix en la función initialize_agent. Esto sobrescribirá la plantilla de prompt predeterminada del tipo de agente Zero Shot
#agent_kwargs = {'prefix': f'Eres un asistente amigable de recursos humanos. Tu tarea es ayudar al usuario actual: {user}, con preguntas relacionadas a recursos humanos. Tienes acceso a las siguientes herramientas.'}
agent_kwargs = {'prefix': f'''Eres un amable asistente de recursos humanos. Tienes el objetivo de 
                responder preguntas que esten relacionadas con los RRHH brindando los nombres. 
                Tienes acceso al siguiente data frame serializado:'''}

# inicializar el agente LLM
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )
# definir la función de pregunta y respuesta para el frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response