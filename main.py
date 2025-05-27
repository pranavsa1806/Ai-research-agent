from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
from tool import search_tool,wiki_tool,save_tool,save_to_txt

# Load environment variables from .env file
load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Set the base URL for OpenRouter
openrouter_base_url = "https://openrouter.ai/api/v1"

# Load API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize ChatOpenAI with OpenRouter
llm = ChatOpenAI(
    openai_api_base=openrouter_base_url,
    openai_api_key=api_key,
    model="mistralai/mistral-7b-instruct"  # or try 'openchat/openchat-7b'
)

parser=PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (     
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())




tools=[search_tool,wiki_tool,save_tool]
agent=create_tool_calling_agent(llm=llm, prompt=prompt,tools=tools)
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

query=input("what can i help in your research?")
raw_response=agent_executor.invoke({"query": query} )    
#print(raw_response)



raw_output = raw_response.get("output")

if raw_output.startswith("```") and raw_output.endswith("```"):
    raw_output = raw_output.strip("`").strip()

structured_response = parser.parse(raw_output)
if structured_response:
    result_string = str(structured_response)
    print(save_to_txt(result_string))