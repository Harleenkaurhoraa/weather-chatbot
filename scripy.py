import os
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
import requests

from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from IPython.display import Image, display

load_dotenv()
CYFUTURE_API_KEY = os.getenv("CYFUTURE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = os.getenv("BASE_URL")

"""
To store ai and human message logic
"""
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    city: str
    memory: ConversationBufferMemory

def initialize_memory():
    return ConversationBufferMemory(
        k=5,  
        memory_key="chat_history",
        return_messages=True
    )

def ask_ai(question: str) -> str:
    """ use CyfutureAI api for llm"""
    try:
        response = requests.post(
            "https://api.cyfuture.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CYFUTURE_API_KEY}"},
            json={
                "model": "llama8",
                "messages": [{"role": "user", "content": question}]
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error:{e}")
        return "Sorry, I couldn't process that request."

DATA_FOLDER = r"C:\GenAI Projects\weather chatbot\data_pdfs"
def data_ingestion(folder_path=DATA_FOLDER):
    """
    Loads all PDFs from a given folder (folder_path).
    Splits them into chunks for vector search.
    """
    try:
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error:{e}")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = data_ingestion()

## functionall call 

print(docs[:2])

FAISS_INDEX = "faiss_index"
def create_vector_store(docs):
    """
    Creates a local FAISS index from the given documents
    and saves it to the "faiss_index" folder.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore_faiss = FAISS.from_documents(docs,embeddings)
    vectorstore_faiss.save_local(FAISS_INDEX)
    print("FAISS index created and saved successfully.")
    return vectorstore_faiss.as_retriever(search_kwargs={"k": 3})

create_vector_store(docs)

import requests

def get_current_weather(location):
    base_url = "http://api.weatherapi.com/v1/current.json"
    api_key = "93e24c7b0ea440c78c7185341252004"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "yes"
    }

    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        location = data['location']['name']
        # country = data['location']['country']
        temp_c = data['current']['temp_c']
        condition = data['current']['condition']['text']
        feelslike_c = data['current']['feelslike_c']
        humidity = data['current']['humidity']
        
        return (
            f"Weather in {location}\n"#, #{country}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temp_c}°C (Feels like {feelslike_c}°C)\n"
            f"Humidity: {humidity}%"
        )
    else:
        return f"Failed to fetch weather data. Status Code: {response.status_code}"

# Example usage:
print(get_current_weather("Delhi"))


# nodes and graphs
def first_node_classifing_intent(state):
    last_message = state["messages"][-1].content
    
    prompt = f"""
    Determine the intent of the following message:
    "{last_message}"
    
    If the message is asking about current weather, weather forecast, or weather conditions for a specific location, respond with "weather_api".
    If the message is asking about general weather information, climate patterns, or safety measures, respond with "rag".
    
    Just respond with either "weather_api" or "rag".
    """
    
    intent_response = ask_ai(prompt)
    intent = intent_response.strip().lower()
    state["intent"] = intent

    city = "unknown"
    if intent == "weather_api":
        city_prompt = f"""
        Extract the city name from this message: "{last_message}"
        If no city is mentioned, respond with "unknown".
        Just provide the city name or "unknown", nothing else.
        """
        city_response = ask_ai(city_prompt)
        city = city_response.strip().lower()
    
    # Save context in memory
    state["memory"].save_context(
        {"input": last_message},
        {"output": f"Intent: {intent}, City: {city}"}
    )
    # Store in state for next node
    state["intent"] = intent
    state["city"] = city
    
    return {"intent": intent, "city": city}

# # weather agent

import nest_asyncio
nest_asyncio.apply()

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

class Weather_agent(BaseModel):
    location: str
    description: str
    temperature_celsius: float

# Define get_weather_forecast function
def get_weather_forecast(ctx: RunContext,city: str) -> Weather_agent:
    """Returns current weather forecast for a city using OpenWeatherMap API."""
    
    params = {
        'q': city,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(BASE_URL, params=params)
    response = response.json()
    
    return Weather_agent(
        location=response["name"],
        description=response["weather"][0]["description"],
        temperature_celsius=response["main"]["temp"]
    )

def weather_node(state):
    city = state["city"]
    forecast = get_weather_forecast(RunContext(), city) if city != "unknown" else None
    response = f"In {forecast.location}, it's currently {forecast.description} with a temperature of {forecast.temperature_celsius}°C." if forecast else "I need a city name to check the weather."
    return {"messages": [AIMessage(content=response)]}

# # rag-queries

# have to load the existing FAISS index 
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore_faiss = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
retriever = vectorstore_faiss.as_retriever(search_kwargs={"k": 3})

def handle_rag_query(state):
    """Handle general weather information queries"""
    # Simplified for now to fix the error
    last_message = state["messages"][-1].content
    response = f"You asked about general weather information: '{last_message}'. This would normally use RAG to provide an informative answer."
    return {"messages": [AIMessage(content=response)]}

# ## function for router 

def router(state):
    """Route to the appropriate node based on intent"""
    return "weather_node" if state["intent"] == "weather_api" else "handle_rag_query"

def build_graph():
    """Build and return the graph"""
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("first_node_classifing_intent", first_node_classifing_intent)
    builder.add_node("weather_node", weather_node)
    builder.add_node("handle_rag_query", handle_rag_query)
    
    # Add edges (logic)
    builder.add_edge(START, "first_node_classifing_intent")
    builder.add_conditional_edges(
        "first_node_classifing_intent",
        router
    )
    builder.add_edge("weather_node", END)
    builder.add_edge("handle_rag_query", END)
    
    return builder.compile()

# print(graph.get_graph().draw_mermaid())
#display(Image(graph.get_graph().draw_mermaid_png()))

memory = initialize_memory()

def main():
    print("Starting main function")
    
    try:
        graph = build_graph()
        print("Graph built successfully")
        
        memory = initialize_memory()
        print("Memory initialized")
        
        print("Weather Chatbot ready. Ask about weather or type 'exit' to quit.")
        
        history = []
        while (query := input("\nYou: ").strip().lower()) != "exit":
            history.append(HumanMessage(content=query))
            print(f"Invoking graph with message: {query}")
            
            try:
                result = graph.invoke({"messages": history, "intent": "", "city": "", "memory": memory})
                reply = result["messages"][-1]
                history.append(reply)
                print(f"\n Bot: {reply.content}")
            except Exception as e:
                print(f"Error invoking graph: {e}")
                print("Continuing with next query...")
        
        print("Goodbye!")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()


