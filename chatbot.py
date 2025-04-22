import os
from typing import TypedDict, Annotated
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

# # Load environment variables
load_dotenv()
CYFUTURE_API_KEY = os.getenv("CYFUTURE_API_KEY")
# WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") 
# BASE_URL = os.getenv("BASE_URL")

# load_dotenv()  # Load the .env file

# print("Weather API Key:", WEATHER_API_KEY)  # Check if the key is being loaded
# print("Base URL:", BASE_URL)  # Check if the URL is being loaded


# Define state with window buffer memory
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    city: str
    memory: ConversationBufferMemory

def initialize_memory():
    return ConversationBufferMemory(
        k=5,  # Remember last 5 interactions
        memory_key="chat_history",
        return_messages=True
    )

def ask_ai(question: str) -> str:
    """Simple function to ask CyfutureAI a question"""
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
        print(f"Error in ask_ai: {e}")
        return "Sorry, I couldn't process that request."

def first_node_classifing_intent(state):
    """Classify intent and extract city"""
    last_message = state["messages"][-1].content
    
    # For debugging
    print(f"Processing message: {last_message}")
    
    prompt = f"""
    Determine the intent of the following message:
    "{last_message}"
    
    If the message is asking about current weather, weather forecast, or weather conditions for a specific location, respond with "weather_api".
    If the message is asking about general weather information, climate patterns, or safety measures, respond with "rag".
    
    Just respond with either "weather_api" or "rag".
    """
    
    intent_response = ask_ai(prompt)
    intent = intent_response.strip().lower()
    print(f"Detected intent: {intent}")
    
    city = "unknown"
    if intent == "weather_api":
        city_prompt = f"""
        Extract the city name from this message: "{last_message}"
        If no city is mentioned, respond with "unknown".
        Just provide the city name or "unknown", nothing else.
        """
        city_response = ask_ai(city_prompt)
        city = city_response.strip().lower()
        print(f"Detected city: {city}")
    
    # Save context in memory
    if "memory" in state:
        state["memory"].save_context(
            {"input": last_message},
            {"output": f"Intent: {intent}, City: {city}"}
        )
    
    return {"intent": intent, "city": city}

BASE_URL="https://api.weatherapi.com/v1/current.json?"
WEATHER_API_KEY="93e24c7b0ea440c78c7185341252004"
print("API key:", WEATHER_API_KEY)
def get_weather_forecast(city: str):
    """Returns current weather forecast for a city using WeatherAPI.com."""
    print(f"Getting weather for: {city}")
    
    try:
        params = {
            'q': city,
            'key': WEATHER_API_KEY  # WeatherAPI uses 'key' not 'appid'
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        print(f"Weather API response status: {response.status_code}")
        
        # WeatherAPI.com has a different response structure
        return {
            "location": data["location"]["name"],
            "description": data["current"]["condition"]["text"],
            "temperature_celsius": data["current"]["temp_c"]
        }
    except Exception as e:
        print(f"Error getting weather: {e}")
        return {
            "location": city,
            "description": "unavailable",
            "temperature_celsius": 0
        }

def weather_node(state):
    """Handle weather forecast requests"""
    city = state["city"]
    
    if city == "unknown":
        response = "I need a city name to check the weather."
    else:
        try:
            weather_data = get_weather_forecast(city)
            response = f"In {weather_data['location']}, it's currently {weather_data['description']} with a temperature of {weather_data['temperature_celsius']}°C."
        except Exception as e:
            print(f"Error in weather_node: {e}")
            response = f"Sorry, I couldn't get the weather for {city}."
    
    return {"messages": [AIMessage(content=response)]}

def handle_rag_query(state):
    """Handle general weather information queries using RAG"""
    last_message = state["messages"][-1].content
    prompt = f" You are a helpful weather information assistantAnswer this question about weather or climate: '{last_message}'. Provide factual, scientific information.If you don't know the answer, just say that you don't know"
    response = ask_ai(prompt)
    
    return {"messages": [AIMessage(content=response)]}

def router(state):
    """Route to the appropriate node based on intent"""
    if state["intent"] == "weather_api":
        print("Routing to weather_node")
        return "weather_node"
    else:
        print("Routing to handle_rag_query")
        return "handle_rag_query"

def build_graph():
    """Build and return the graph"""
    print("Building the graph...")
    
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("first_node_classifing_intent", first_node_classifing_intent)
    builder.add_node("weather_node", weather_node)
    builder.add_node("handle_rag_query", handle_rag_query)
    
    # Add edges
    builder.add_edge(START, "first_node_classifing_intent")
    builder.add_conditional_edges(
        "first_node_classifing_intent",
        router
    )
    builder.add_edge("weather_node", END)
    builder.add_edge("handle_rag_query", END)
    
    print("Graph built successfully, compiling...")
    return builder.compile()

def main():
    # Add debug prints
    print("Starting main function")
    
    try:
        graph = build_graph()
        print("Graph built successfully")
        
        memory = initialize_memory()
        print("Memory initialized")
        
        print("🤖 Weather Chatbot ready. Ask about weather or type 'exit' to quit.")
        
        history = []
        while (query := input("\nYou: ").strip().lower()) != "exit":
            history.append(HumanMessage(content=query))
            print(f"Invoking graph with message: {query}")
            
            try:
                result = graph.invoke({"messages": history, "intent": "", "city": "", "memory": memory})
                reply = result["messages"][-1]
                history.append(reply)
                print(f"\n🤖 Bot: {reply.content}")
            except Exception as e:
                print(f"Error invoking graph: {e}")
                print("Continuing with next query...")
        
        print("🤖 Goodbye!")
    except Exception as e:
        print(f"Error in main function: {e}")

# Make sure to use double underscores
if __name__ == "__main__":
    main()