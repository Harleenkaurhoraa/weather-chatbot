# Weather Chatbot

A conversational AI assistant that provides weather information and answers general weather-related questions.

## Overview

This project implements a weather chatbot that can:
1. Provide real-time weather information for specific cities
2. Answer general questions about weather, climate patterns, and safety measures
3. Use a combination of API calls and Retrieval-Augmented Generation (RAG) to provide accurate information

## Components

### 1. `chatbot.py`

The main executable script that:
- Initializes the conversational agent
- Builds a graph-based conversation flow
- Handles user input and generates responses
- Routes queries to appropriate handlers based on intent classification

### 2. `weather_chatbot_api.ipynb`

A Jupyter notebook that contains:
- Development and testing of the chatbot functionality
- Implementation of the LangGraph conversation flow
- Integration with weather APIs
- RAG implementation for general weather knowledge

## Features

- **Intent Classification**: Automatically determines if the user is asking about current weather conditions or general weather information
- **City Extraction**: Identifies city names in user queries for weather lookups
- **Weather API Integration**: Connects to WeatherAPI.com to fetch real-time weather data
- **Retrieval-Augmented Generation**: Uses a FAISS vector database of weather information to provide accurate answers to general questions
- **Conversation Memory**: Maintains context throughout the conversation

## Technologies Used

- **LangChain & LangGraph**: For building the conversational flow and RAG system
- **FAISS**: For vector storage and similarity search
- **CyfutureAI API**: For LLM capabilities
- **WeatherAPI.com**: For real-time weather data
- **Python-dotenv**: For environment variable management

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   CYFUTURE_API_KEY=your_cyfuture_api_key
   WEATHER_API_KEY=your_weather_api_key
   BASE_URL=https://api.weatherapi.com/v1/current.json?
   ```
4. Ensure you have Ollama installed for embeddings generation

## Usage

Run the chatbot using:
```
python chatbot.py
```

Example interactions:
- "What's the weather like in London today?"
- "Tell me about hurricanes and how they form"
- "What should I do during a thunderstorm?"

## Data Sources

The chatbot uses a combination of:
- Real-time data from WeatherAPI.com
- Pre-processed PDF documents containing weather information (stored in the `data_pdfs` directory)

## Project Structure

```
weather-chatbot/
├── chatbot.py                # Main executable script
├── weather_chatbot_api.ipynb # Development notebook
├── requirements.txt          # Project dependencies
├── README.md                 # This file
├── .env                      # Environment variables (not tracked in git)
├── faiss_index/              # Vector database for RAG
└── data_pdfs/                # PDF documents for knowledge base
```

## Future Improvements

- Add support for weather forecasts (multi-day predictions)
- Implement more detailed climate data analysis
- Add location detection based on user's IP address
- Expand the knowledge base with more weather-related information
- Add visualization of weather data
