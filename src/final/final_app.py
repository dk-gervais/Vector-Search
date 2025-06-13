# Import the Python libaries that will be used for this app.
# Libraries of note: 
# Streamlit, a Python library that makes it easy to create and share beautiful, custom web apps for data science and machine learning.
# ChatOpenAI, a class that provides a simple interface to interact with OpenAI's models.
# ConversationChain and ConversationSummaryMemory, classes that represents a conversation between a user and an AI and retain the context of a conversation.
# OpenAIEmbeddings, a class that provides a way to perform vector embeddings using OpenAI's embeddings.
# IRISVector, a class that provides a way to interact with the IRIS vector store.
import streamlit as st
from langchain_openai import ChatOpenAI 
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_iris import IRISVector
import os

# Import dotenv, a module that provides a way to read environment variable files, and load the dotenv (.env) file that provides a few variables we need
from dotenv import load_dotenv
load_dotenv(override=True)

# Load the urlextractor, a module that extracts URLs and will enable us to follow web-links
from urlextract import URLExtract
extractor = URLExtract()

# Define the IRIS connection - the username, password, hostname, port, and namespace for the IRIS connection.
username = os.environ['IRIS_USER']
password = os.environ['IRIS_PASSWORD']
hostname = 'localhost'
port = 1972
namespace = 'USER'  # This is the namespace for the IRIS connection

# Create the connection string for the IRIS connection
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

# Create an instance of OpenAIEmbeddings, a class that provides a way to perform vector embeddings using OpenAI's embeddings.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(
        temperature=0.0,
        model_name='gpt-4-turbo',
    )

# *** Instantiate IRISVector ***

# Define the name of the finance collection in the IRIS vector store.
FINANCE_COLLECTION_NAME = "financial_tweets"
# Create an instance of IRISVector.
db2 = IRISVector(
    # The embedding function to use for the vector embeddings.
    embedding_function=embeddings,
    # The name of the collection in the IRIS vector store.
    collection_name=FINANCE_COLLECTION_NAME,
    # The connection string to use for connecting to the IRIS vector store.
    connection_string=CONNECTION_STRING,
)

### Used to have a starting message in our application
# Check if the "messages" key exists in the Streamlit session state.
# If it doesn't exist, create a new list and assign it to the "messages" key.
if "messages" not in st.session_state:
    # Initialize the "messages" list with a welcome message from the assistant.
    st.session_state["messages"] = [
        # The role of this message is "assistant", and the content is a welcome message.
        {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
    ]

if "conversation_sum" not in st.session_state:
    st.session_state["conversation_sum"] = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm),
        verbose=True,
    )

# Add a title for the application
# This line creates a header in the Streamlit application with the title "GS 2024 Vector Search"
st.header('GS 2024 Vector Search')

# Customize the UI
# In streamlit we can add settings using the st.sidebar
with st.sidebar:
    st.header('Settings')
    # Allow user to toggle which model is being used (gpt-4 in this workshop)
    choose_LM = "gpt-4-turbo"
    # Allow user to toggle whether explanation is shown with responses
    # explain = st.radio("Show explanation?:",("Yes", "No"),index=0)
    # link_retrieval = st.radio("Retrieve Links?:",("No","Yes"),index=0)

# In streamlet, we can add our messages to the user screen by listening to our session
for msg in st.session_state['messages']:
    # If the "chat" is coming from AI, we write the content with the ISC logo
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    # If the "chat" is the user, we write the content as the user image, and replace some strings the UI doesn't like
    else:
        st.chat_message(msg["role"]).write(msg["content"].replace("$", "\$"))

# Check if the user has entered a prompt (input) in the chat window
if prompt := st.chat_input(): 

    # Add the user's input to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's input in the chat window, escaping any '$' characters
    st.chat_message("user").write(prompt.replace("$", "\$"))

    # Here we respond to the user based on the messages they receive 
    with st.chat_message("assistant"):
        docs_with_score = db2.similarity_search_with_score(prompt)
        relevant_docs = ["".join(str(doc.page_content)) + " " for doc, _ in docs_with_score]
        conversation_history = st.session_state["conversation_sum"].memory.load_memory_variables({})['history']
        prompt = f"""
            You should only make use of the provided Relevant Documents. They contain important information
            belonging to the user, and it is critical that any advice you give is grounded in these documents.
            If the documents provided are irrelevant to the question, simply state that you do not have the
            relevant information available in the database.
            
            Prompt: {prompt}

            Conversation History: {conversation_history}

            Relevant Documents: {relevant_docs}
            """
        # Invoke our LLM, passing the user prompt directly to the model
        resp = st.session_state["conversation_sum"].predict(input=prompt)
        
        # Finally, we make sure that if the user didn't put anything or cleared session, we reset the page
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
            ]

        # And we add to the session state the message history
        st.session_state.messages.append({"role": "assistant", "content": resp})
        print(resp)
        # And we also add the response from the AI
        st.write(resp.replace("$", "\$"))
        with st.expander("Supporting Evidence"):
                for doc, _ in docs_with_score[:1]:
                    doc_content = "".join(str(doc.page_content))
                    # st.write(f"""Here are the relevant documents""")
                    st.write(f"""{doc_content}""")
                    urls = extractor.find_urls(doc_content)
                    print(urls)  # prints: ['stackoverflow.com']
                    for url in urls:
                        st.page_link(url, label="Source")
