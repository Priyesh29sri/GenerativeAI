from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
import streamlit as st

System_Message = ("""You are an Indian patriot and have love for the contry India.
                   you answer in patriotic and loving way.
                  you dont answer any question not realated to India.""")

llm = OllamaLLM(
    model="mistral",
    temperature = 0.7,
    verbose = True, #Logs prompts and outputs step-by-step	Debugging, learning, transparency
    Streaming = True # Shows tokens in real-time as generated	For responsiveness, chat UIs
)

memory = ConversationBufferMemory(k=3)
conversation = ConversationChain(
    llm = llm,
    memory = memory
)
memory.chat_memory.add_message(SystemMessage(content = System_Message))

response = conversation.predict(input = "what is color of Indian Flag?")
print(response)

response = conversation.predict(input= "who intevented this flag? ")
print(response)

st.header("Chatbot")
st.subheader("Get answers in a proud and patriotic tone!")

question = st.text_input("Enter your question about India")

if st.button("Generate"):
    if question.strip():
        response = conversation.predict(input=question)
        st.markdown("Response:")
        st.success(response)
    else:
        st.warning("Please enter a valid question.")

