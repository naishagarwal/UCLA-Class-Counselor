import getpass
import os
import yaml
from langchain_openai import ChatOpenAI
from elasticsearchclass import ElasticsearchRetriever
#from langchain.chains import RetrievalAugmentedGeneration
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
from gradio_ui.gradio_ui import create_ui

#importing environment variables
with open("rag/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config['LANGCHAIN_API_KEY']
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

from langchain_openai import ChatOpenAI

#llm (using gpt-4o-mini for now, can change to any other model as well)
llm = ChatOpenAI(model="gpt-4o-mini")
#instead of a standard LLM, can replace this with a fine-tuned one
#print(llm.invoke('Hello')) testing to see if llm works

#initializing elastic search retriever
es_retriever = ElasticsearchRetriever(
    es_host = config['es_host'],
    es_username = config['es_username'],
    es_password = config['es_password'],
    indices = ['f_24_embedding', 'professor_info_embedding'],
    ca_cert_path = config['ca_cert_path']
)

#set up prompt template
prompt_template = PromptTemplate(template = "Q: {question}\nContext: {context}\nA:")

def create_prompt(inputs):
    question = inputs["question"]
    context = inputs["context"]
    # print(question)
    # print(context)
    return prompt_template.format(question=question, context=context)

def create_history(history):
    conversation_history = ""
    for turn in history:
        conversation_history += f"User: {turn[0]}\nAI: {turn[1]}]\n"
    return conversation_history

def clear_history():
    return [], []

def generate_response(query: str, history: list) -> str:

    #get conversation history
    conv_history = create_history(history)

    #full context including history
    full_context = f"{conv_history} User: {query}\n"

    #pass full_context into LLM to generate a new query, then pass that query into database to get appropriate results
    question = "Given the following conversation history and user query, generate a new query that appropriately ties in the conversation history."

    history_prompt = create_prompt({"question": question, "context": full_context})
    new_query = llm(history_prompt).content
    print(new_query)

    #retrieve relevant documents
    #Change full_context here to be new_query
    retrieval_result = es_retriever({"question": new_query})
    #history_result = es_retriever({"question:": conv_history})
    context = retrieval_result["context"]

    # print("History")
    # print(conv_history)

    #create prompt
    prompt = create_prompt({"question": full_context, "context": context})

    print(prompt)

    #get LLM response
    llm_response = llm(prompt)

    #parse out content portion of LLM response 
    parsed_response = llm_response.content

    #update history
    history.append([query, parsed_response])

    return history, history

if __name__ == "__main__":
    ui = create_ui(generate_response, clear_history) 
    ui.launch(share = True)


