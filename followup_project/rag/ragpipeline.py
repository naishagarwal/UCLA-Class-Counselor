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
base_dir = os.path.dirname(os.path.realpath('config.yaml'))
with open(base_dir + '/followup_project/rag/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

os.environ["LANGCHAIN_TRACING_V2"] = "true"

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

    #retrieve relevant documents
    retrieval_result = es_retriever({"question": full_context})
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


