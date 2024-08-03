'''
    Steps:
        1. Connect the elastic search database using Langchain
        2. Construct the RAG pipeline to generate results

        Make sure to test steps periodically

'''

import getpass
import os
import yaml
from langchain_openai import ChatOpenAI
from elasticsearchclass import ElasticsearchRetriever
#from langchain.chains import RetrievalAugmentedGeneration
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



#importing environment variables
with open("rag/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config['LANGCHAIN_API_KEY']
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

from langchain_openai import ChatOpenAI

#llm (using gpt-4o-mini for now, can change to any other model as well)
llm = ChatOpenAI(model="gpt-4o-mini")
#print(llm.invoke('Hello')) testing to see if llm works

#initializing elastic search retriever
es_retriever = ElasticsearchRetriever(
    es_host = config['es_host'],
    es_username = config['es_username'],
    es_password = config['es_password'],
    index_name = 'f_24_embedding',
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

#set up rag pipeline
# rag_chain = (
#      "context": es_retriever 
#     | "question": RunnablePassthrough()
#     | create_prompt
#     | llm
#     | StrOutputParser()
# )
# rag_chain = (
#     {
#         "context": es_retriever | prompt_template, #combines elastic search retriever and prompt template
#         "question": RunnablePassthrough(), #forwarding query
#         "prompt": prompt_template, #using prompt template to format query
#         "llm": llm, #calls llm with formatted query
#         "output_parser": StrOutputParser() #parses llm output into string format
#     }
# )

# def generate_response(query: str) -> str:
#     response = rag_chain.invoke({"question": query})
#     return response

def generate_response(query: str) -> str:
    #retrieve relevant documents
    retrieval_result = es_retriever({"question": query})
    context = retrieval_result["context"]

    #create prompt
    prompt = create_prompt({"question": query, "context": context})

    print(prompt)

    #get LLM response
    llm_response = llm(prompt)

    #parse response
    parsed_response = StrOutputParser().parse(llm_response)

    return parsed_response

if __name__ == "__main__":
    query = "What day and time is Computer Science 263 - Natural Language Processing taught?"
    response = generate_response(query)
    print(response)


