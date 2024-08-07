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

    #content
    #parsed_response = llm_response['choices'][0]['message']['content']

    #parse response
    #parsed_response = StrOutputParser().parse(llm_response)

    parsed_response = llm_response.content

    return parsed_response

if __name__ == "__main__":
    # query = "Hi! I am a computer science major just entering UCLA. Can you recommend me some computer science classes I can take?"
    # response = generate_response(query)
    # print(response)

    # gr.ChatInterface(generate_response,
    # chatbot=gr.Chatbot(height=300),
    # textbox=gr.Textbox(placeholder="You can ask me anything", container=False, scale=7),
    # title="UCLA Class Counselor",
    # retry_btn=None,
    # undo_btn="Delete Previous",
    # clear_btn="Clear").launch()

    gr.Interface(
        fn=generate_response,
        inputs=gr.Textbox(placeholder="You can ask me anything", lines=2),
        outputs=gr.Textbox(),
        title="UCLA Class Counselor"
    ).launch()

    #gr.ChatInterface(generate_response).launch()


