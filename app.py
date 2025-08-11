import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub

st.set_page_config(page_title='Search Engine on Wiki,Arxiv, langchain & DuckDuckGO')
# st.subheader('')
# Search tools
with st.spinner('Setting up .........'):
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=4000)
    api_wrapper_wikipedia = WikidataAPIWrapper(top_k_results=1,doc_content_chars_max=4000)

    wiki_tool=WikidataQueryRun(api_wrapper=api_wrapper_wikipedia)
    arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
    search=DuckDuckGoSearchRun(name='search')

    loader=WebBaseLoader('https://docs.langchain.com/')
    docs=loader.load()

    documents=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb=FAISS.from_documents(documents, embeddings)

    retriever=vectordb.as_retriever()

    #retriever to tool
    retrieve_tool=create_retriever_tool(retriever,'langchain-search','search any ifo related to langchain')

    tools=[wiki_tool,arxiv_tool,retrieve_tool,search]
    

with st.sidebar:
    groq_api_key=st.sidebar.text_input('Enter GROQ API key',type='password')

query=st.text_input('Enter your question')

if st.button('Search'):
    if not groq_api_key.strip() or not query.strip():
        st.error('Please Provide the information')
    else:
        llm=ChatGroq(groq_api_key=groq_api_key,model='gemma2-9b-it')

        #prompt template
        # prompt_template='''You are a best chatbot. Based on the query i want you to provide the detailed answer with minimum output word of 1000'''
        prompt=hub.pull('hwchase17/openai-functions-agent')

        #chain
        agent=create_openai_tools_agent(llm,tools,prompt)

        #executor

        agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

        answer=agent_executor.invoke({'input':query})
        st.write(answer['output'])
        
        
