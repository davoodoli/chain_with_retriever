from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

#loading  the documents
loader = TextLoader('note.txt')
documents = loader.load()

#splitting the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
docs = text_splitter.split_documents(documents=documents)

#create the embedding and the llm model
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")

#create the vectorstore and make it's retriver
vectorstoredb = FAISS.from_documents(documents=docs,embedding=embedding)
retriever = vectorstoredb.as_retriever()

#prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the following question just based on the provided context:
    question : {input}
    <context>
    {context}
    </context>
    """
)

#create the chain and the retrieval chain
document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt_template)
retrieval_chain = create_retrieval_chain(retriever,document_chain)

# Retrieve relevant documents
question = "Where can be directed the impact of AI on culture?"
relevant_docs = retriever.get_relevant_documents(question)


# Extract text from retrieved documents
context = "\n".join([doc.page_content for doc in relevant_docs])

#example
response = retrieval_chain.invoke({"input":question,"context":context})
print(response['answer'])
