from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import os
import re


loader = TextLoader("data.txt")
docs=loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 100
)
chunks = splitter.split_documents(docs)
print(len(chunks))


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)



vectorstore = FAISS.from_documents(chunks,embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt = ChatPromptTemplate.from_messages([
    ("system","Answer ONLY using the provided context."),
    ("human","Context:\n{context}\nQuestion:\n{question}")
])


model = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    openai_api_key=os.environ.get("TOGETHER_API_KEY"),
    model="mistralai/Mistral-7B-Instruct-v0.3"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context" : retriever | format_docs,
        "question" : RunnablePassthrough(),
    }
    | prompt
    | model 
    | StrOutputParser()
)


question = "What was the aggregate market value of voting stock in 2022?"

retrieved_docs = retriever.invoke(question)
context_text = format_docs(retrieved_docs)
answer = rag_chain.invoke(question)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def context_adherence(answer, context):
    """
    Percentage of answer tokens found in context
    """
    a_tokens = set(tokenize(answer))
    c_tokens = set(tokenize(context))
    if not a_tokens:
        return 0.0
    return len(a_tokens & c_tokens) / len(a_tokens)


def completeness(answer, context):
    """
    How much of the context is reflected in the answer
    """
    a_tokens = set(tokenize(answer))
    c_tokens = set(tokenize(context))
    if not c_tokens:
        return 0.0
    return len(a_tokens & c_tokens) / len(c_tokens)


def chunk_attribution(answer, chunks):
    """
    Boolean per chunk:
    Was chunk used at all?
    """
    a_tokens = set(tokenize(answer))
    results = []
    for chunk in chunks:
        c_tokens = set(tokenize(chunk.page_content))
        results.append(len(a_tokens & c_tokens) > 0)
    return results


def chunk_utilization(answer, chunks):
    """
    Fraction of chunk tokens used in answer
    """
    a_tokens = set(tokenize(answer))
    utilizations = []
    for chunk in chunks:
        c_tokens = set(tokenize(chunk.page_content))
        if not c_tokens:
            utilizations.append(0.0)
        else:
            utilizations.append(len(a_tokens & c_tokens) / len(c_tokens))
    return utilizations


# ==============================
# 9. COMPUTE METRICS
# ==============================

context_score = context_adherence(answer, context_text)
completeness_score = completeness(answer, context_text)

attribution = chunk_attribution(answer, retrieved_docs)
utilization = chunk_utilization(answer, retrieved_docs)


# ==============================
# 10. PRINT RESULTS
# ==============================

print("\n====================")
print("QUESTION")
print("====================")
print(question)

print("\n====================")
print("ANSWER")
print("====================")
print(answer)

print("\n====================")
print("RAG METRICS")
print("====================")

print(f"Context Adherence: {context_score:.2f}")
print(f"Completeness:      {completeness_score:.2f}")

print("\nChunk Attribution:")
for i, used in enumerate(attribution):
    print(f"  Chunk {i+1}: {'USED' if used else 'NOT USED'}")

print("\nChunk Utilization:")
for i, score in enumerate(utilization):
    print(f"  Chunk {i+1}: {score:.2f}")

print("\n====================")
print("RETRIEVED CHUNKS")
print("====================")

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:400])
