import os
import fitz
import boto3
import hashlib
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

# ✅ Set environment variables or pass directly

aws_region = "us-east-1"
vector_bucket_name = "s3-vectors-bucket-2025"
vector_index_name = "rag-index-2025"

# === Load PDF ===
def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

pdf_text = load_pdf_text("sample.pdf")
print(pdf_text)
print("✅ PDF loaded")

# === Split Text ===
text_splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    model_name="gpt-3.5-turbo"
)
documents = text_splitter.create_documents([pdf_text])
print(documents)
print(f"✅ PDF split into {len(documents)} chunks")

# === Generate Embeddings ===
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]
embeddings = embedding_model.embed_documents(texts)
# print(embeddings)
print(f"✅ Generated {len(embeddings)} embeddings")

# === S3 Vectors Client ===
client = boto3.client("s3vectors", region_name=aws_region)

# === Create Vector Bucket === (Run Once)
try:
    client.create_vector_bucket(vectorBucketName=vector_bucket_name)
    print("✅ Vector bucket created")
except client.exceptions.ConflictException:
    print("ℹ️ Vector bucket already exists, skipping creation")

# === Insert Vectors into Index (this will create index if not exists) ===
vectors_to_insert = []
for i, embedding in enumerate(embeddings):
    key = hashlib.md5(texts[i].encode()).hexdigest()
    vectors_to_insert.append({
        "key": key,  # Unique key
        "data": { "float32": embedding },  # Embedding vector values
        "metadata": { "source_text": texts[i] }  # Your metadata
    })

# Batch insert (max 1000 per batch)
print(f"Total data to insert is {len(vectors_to_insert)}")
batch_size = 100
for i in range(0, len(vectors_to_insert), batch_size):
    batch = vectors_to_insert[i:i+batch_size]
    try:
        result = client.put_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=vector_index_name,
            vectors=batch
        )
        print(f"Insertion Result is {result}")
        print(f"✅ Inserted batch {i//batch_size + 1}")
    except Exception as e:
        print(f"Error in creating vectors:{e}")


# === Query Similar Vectors ===
query_text = "What is mentioned about generative AI?"
query_embedding = embedding_model.embed_query(query_text)

response = client.query_vectors(
    vectorBucketName=vector_bucket_name,
    indexName=vector_index_name,
    queryVector={"float32": query_embedding},
    topK=5,
    returnMetadata=True,
    returnDistance=True
)
# Debug: Print full response
import json
print("QueryVectors Response:\n", json.dumps(response, indent=4))

# Safe check for Matches
matches = response.get("vectors", [])

if matches:
    retrieved_chunks = [item["metadata"]["source_text"] for item in matches]
    print(f"✅ Retrieved {len(retrieved_chunks)} matching chunks")
else:
    print("⚠️ No matching vectors found or query failed.")
    retrieved_chunks = []

# === Compose Context and Ask LLM ===
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

context = "\n\n".join(retrieved_chunks)
final_prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"

llm_response = llm.invoke(final_prompt)
print("\n✅ Answer:\n", llm_response)
