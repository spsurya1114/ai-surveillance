
from rag.text_converter import convert_logs_to_text
from rag.embedder import RAGEmbedder
from rag.llm import ask_llm

# Load logs → text
texts = convert_logs_to_text("data/logs/CAM_01.json")

# Build vector index
rag = RAGEmbedder()
rag.build_index(texts)

# Ask query
query = "Where did person 1 go?"

# Retrieve relevant context
results = rag.search(query)
context = "\n".join(results)

# Ask LLM
answer = ask_llm(query, context)

print("\nQuery:", query)
print("\nContext:")
print(context)

print("\nAnswer:")
print(answer)

