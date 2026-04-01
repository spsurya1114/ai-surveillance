
from rag.text_converter import convert_logs_to_text
from rag.embedder import RAGEmbedder

# Load text
texts = convert_logs_to_text("data/logs/CAM_01.json")

# Build index
rag = RAGEmbedder()
rag.build_index(texts)

# Test query
query = "Where did person 1 go?"

results = rag.search(query)

print("\nQuery:", query)
print("\nTop Results:")

for r in results:
    print("-", r)