
from rag.text_converter import convert_logs_to_text

texts = convert_logs_to_text("data/logs/CAM_01.json")

for t in texts[:10]:
    print(t)
