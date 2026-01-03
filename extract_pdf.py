from pypdf import PdfReader

reader = PdfReader("Devoir XAI 25-26.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("instructions.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Done writing to instructions.txt")
