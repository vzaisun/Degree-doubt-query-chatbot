from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk():
    with open("ucsd_docs.txt", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks
