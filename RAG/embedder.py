from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz

class Embidder():
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chuncks=  []

    def read_pdf_chuncks(self,path,chunck_size=300):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()

        text = text.replace('\n',' ').strip()
        

        for i in range(0,len(text),chunck_size):
            self.chuncks.append(text[i:chunck_size+i])     
        

    def embed_index_doc(self):
        # Embed documents
        doc_embeddings = self.embedder.encode(self.chuncks, convert_to_numpy=True)

        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(doc_embeddings)
        

    def retrieve_context(self,question, k=2):
        question_embedding = self.embedder.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(question_embedding, k)

        return [self.chuncks[i] for i in indices[0]]
    

    def add_pdf(self,path,chunck_size=300):
        self.read_pdf_chuncks(path,chunck_size)
        self.embed_index_doc()
         