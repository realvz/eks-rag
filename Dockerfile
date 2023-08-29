FROM ghcr.io/huggingface/text-generation-inference:0.9.2
RUN pip install faiss-cpu --no-cache-dir 
RUN pip install langchain --upgrade --no-cache-dir 
RUN pip install pypdf --no-cache-dir 
RUN pip install sentence_transformers --no-cache-dir 
RUN pip install xformers --no-cache-dir 
RUN pip install gradio --no-cache-dir 
RUN mkdir /data
WORKDIR /data
RUN mkdir /data/pdfs
COPY model.py .
