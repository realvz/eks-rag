FROM ghcr.io/huggingface/text-generation-inference:0.9.2
RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get -y install \
     wget

RUN pip install faiss-cpu --no-cache-dir 
RUN pip install langchain --upgrade --no-cache-dir 
RUN pip install pypdf --no-cache-dir 
RUN pip install sentence_transformers --no-cache-dir 
RUN pip install xformers --no-cache-dir 
RUN pip install gradio --no-cache-dir 
RUN mkdir /data
WORKDIR /data
RUN mkdir -p /data/pdfs/user_manuals
RUN wget https://gimmemanuals.com/owners/2022/01/2022-bmw-x5-owners-manual.pdf -P pdfs/user_manuals
COPY chatbot.py .
