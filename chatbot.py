import os
import pathlib
import regex as re
import gradio as gr
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline

# Modularizing Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
def setup_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        keep_separator=False,
        add_start_index=False
    )

# Modularizing Document Loading
def load_documents(directory):
    pdf_documents = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    langchain_documents = []
    for document in pdf_documents:
        loader = PyPDFLoader(document)
        data = loader.load()
        langchain_documents.extend(data)
    return langchain_documents

# Modularizing Text Preprocessing
def preprocess_text(text):
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r'[\\X]', "", text)
    text = re.sub(r"(\\u[0-9A-Fa-f]+)", " ", text)
    return text

# Modularizing Indexing
def setup_index(docs, emb, index_name, index_path):
    db = FAISS.from_documents(docs, embedding=emb)
    pathlib.Path(index_path).mkdir(parents=True, exist_ok=True)
    db.save_local(folder_path=index_path, index_name=index_name)
    db_local = FAISS.load_local(folder_path=index_path, embeddings=emb, index_name=index_name)
    return db_local

# Modularizing Pipeline and Prompt
def setup_pipeline_and_prompt(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipeline_obj = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_template = """Human: Pretend you are bot that answers questions about the BMW X5 car. Use the following pieces of context from the car's user manual to provide a concise answer to the question at the end.

    {context}

    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return pipeline_obj, PROMPT

# Modularizing Chatbot Initialization
def setup_chatbot(llm, db_local, chain_type):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=db_local.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    #return qa

# Modularizing User and Bot Message Handling
def handle_messages(user_message, history, qa):
    response = qa({'query': user_message})
    bot_message = response['result']
    history[-1][1] = ""
    history[-1][1] += bot_message
    return history

# Main function
def main():
    text_splitter = setup_text_splitter()
    index_name = 'user_manuals'
    directory = '/data/pdfs/' + index_name
    langchain_documents = load_documents(directory)
    
    split_docs = text_splitter.split_documents(langchain_documents)

    for d in split_docs:
        d.page_content = preprocess_text(d.page_content)

    emb = HuggingFaceEmbeddings()
    index_path = '/data/faiss/faiss_indices'
    db_local = setup_index(split_docs, emb, index_name, index_path)

    model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
    pipeline_obj, PROMPT = setup_pipeline_and_prompt(model_id)

    llm = HuggingFacePipeline(pipeline=pipeline_obj)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db_local.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        #llm_chain, llm = init_chain(model, tokenizer)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            print("Question: ", history[-1][0])
            #bot_message = chain.run(input_documents=documents,question=history[-1][0])
            response = qa({'query':history[-1][0]})
            bot_message = response['result']
            print("Response: ", bot_message)
            history[-1][1] = ""
            history[-1][1] += bot_message
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(share=True)
if __name__ == "__main__":
    main()

