# Retrieval-augmented generation demo on Amazon EKS

This repository deploys a chatbot on Amazon EKS. The chatbot uses an LLM with retrieval-augmented generation to answer user queries about BMW X5 car. 

Using Langchain, the chatbot parses the car's user manual and stores index in FAISS. The UI is built using Gradio. 
The current demo uses vilsonrodrigues/falcon-7b-instruct-sharded model from Huggingface
