# Course RAG

When I was teaching, one student called ChatGPT their north star.  This bugged me because ChatGPT sometimes gave them bad answers (something I, of course, would never do).  So I decided to create a tool that would allow students to look at the course materials first, then an LLM.

## How do I use it?
Update the file paths, and get an API key from together.ai

## What needs to be done?
Need to turn it into a Flask app.  Also edit the LLM/embeddings model to make switching from one model to another seamless.

## What is your question/purpose?

To create an instance of Retrival Augmented Generation (RAG) that shows you where an answer came from.  This is superior to standard search because it allows natural language queries instead of simple 

## Why does it matter?

You can use this tool to ask natural questions from a folder rather than just doing a keyword search

## Why is it hard?

The open source tools change rapidly and getting a good combination of embedding model and LLM isn't trivial.

## What did I do?

1. It takes in files from a directory.  The current version takes in powerpoints, but it can be modified to take in PDFs too.
2. The PDFs are split into embeddings (loosely smaller chunks) via an embeddings model and stored locally. 
3. When you ask it a query, the program searches in your documents then the LLM.  
4. When you get an answer, you get the top 5 sources where the answers were pulled from

## What did I learn?

RAGs are cool.  
