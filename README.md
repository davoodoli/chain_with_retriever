# 🔍 LangChain + FAISS + OpenAI Retrieval QA Example

This project demonstrates how to build a **context-aware question-answering system** using:

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [OpenAI](https://platform.openai.com/) for embeddings and chat completion
- Local file as the knowledge base
- Custom prompt for focused responses

---

## 📁 What It Does

✅ Loads a local `.txt` file  
✅ Splits it into chunks  
✅ Embeds the chunks using OpenAI  
✅ Stores them in a FAISS vectorstore  
✅ Retrieves relevant chunks for a given user query  
✅ Feeds them into a custom prompt + LLM  
✅ Returns an answer based only on the provided context
