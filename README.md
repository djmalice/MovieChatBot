# 🎬 RAG-Based Movie Recommendation Q&A System  

This project builds a **Retrieval-Augmented Generation (RAG)** system for **movie recommendations**, leveraging **FAISS** for efficient similarity search and deploying the solution via **Hugging Face** using **Chainlit** for an interactive experience.  

🔗 **Live Demo:** [MovieChatBot on Hugging Face](https://huggingface.co/spaces/chintansheth87/MovieChatBot)  

## 🚀 Features  

- **Question-Answering System**: Provides movie recommendations based on natural language queries.  
- **Vector Search with FAISS**: Uses Facebook AI Similarity Search (FAISS) for efficient retrieval of relevant movies.  
- **Hugging Face Deployment**: Hosted and accessible via Hugging Face Spaces.  
- **Interactive UI with Chainlit**: Enables an intuitive and conversational experience for users.  
- **OpenAI API Calls with LangChain Runnable**: Uses LangChain's runnable framework to make API calls to OpenAI for enhanced responses.  

## 🛠️ Tech Stack  

- **FAISS**: Fast retrieval of semantically similar movie embeddings.  
- **Hugging Face**: Model hosting and inference.  
- **Chainlit**: Frontend for conversational interactions.  
- **Python**: Backend implementation.  
- **LangChain Runnable**: Enables structured OpenAI API calls for generating responses.  

## 🔧 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/rag-movie-recommendation.git  
   cd rag-movie-recommendation  
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  
3. Run the application locally:  
   ```bash
   chainlit run app.py  
   ```  

## 📌 Usage  

- Ask questions like:  
  - *"Recommend me a sci-fi movie like Interstellar."*  
  - *"What are some good thriller movies?"*  
- The system retrieves relevant movie recommendations using FAISS and generates responses using a language model.  

## 📈 Future Enhancements  

- Expand the dataset for better recommendations.  
- Integrate multi-modal features (images, trailers).  
- Fine-tune models for improved relevance.  

## 🤝 Contributing  

Feel free to fork the repository, create pull requests, or submit issues!  

## 📜 License  

This project is open-source and available under the **MIT License**.  
