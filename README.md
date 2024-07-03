# RAG Application with Vertex AI, LangChain, and Streamlit

This application showcases the power of Google Cloud's Vertex AI, LangChain, and Streamlit to create a Retrieval Augmented Generation (RAG) system. RAG systems enhance Large Language Models (LLMs) by providing them with relevant knowledge from a datastore, resulting in more accurate and contextually aware responses.

## Key Features

- **Vertex AI Data Store:** Efficiently stores and manages your knowledge base.
- **Vertex AI Retriever:** Leverages Vertex AI's advanced retrieval capabilities to find the most pertinent information.
- **Vertex AI Ranking:** Intelligently ranks retrieved documents based on their relevance to the user's query.
- **Vertex AI Grounding:** Ensures generated responses are firmly grounded in the retrieved information, improving accuracy and reducing hallucinations.
- **LangChain:** A flexible framework for building LLM-powered applications, handling retrieval, ranking, and response generation.
- **Streamlit:** Creates an intuitive web interface for seamless user interaction.
- **Cloud Run:** Enables scalable and easy deployment of the application.
- **Identity-Aware Proxy (IAP):** Secures access to the application, ensuring only authorized users can interact with it.

## How it Works

1. **User Authentication:** Users authenticate themselves through IAP.
2. **User Query:** The user submits a question or request via the Streamlit interface.
3. **Retrieval:** Vertex AI Retriever searches the knowledge base in Vertex AI Data Store for relevant documents.
4. **Ranking:** Vertex AI Ranking assesses the relevance of retrieved documents to the query.
5. **Grounding:** Vertex AI Grounding verifies that the generated response is consistent with the retrieved information.
6. **Response Generation:** LangChain utilizes the retrieved and ranked documents to craft a comprehensive response.
7. **Display:** The generated response is presented to the user in the Streamlit interface.

## Getting Started

1. **Prerequisites**
   - Google Cloud Project (with billing enabled)
   - Familiarity with Python and command line
2. **Setup**
   - Enable required Google Cloud APIs (Vertex AI, Data Store, Cloud Run, IAP)
   - Create and configure a Vertex AI endpoint
   - Populate Vertex AI Data Store with your knowledge base
   - Install project dependencies: `pip install -r requirements.txt`
   - (Optional) Configure IAP for additional security
3. **Deployment**
   - Deploy the application to Cloud Run
4. **Usage**
   - Access the deployed application URL and authenticate (if IAP is enabled)

## Benefits

- **Enhanced Accuracy:** Grounding mechanisms minimize factual errors and improve the reliability of generated responses.
- **Scalability:** Cloud Run ensures the application can handle increased user traffic and large knowledge bases.
- **Security:** IAP protects sensitive data and ensures only authorized users can access the application.
- **Ease of Use:** Streamlit offers a user-friendly interface, making the application accessible to a wider audience.
