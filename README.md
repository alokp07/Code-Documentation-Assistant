# Code Documentation Assistant 📚

An intelligent code documentation tool that helps developers understand and navigate large codebases using semantic search and AI-powered analysis.

## 🎯 Features

- **Semantic Code Search**: Find functions, classes, and patterns using natural language queries
- **Architecture Analysis**: Get insights into codebase structure and patterns  
- **Multi-language Support**: Works with Python, JavaScript, Java, C++, TypeScript, and more
- **Smart Code Parsing**: Automatically extracts functions, classes, imports, and relationships
- **Interactive UI**: Clean, intuitive Streamlit interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Vector Database**: Custom implementation (Chroma/Pinecone integration ready)
- **Embeddings**: Semantic search with OpenAI/Anthropic API integration ready
- **Code Parsing**: Multi-language AST analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd code-documentation-assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage

1. **Upload Repository**: Export your GitHub repo as a ZIP file and upload it
2. **Process**: Let the AI parse and index your codebase (takes 10-30 seconds)
3. **Query**: Ask questions in natural language about your code

### Example Queries

- "How does the authentication work?"
- "Show me the main application structure" 
- "Find all database-related functions"
- "What are the API endpoints?"
- "Show me error handling patterns"

## 📁 Project Structure

```
code-documentation-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── demo/                 # Demo files and examples
```

## 🔧 Advanced Configuration

### Adding Real Vector Database (Chroma)

Replace the mock vector database with Chroma:

```bash
pip install chromadb
```

### Adding OpenAI Embeddings

```bash
pip install openai langchain
```

Update the embedding model in `app.py`:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Replace MockEmbeddings with:
embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 💡 How It Works

1. **File Extraction**: Extracts code files from uploaded ZIP repositories
2. **Code Parsing**: Uses regex and AST parsing to identify functions, classes, imports
3. **Semantic Indexing**: Creates embeddings for code snippets and stores in vector database
4. **Query Processing**: Converts natural language queries to semantic search
5. **Response Generation**: Formats results with code examples and architecture insights

## 🎬 Demo

The application includes a built-in demo mode that shows sample results without requiring file uploads. Perfect for showcasing capabilities to potential employers or clients.

## 📈 Roadmap

- [ ] Integration with real vector databases (Chroma, Pinecone)
- [ ] OpenAI/Anthropic API integration for better embeddings
- [ ] Support for more programming languages
- [ ] Code relationship mapping and visualization
- [ ] Integration with GitHub API for direct repo access
- [ ] Advanced code metrics and analysis
- [ ] Team collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Live Demo](https://your-streamlit-app.streamlit.app) (Deploy and add link)
- [GitHub Repository](https://github.com/yourusername/code-documentation-assistant)

---

**Built with ❤️ for developers who love clean, documented code**
