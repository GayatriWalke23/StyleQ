<div align="center">

# StyleQ

[![React](https://img.shields.io/badge/React-18.x-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.x-009688.svg)](https://fastapi.tiangolo.com/)
[![Material-UI](https://img.shields.io/badge/Material--UI-5.x-0081CB.svg)](https://mui.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🎨 An intelligent text style analysis and adaptation system powered by AI

[Features](#features) • [Getting Started](#getting-started) • [Documentation](#documentation) • [Contributing](#contributing)

<img src="https://raw.githubusercontent.com/yourusername/styleq/main/docs/images/demo.gif" alt="StyleQ Demo" width="600"/>

</div>

## ✨ Features

- 🎯 **Style Analysis**: Advanced text analysis to understand writing styles
- 🔄 **Style Transfer**: Transform text to match different writing styles
- 📊 **Style Metrics**: Measure formality, complexity, tone, structure, and engagement
- 🎨 **Modern UI**: Beautiful and intuitive user interface
- 🚀 **Real-time**: Instant style analysis and text generation
- 📱 **Responsive**: Works seamlessly on desktop and mobile

## 🚀 Getting Started

### Prerequisites

- Node.js 16.x or later
- Python 3.9 or later
- pip (Python package manager)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/styleq.git
cd styleq
```

2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Install frontend dependencies
```bash
cd frontend
npm install
```

### Running the Application

1. Start the backend server
```bash
cd api
uvicorn main:app --reload
```

2. Start the frontend development server
```bash
cd frontend
npm start
```

Visit `http://localhost:3000` to use the application.

## 🏗️ Project Structure

```
StyleQ/
├── api/                # FastAPI backend server
│   ├── main.py         # Main application entry
│   ├── models/         # Data models and schemas
│   └── services/       # Business logic
├── frontend/           # React frontend
│   ├── src/
│   │   ├── components/ # React components
│   │   └── services/   # API integration
├── models/            # ML models and adapters
│   ├── embeddings/    # Style embedding models
│   └── lora/         # LoRA adapter definitions
├── data/             # Training and example data
└── tests/            # Unit and integration tests
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Style Analysis Guide](docs/style-analysis.md)
- [Development Guide](docs/development.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🛠️ Technologies

### Frontend
- React 18.x with modern hooks and context
- Material-UI 5.x for beautiful UI components
- Axios for API communication
- React Router for navigation

### Backend
- FastAPI for high-performance API endpoints
- PyTorch for deep learning models
- BERT for text embeddings with mean pooling
- LoRA adapters for efficient style transfer
- Style embedding system with L2 normalization

### Key Features
- Style Profile System
  - Formality (0-100%)
  - Complexity (0-100%)
  - Tone (0-100%)
  - Structure (0-100%)
  - Engagement (0-100%)
- Real-time style analysis
- Style interpolation and matching
- Comprehensive test coverage

### Environment Variables
```env
# Backend (.env)
MODEL_PATH=models/bert-base-uncased
EMBEDDING_DIM=768
BATCH_SIZE=16
MAX_LENGTH=512

# Frontend (.env)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_DEBUG=false
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [Material-UI](https://mui.com/) for the beautiful UI components
- The open source community for various tools and libraries

---

<div align="center">

</div>
1. Backend Setup
```bash
cd api
pip install -r requirements.txt
python main.py
```

2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

## License

See [LICENSE](LICENSE) file for details.
