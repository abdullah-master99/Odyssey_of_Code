# ConsultBid AI - RFP Analysis System

An AI-powered system for analyzing Request for Proposal (RFP) documents and evaluating company eligibility using advanced NLP techniques.

## Features

- Automated RFP requirement extraction
- Company capability analysis
- Eligibility evaluation with detailed reports
- PDF document processing
- Interactive web interface
- Export results as PDF

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdullah-master99/Odyssey_of_Code.git
cd Odyssey_of_Code
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```env
GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL="all-mpnet-base-v2"
LLM_MODEL="llama-3.2-90b-vision-preview"
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload RFP and company documents

4. Click "Evaluate Eligibility" to get the analysis

## Project Structure

- `/agents` - AI agents for different analysis tasks
- `/data` - Document storage and results
- `/static` - Web assets (CSS, JS, images)
- `/templates` - HTML templates
- `/embeddings` - Document embeddings storage

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request