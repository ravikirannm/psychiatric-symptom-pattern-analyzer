from flask import Flask, request
from .analyzer import SymptomAnalyzer
from .rag_ingestion import MedicalRAGIngestion

symptom_analyzer = SymptomAnalyzer()
rag_ingestion = MedicalRAGIngestion()

# Create an instance of the Flask class
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    # Placeholder for symptom analysis logic
    data = request.get_json()
    user_query = data.get('query', '')

    result = symptom_analyzer.analyze(user_query)
    return result

@app.route('/reset-rag', methods=['GET'])
def reset_rag():
    rag_ingestion.ingest_medrag_corpus()
    return "RAG pipeline reset successfully!", 200

# Start the application
if __name__ == '__main__':
    app.run(debug=True)
