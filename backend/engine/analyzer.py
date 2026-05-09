# Use Ollama qwen3.5:9b for symptom analysis
import ollama
from constants import OLLAMA_MODEL, OLLAMA_URL
import logging
import json
from .data_model import PatientSymptomList, MedicalCorpusInput, SymptomAnalysisResponse
from .query_preprocess import QueryPreprocessor
from .rag_retriever import MedicalRAGRetriever
from .verifier import MedicalQueryVerifier

query_preprocessor = QueryPreprocessor()
rag_retriever = MedicalRAGRetriever()
verifier = MedicalQueryVerifier()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class SymptomAnalyzer:
    def __init__(self):
        # Initialize any necessary variables or configurations here
        self.client = ollama.Client(host=OLLAMA_URL)

    def analyze(self, user_query):
        system_prompt = """
           You are a clinical language formatter. Your only job is to convert 
            a patient's natural language symptom description into 5 structured 
            clinical reformulations. Each variant must use a different 
            clinical framing style to maximize medical entity coverage.

            Rules:
            - Preserve all symptoms mentioned, do not add or invent new ones
            - Do not speculate, diagnose, or suggest conditions
            - If duration/severity is unclear, mark as "unspecified"
            - Each variant must be meaningfully different in terminology and style
            - Output must be valid JSON array only, no preamble, no explanation

            Variant styles:
            1. Standard medical terminology (formal clinical note style)
            2. SOAP note style (Subjective/Objective focused)
            3. Symptom-mechanism style (describe how the symptom behaves)
            4. Anatomical-systems style (organ/system focused language)
            5. Diagnostic criteria style (ICD/DSM-aligned language)

            Output format (strict JSON):
            [
                {
                    "variant": 1,
                    "style": "Standard Medical",
                    "clinical_presentation": "",
                    "duration": "",
                    "severity": "",
                    "location": "",
                    "onset_pattern": "",
                    "associated_symptoms": [],
                    "patient_context": ""
                },
                ... repeat for variants 2-5
            ]
        """
        
        logger.info(f"Analyzing symptoms: {user_query}")
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            format=PatientSymptomList.model_json_schema(), # Force JSON structure
            options={'temperature': 0.2} # Low temperature for medical accuracy
        )
        symptom_py_model = PatientSymptomList.model_validate_json(response['message']['content'])
        symptom_dicts = [symptom.model_dump() for symptom in symptom_py_model.symptoms]
        logger.info(f"Analysis result: {symptom_dicts}")
        keywords = query_preprocessor.get_clinical_ner_results(symptom_dicts)
        processed_inputs = []
        for symptom in symptom_dicts:
            symptom["Keywords"] = keywords
            symptom["OriginalQuery"] = user_query
            # Convert dict to text 
            symptom_text = "".join([f"{key}: {value}" for key, value in symptom.items()])
            processed_inputs.append(symptom)
        retrieved_docs = rag_retriever.retrieve(processed_inputs)
        

        system_prompt_pass_2 = f"""
            You are a medical search query specialist.
            Your ONLY job is to generate optimized search queries for PubMed"""
        user_prompt_pass_2 = f"""
            and ICD-11 based on the patient input below.
            === PATIENT INPUT ===
            {json.dumps(processed_inputs, indent=1)}

            === Extracted Medical Documents ===
            {json.dumps(retrieved_docs, indent=1)}

            === YOUR TASK ===
            Generate search queries in strict JSON only. No explanation. 
            No preamble. No markdown.

            {{
                "pubmed": {{
                    "primary_query": "most specific boolean query using MeSH terms",
                    "fallback_query": "broader query if primary returns nothing",
                    "filters": ["Journal Article", "Review", "Clinical Trial"]
                }},
                "icd11": {{
                    "search_terms": [
                    "most likely condition name",
                    "second likely condition",
                    "third likely condition"
                    ]
                }}
            }}

            Rules:
            - PubMed primary query must use AND/OR/NOT boolean operators
            - PubMed must use MeSH terms where possible e.g. [MeSH Terms]
            - ICD-11 terms must be standard disease names, not symptoms
            - Maximum 3 ICD-11 search terms
            - Do not diagnose, only generate search queries
        """
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_pass_2},
                {"role": "user", "content": user_prompt_pass_2}
            ],
            format=MedicalCorpusInput.model_json_schema(), # Force JSON structure
            options={'temperature': 0.2} # Low temperature for medical accuracy
        )
        medical_corpus_api_input = MedicalCorpusInput.model_validate_json(response['message']['content'])
        # Convert to dict for easier downstream use
        api_inputs = medical_corpus_api_input.model_dump()
        logger.info(f"Generated search queries: {api_inputs}")
        pubmed_results = verifier.fetch_pubmed(api_inputs.get("pubmed", {}))
        icd11_results = verifier.fetch_icd11(api_inputs.get("icd11", {}))

        system_prompt_pass_3 = """
            You are an expert clinical diagnostic assistant.
            You will be given comprehensive medical information gathered from multiple 
            authoritative sources about a patient's symptoms.

            Your job is to synthesize ALL inputs into a clear, evidence-based 
            clinical assessment.

            Output strict JSON only. No explanation. No preamble. No markdown.

            Rules:
            - Base reasoning on retrieved documents and PubMed evidence
            - Use ICD-11 codes for all conditions mentioned
            - Rank conditions by likelihood given the symptoms
            - Follow-up questions must help narrow the differential diagnosis
            - Red flags must be specific, not generic
            - Never give a definitive diagnosis — always say "possible" or "likely"
            - Keep language clear enough for a patient to understand"""

        user_prompt_pass_3 = f"""
            === PATIENT SYMPTOM VARIANTS & KEYWORDS ===
            {json.dumps(processed_inputs, indent=1)}

            === MEDICAL KNOWLEDGE BASE (RAG) ===
            {json.dumps(retrieved_docs, indent=1)}

            === PUBMED EVIDENCE ===
            {json.dumps(pubmed_results, indent=1)}

            === ICD-11 MATCHED CONDITIONS ===
            {json.dumps(icd11_results, indent=1)}

            Based on ALL sources above, generate the clinical assessment:
            {{
                "possible_conditions": [
                    {{
                    "name": "condition name",
                    "icd11_code": "ICD-11 code",
                    "likelihood": "high/medium/low",
                    "reasoning": "why this condition fits the symptoms",
                    "supporting_evidence": "which source supports this"
                    }}
                ],
                "follow_up_questions": [
                    {{
                    "question": "question to ask patient",
                    "purpose": "what this helps rule in or out"
                    }}
                ],
                "red_flags": [
                    {{
                    "symptom": "specific warning sign",
                    "action": "what to do immediately"
                    }}
                ],
                "recommended_tests": [
                    {{
                    "test": "test name",
                    "reason": "what it rules in or out"
                    }}
                ],
                "disclaimer": "This is not a medical diagnosis. Please consult a qualified healthcare professional."
            }}
        """

        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_pass_3},
                {"role": "user", "content": user_prompt_pass_3}
            ],
            format=SymptomAnalysisResponse.model_json_schema(),
            options={"temperature": 0}
        )

        symptom_analysis_output = SymptomAnalysisResponse.model_validate_json(response['message']['content'])
        # Convert to dict for easier downstream use
        symptom_analysis = symptom_analysis_output.model_dump()
        
        return {
            "symptom_variants": symptom_dicts,
            "generated_queries": api_inputs,
            "pubmed_results": pubmed_results,
            "icd11_results": icd11_results,
            "symptom_analysis": symptom_analysis
        }
    