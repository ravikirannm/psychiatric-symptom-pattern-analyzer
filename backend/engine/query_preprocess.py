from transformers import pipeline, AutoModel, AutoTokenizer
import torch
import logging
from .data_model import PatientSymptom,PatientSymptomList
logger = logging.getLogger(__name__)


class QueryPreprocessor:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to("cuda")
        # self.model.eval()
        self.clinical_ner = pipeline("ner", model="d4data/biomedical-ner-all",aggregation_strategy="simple", device=0)

    def dict_to_text_variations(self, data: PatientSymptom) -> list[str]:
        """Convert structured symptom data into text variation for embedding."""
        
        text = f"{data['Style']} {data['ClinicalPresentation']} {data['Duration']} {data['Severity']} {data['Location']} {data['OnsetPattern']} {', '.join(data['AssociatedSymptoms'])}  {data['PatientReportedContext']}"
        return text

    def get_clinical_ner_results(self, symptom_list: PatientSymptomList):
        text_variations = [self.dict_to_text_variations(symptom) for symptom in symptom_list]
        all_entities = set()
        results = []
        for v in text_variations:
            # Only feed clinical content, NOT field names
            clinical_text = v
            
            results = self.clinical_ner(clinical_text)
            
            for entity in results:
                word = entity["word"].strip()
                label = entity["entity_group"]
                score = entity["score"]
                
                # Filter: useful label + confidence + no subwords + length
                if (score > 0.7
                    and not word.startswith("##")
                    and len(word) > 2):
                    all_entities.add(word.lower())
        logger.info(f"Unique clinical entities extracted: {all_entities}")
        
        return f"Keywords: {', '.join(all_entities)}"