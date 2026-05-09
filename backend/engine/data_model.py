from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class PatientSymptom(BaseModel):
    Variant: int
    Style: str
    ClinicalPresentation: str
    Duration: str
    Severity: str
    Location: str
    OnsetPattern: str
    AssociatedSymptoms: List[str]
    PatientReportedContext: str

class PatientSymptomList(BaseModel):
    symptoms: List[PatientSymptom]

class PubMedConfig(BaseModel):
    primary_query: str
    fallback_query: str
    filters: List[str]

class ICD11Config(BaseModel):
    search_terms: List[str]

class MedicalCorpusInput(BaseModel):
    pubmed: PubMedConfig
    icd11: ICD11Config

class LikelihoodEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

# --- NESTED MODELS ---
class Condition(BaseModel):
    name: str = Field(..., description="Name of the medical condition")
    icd11_code: str = Field(..., description="Official ICD-11 diagnostic code")
    likelihood: LikelihoodEnum
    reasoning: str = Field(..., description="Explanation of why this fits the patient data")
    supporting_evidence: str = Field(..., description="Reference to medical textbooks or PubMed papers")

class FollowUp(BaseModel):
    question: str
    purpose: str = Field(..., description="Clinical reasoning for asking this question")

class RedFlag(BaseModel):
    symptom: str
    action: str = Field(..., description="Immediate clinical response required")

class Test(BaseModel):
    test: str
    reason: str = Field(..., description="Differential diagnosis utility of the test")

# --- MAIN RESPONSE MODEL ---
class SymptomAnalysisResponse(BaseModel):
    possible_conditions: List[Condition]
    follow_up_questions: List[FollowUp]
    red_flags: List[RedFlag]
    recommended_tests: List[Test]
    disclaimer: str = "This is not a medical diagnosis. Please consult a qualified healthcare professional."