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
    unconfirmed_hallmark_symptoms: List[str] = Field(..., description="Symptoms that are commonly associated but not present in this case")

class FollowUp(BaseModel):
    question: str
    targets_condition: str = Field(..., description="The condition this question is designed to clarify")
    symptom_being_probed: str = Field(..., description="The specific clinical symptom being asked about")
    rules_in_if_yes: List[str] = Field(..., description="Condition(s) this positive answer would support")
    rules_out_if_no: List[str] = Field(..., description="Condition(s) this negative answer would help exclude")
    discriminates_between: List[str] = Field(..., description="Conditions this question helps distinguish between")

class RedFlag(BaseModel):
    symptom: str
    associated_condition: str = Field(..., description="The condition this flag is tied to")
    action: str = Field(..., description="Immediate clinical response required")

class Test(BaseModel):
    test: str
    reason: str = Field(..., description="Differential diagnosis utility of the test")
    targets_condition: str = Field(..., description="The condition this test helps confirm or exclude")

# --- MAIN RESPONSE MODEL ---
class SymptomAnalysisResponse(BaseModel):
    possible_conditions: List[Condition]
    follow_up_questions: List[FollowUp]
    red_flags: List[RedFlag]
    recommended_tests: List[Test]
   