from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class PatientSymptom(BaseModel):
    Variant: int
    Style: str
    ClinicalPresentation: str
    Duration: str
    Severity: str
    PsychiatricDomain: str  # e.g., affective, cognitive, behavioral, somatic, perceptual
    OnsetPattern: str
    AssociatedSymptoms: List[str]
    PatientReportedContext: str
    FunctionalImpact: str = ""   # impact on work, relationships, self-care
    CoursePattern: str = ""      # episodic, chronic, progressive, remitting


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


class RiskLevel(str, Enum):
    none_reported = "none_reported"
    low = "low"
    moderate = "moderate"
    high = "high"
    imminent = "imminent"


class PsychiatricDomainEnum(str, Enum):
    safety = "safety"
    mood = "mood"
    anxiety = "anxiety"
    psychosis = "psychosis"
    trauma = "trauma"
    substance = "substance"
    cognition = "cognition"
    functioning = "functioning"
    somatic = "somatic"
    personality = "personality"
    sleep = "sleep"
    eating = "eating"


# --- NESTED MODELS ---

class Condition(BaseModel):
    name: str = Field(..., description="Name of the psychiatric condition")
    icd11_code: str = Field(..., description="Official ICD-11 diagnostic code")
    dsm5_code: str = Field(default="", description="DSM-5 code if applicable")
    likelihood: LikelihoodEnum
    reasoning: str = Field(..., description="Explanation of why this fits the patient data")
    supporting_evidence: str = Field(..., description="Reference to medical literature or PubMed")
    unconfirmed_hallmark_symptoms: List[str] = Field(
        ..., description="DSM-5/ICD-11 diagnostic criteria not yet confirmed by the patient"
    )
    dsm5_criteria_met: List[str] = Field(
        default_factory=list,
        description="DSM-5 criteria already reported or confirmed by the patient"
    )


class RiskAssessment(BaseModel):
    suicidal_ideation: RiskLevel = Field(..., description="Level of suicidal ideation risk")
    self_harm_risk: RiskLevel = Field(..., description="Level of non-suicidal self-harm risk")
    harm_to_others: RiskLevel = Field(..., description="Level of risk to others")
    protective_factors: List[str] = Field(
        default_factory=list, description="Factors that reduce overall risk"
    )
    risk_rationale: str = Field(..., description="Brief clinical rationale for the risk ratings")
    safety_screen_needed: bool = Field(
        default=True,
        description="Whether a direct safety screening question must be asked"
    )


class PsychiatricFormulation(BaseModel):
    predisposing: List[str] = Field(
        default_factory=list,
        description="Biological, psychological, or social vulnerability factors present before onset"
    )
    precipitating: List[str] = Field(
        default_factory=list, description="Recent triggers or stressors that brought on the episode"
    )
    perpetuating: List[str] = Field(
        default_factory=list, description="Factors maintaining or worsening the current presentation"
    )
    protective: List[str] = Field(
        default_factory=list, description="Strengths, social supports, and resilience factors"
    )


class FollowUp(BaseModel):
    question: str
    psychiatric_domain: PsychiatricDomainEnum = Field(
        ..., description="Which psychiatric dimension this question probes"
    )
    targets_condition: str = Field(..., description="The condition this question is designed to clarify")
    symptom_being_probed: str = Field(..., description="The specific clinical symptom being asked about")
    rules_in_if_yes: List[str] = Field(..., description="Condition(s) a positive answer would support")
    rules_out_if_no: List[str] = Field(..., description="Condition(s) a negative answer would help exclude")
    discriminates_between: List[str] = Field(..., description="Conditions this question helps distinguish")


class RedFlag(BaseModel):
    symptom: str
    associated_condition: str = Field(..., description="The condition this flag is tied to")
    action: str = Field(..., description="Immediate clinical response required")


class Test(BaseModel):
    test: str
    reason: str = Field(..., description="Differential diagnosis utility of the test")
    targets_condition: str = Field(..., description="The condition this test helps confirm or exclude")
    test_type: str = Field(
        default="", description="One of: rating_scale, lab, imaging, neuropsychological"
    )


# --- MAIN RESPONSE MODEL ---

class SymptomAnalysisResponse(BaseModel):
    possible_conditions: List[Condition]
    risk_assessment: RiskAssessment
    psychiatric_formulation: PsychiatricFormulation
    follow_up_questions: List[FollowUp]
    red_flags: List[RedFlag]
    recommended_tests: List[Test]
