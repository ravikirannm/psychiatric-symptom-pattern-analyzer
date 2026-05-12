export interface PubmedResult {
    id: string;
    pub_date: string;
    source: string;
    title: string;
}

export interface ICD11Result {
    code: string;
    description: string;
}

export interface SymptomAnalysis {
    follow_up_questions: { purpose: string; question: string }[];
    possible_conditions: {
        icd11_code: string;
        likelihood: string;
        name: string;
        reasoning: string;
        supporting_evidence: string;
    }[];
    recommended_tests: { reason: string; test: string }[];
    red_flags: { action: string; symptom: string }[];
}