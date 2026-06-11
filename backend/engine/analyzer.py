import ollama
from constants import OLLAMA_MODEL, OLLAMA_URL
import logging
import json
import threading
from .data_model import PatientSymptomList, MedicalCorpusInput, SymptomAnalysisResponse
from .query_preprocess import QueryPreprocessor
from .rag_retriever import MedicalRAGRetriever
from .verifier import MedicalQueryVerifier
from .database import DBManager
from .memory import ConversationMemory

query_preprocessor = QueryPreprocessor()
rag_retriever = MedicalRAGRetriever()
verifier = MedicalQueryVerifier()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CRISIS_RESOURCES = (
    "If you are in immediate danger, please call emergency services (911 in the US) or go to your nearest emergency room. "
    "For mental health crisis support: National Suicide Prevention Lifeline — call or text 988 (US). "
    "Crisis Text Line — text HOME to 741741. "
    "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
)


class SymptomAnalyzer:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_URL)
        self.db_manager = DBManager()

    def analyze(self, user_query, user_id, thread_id=None):
        # 1. Initialize Memory
        yield {"type": "progress", "thread_id": thread_id, "message": "Initializing reasoning engine..."}

        memory = ConversationMemory(self.db_manager, user_id, thread_id)
        memory._ensure_thread()
        ctx = memory.get_working_context()
        history_str = ctx['history']
        shared_str = json.dumps(ctx['shared_memory'], indent=1)
        thread_str = json.dumps(ctx['thread_memory'], indent=1)

        # ── PASS 1: Psychiatric Clinical Reformulation ────────────────────────
        yield {"type": "progress", "thread_id": thread_id, "message": "Structuring psychiatric presentation..."}

        system_prompt = f"""
            You are a psychiatric clinical language formatter. Your only job is to convert
            a patient's natural language description into 5 structured psychiatric
            reformulations. Each variant uses a different evidence-based clinical framing
            to maximize coverage of mental health dimensions.

            ==== SHARED CONTEXT FROM PREVIOUS SESSIONS ====
            {shared_str}
            ==== WORKING MEMORY FROM CURRENT CONVERSATION ====
            {thread_str}

            STRICT RULES:
            - Preserve all symptoms and experiences mentioned — do not add or invent new ones
            - Do not diagnose or suggest specific conditions
            - If duration/severity is unclear, mark as "unspecified"
            - Each variant must be meaningfully different in terminology and framing
            - PsychiatricDomain must be one of: affective, cognitive, behavioral, somatic,
              perceptual, relational, sleep, eating, or mixed
            - CoursePattern must describe the temporal pattern: acute, episodic, chronic-continuous,
              chronic-progressive, remitting-relapsing, or unspecified
            - FunctionalImpact must describe impairment in: occupational, social, self-care, or none apparent
            - Output must be valid JSON array only — no preamble, no explanation

            Variant styles:
            1. DSM-5 Symptom Criteria — map reported experiences to specific DSM-5 criterion language
            2. Mental State Examination (MSE) — structure as appearance/behavior, speech, mood, affect,
               thought process, thought content, perception, cognition, insight, judgment
            3. Biopsychosocial — organize into biological factors, psychological factors, social/
               environmental factors
            4. Longitudinal Course — premorbid functioning, precipitating context, onset, current episode,
               any prior episodes
            5. Functional Impairment — detail impact on occupational functioning, interpersonal
               relationships, activities of daily living, and quality of life

            Output format (strict JSON):
            [
                {{
                    "variant": 1,
                    "style": "DSM-5 Symptom Criteria",
                    "clinical_presentation": "",
                    "duration": "",
                    "severity": "",
                    "psychiatric_domain": "",
                    "onset_pattern": "",
                    "associated_symptoms": [],
                    "patient_reported_context": "",
                    "functional_impact": "",
                    "course_pattern": ""
                }},
                ... repeat for variants 2-5
            ]
        """
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history_str:
            messages.append({"role": "user", "content": turn['query']})
            messages.append({"role": "assistant", "content": json.dumps(turn['analysis'], indent=1)})
        messages.append({"role": "user", "content": user_query})

        logger.info(f"Analyzing psychiatric presentation: {user_query}")
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format=PatientSymptomList.model_json_schema(),
            options={'temperature': 0.2}
        )
        symptom_py_model = PatientSymptomList.model_validate_json(response['message']['content'])
        symptom_dicts = [symptom.model_dump() for symptom in symptom_py_model.symptoms]
        logger.info(f"Reformulation result: {symptom_dicts}")

        keywords = query_preprocessor.get_clinical_ner_results(symptom_dicts)
        processed_inputs = []
        for symptom in symptom_dicts:
            symptom["Keywords"] = keywords
            symptom["OriginalQuery"] = user_query
            processed_inputs.append(symptom)
        retrieved_docs = rag_retriever.retrieve(processed_inputs)

        # ── PASS 2: Psychiatric Query Generation ─────────────────────────────
        yield {"type": "progress", "thread_id": thread_id, "message": "Formulating PubMed & ICD-11 psychiatric queries..."}

        system_prompt_pass_2 = f"""
            You are a psychiatric medical search query specialist.
            ==== SHARED CONTEXT FROM PREVIOUS SESSIONS ====
            {shared_str}
            ==== WORKING MEMORY FROM CURRENT CONVERSATION ====
            {thread_str}
            Your ONLY job is to generate optimized search queries for PubMed and ICD-11
            based on psychiatric presentations. Focus on mental health MeSH terms:
            use terms like [MeSH Terms] for conditions such as "Depressive Disorder",
            "Anxiety Disorders", "Stress Disorders, Post-Traumatic", "Psychotic Disorders",
            "Bipolar Disorder", "Obsessive-Compulsive Disorder", "Personality Disorders",
            "Neurodevelopmental Disorders", "Substance-Related Disorders" etc.
            Prefer journals: JAMA Psychiatry, American Journal of Psychiatry,
            World Psychiatry, Psychological Medicine, Lancet Psychiatry."""

        user_prompt_pass_2 = f"""
            === PATIENT PSYCHIATRIC PRESENTATION ===
            {json.dumps(processed_inputs, indent=1)}

            === RETRIEVED KNOWLEDGE BASE DOCUMENTS ===
            {json.dumps(retrieved_docs, indent=1)}

            === YOUR TASK ===
            Generate search queries in strict JSON only. No explanation. No preamble. No markdown.

            {{
                "pubmed": {{
                    "primary_query": "most specific boolean query using psychiatric MeSH terms",
                    "fallback_query": "broader query if primary returns nothing",
                    "filters": ["Journal Article", "Review", "Clinical Trial", "Randomized Controlled Trial"]
                }},
                "icd11": {{
                    "search_terms": [
                        "most likely psychiatric condition name",
                        "second likely condition",
                        "third likely condition"
                    ]
                }}
            }}

            Rules:
            - PubMed primary query must use AND/OR/NOT boolean operators with [MeSH Terms]
            - ICD-11 terms must be standard psychiatric disorder names (chapter 6: Mental, behavioural
              or neurodevelopmental disorders)
            - Maximum 3 ICD-11 search terms
            - Do not diagnose — generate search queries only
        """
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_pass_2},
                {"role": "user", "content": user_prompt_pass_2}
            ],
            format=MedicalCorpusInput.model_json_schema(),
            options={'temperature': 0.2}
        )
        medical_corpus_api_input = MedicalCorpusInput.model_validate_json(response['message']['content'])
        api_inputs = medical_corpus_api_input.model_dump()
        logger.info(f"Generated psychiatric search queries: {api_inputs}")

        yield {"type": "progress", "thread_id": thread_id, "message": "Fetching psychiatric evidence base..."}

        pubmed_results = verifier.fetch_pubmed(api_inputs.get("pubmed", {}))
        icd11_results = verifier.fetch_icd11(api_inputs.get("icd11", {}))

        # ── PASS 3: Psychiatric Clinical Synthesis ────────────────────────────
        yield {"type": "progress", "thread_id": thread_id, "message": "Synthesizing psychiatric assessment..."}

        system_prompt_pass_3 = f"""
            You are an expert psychiatric diagnostic assistant performing a structured
            mental health differential assessment. You will be given comprehensive
            information gathered from multiple sources about a patient's presentation.

            Your job is to synthesize ALL inputs into a clear, evidence-based psychiatric
            assessment following this strict reasoning pipeline:

            STEP 0 — SAFETY TRIAGE (always first, never skipped)
            Assess risk for: suicidal ideation (passive vs active vs with plan/intent),
            non-suicidal self-harm, and harm to others. Even when the patient has not
            mentioned these, rate them as "none_reported" and set safety_screen_needed=true
            so a direct safety question is always included in follow-up. If any risk is
            mentioned or implied, escalate accordingly with immediate action in red_flags.

            STEP 1 — CONDITION IDENTIFICATION
            Identify all plausible psychiatric conditions consistent with the presentation.
            Use BOTH ICD-11 codes (chapter 6) AND DSM-5 codes. Rank by likelihood.
            Consider: mood disorders, anxiety disorders, trauma/stress-related, OCD spectrum,
            psychotic disorders, dissociative disorders, somatic symptom disorders, personality
            disorders, neurodevelopmental disorders, substance-related disorders.
            Always qualify as "possible" or "likely" — never definitive.

            STEP 2 — DSM-5 CRITERION GAP ANALYSIS (critical)
            For each condition in Step 1, list which DSM-5 criteria the patient has already
            reported (dsm5_criteria_met) and which hallmark criteria remain unconfirmed
            (unconfirmed_hallmark_symptoms). These gaps drive follow-up questions.

            STEP 3 — PSYCHIATRIC FORMULATION
            Build a biopsychosocial formulation:
            - Predisposing: biological, psychological, or social vulnerability factors
            - Precipitating: recent stressors or triggers that brought on this episode
            - Perpetuating: factors maintaining or worsening the current state
            - Protective: strengths, supports, and resilience factors

            STEP 4 — TARGETED FOLLOW-UP QUESTIONS
            Generate questions exclusively from gaps in Step 2.
            ALWAYS include at least one question with psychiatric_domain="safety".
            Each question must:
            - Target a symptom that confirms or discriminates between conditions
            - Be in plain language the patient can answer yes/no or describe briefly
            - Not repeat anything the patient has already reported
            - Carry the correct psychiatric_domain label
            - Be tied to which conditions it rules in or out

            STEP 5 — RED FLAGS & TESTS
            Red flags: any safety concern must appear here with an immediate action.
            Tests: prefer validated rating scales (PHQ-9, GAD-7, PCL-5, AUDIT-C, MDQ,
            MMSE, PANSS) before ordering labs. Include labs only to rule out organic
            causes (thyroid, B12, CBC, metabolic panel, toxicology as indicated).
            Specify test_type: rating_scale, lab, imaging, or neuropsychological.

            ==== SHARED CONTEXT FROM PREVIOUS SESSIONS ====
            {shared_str}
            ==== WORKING MEMORY FROM CURRENT CONVERSATION ====
            {thread_str}

            Output strict JSON only. No explanation. No preamble. No markdown.
        """

        user_prompt_pass_3 = f"""
            === PATIENT PSYCHIATRIC PRESENTATION VARIANTS & KEYWORDS ===
            {json.dumps(processed_inputs, indent=1)}

            === KNOWLEDGE BASE (RAG) ===
            {json.dumps(retrieved_docs, indent=1)}

            === PUBMED EVIDENCE ===
            {json.dumps(pubmed_results, indent=1)}

            === ICD-11 MATCHED CONDITIONS ===
            {json.dumps(icd11_results, indent=1)}

            Based on ALL sources and following the STEP 0→5 reasoning pipeline, generate
            the psychiatric assessment in this exact structure:

            {{
                "possible_conditions": [
                    {{
                        "name": "condition name",
                        "icd11_code": "ICD-11 code",
                        "dsm5_code": "DSM-5 code or empty string",
                        "likelihood": "high | medium | low",
                        "reasoning": "why this fits the reported symptoms",
                        "supporting_evidence": "which source(s) support this",
                        "unconfirmed_hallmark_symptoms": [
                            "DSM-5 criterion not yet confirmed by patient"
                        ],
                        "dsm5_criteria_met": [
                            "DSM-5 criterion already reported by patient"
                        ]
                    }}
                ],
                "risk_assessment": {{
                    "suicidal_ideation": "none_reported | low | moderate | high | imminent",
                    "self_harm_risk": "none_reported | low | moderate | high | imminent",
                    "harm_to_others": "none_reported | low | moderate | high | imminent",
                    "protective_factors": ["factor 1", "factor 2"],
                    "risk_rationale": "brief rationale for these ratings",
                    "safety_screen_needed": true
                }},
                "psychiatric_formulation": {{
                    "predisposing": ["vulnerability factor 1"],
                    "precipitating": ["recent stressor or trigger"],
                    "perpetuating": ["maintaining factor"],
                    "protective": ["resilience or support factor"]
                }},
                "follow_up_questions": [
                    {{
                        "question": "plain-language question for the patient",
                        "psychiatric_domain": "safety | mood | anxiety | psychosis | trauma | substance | cognition | functioning | somatic | personality | sleep | eating",
                        "targets_condition": "condition name this question probes",
                        "symptom_being_probed": "specific clinical symptom being asked about",
                        "rules_in_if_yes": ["condition(s) a yes would support"],
                        "rules_out_if_no": ["condition(s) a no would help exclude"],
                        "discriminates_between": ["condition A", "condition B"]
                    }}
                ],
                "red_flags": [
                    {{
                        "symptom": "specific warning sign",
                        "associated_condition": "condition this flag is tied to",
                        "action": "immediate action required"
                    }}
                ],
                "recommended_tests": [
                    {{
                        "test": "test or scale name",
                        "reason": "what it rules in or out",
                        "targets_condition": "condition this helps confirm or exclude",
                        "test_type": "rating_scale | lab | imaging | neuropsychological"
                    }}
                ]
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
        symptom_analysis = symptom_analysis_output.model_dump()
        response_data = {
            "symptom_variants": symptom_dicts,
            "generated_queries": api_inputs,
            "pubmed_results": pubmed_results,
            "icd11_results": icd11_results,
            "symptom_analysis": symptom_analysis
        }

        # ── Chat Response: Trauma-Informed Natural Language ───────────────────
        risk = symptom_analysis.get("risk_assessment", {})
        elevated_risk = any(
            risk.get(k) in ("moderate", "high", "imminent")
            for k in ("suicidal_ideation", "self_harm_risk", "harm_to_others")
        )
        safety_instruction = (
            f"\n\nIMPORTANT — risk assessment flags elevated concern. You MUST include "
            f"the following crisis resources verbatim before any clinical content:\n"
            f'"{CRISIS_RESOURCES}"\n'
            f"Then continue with the clinical response."
        ) if elevated_risk else (
            "\n\nNote: no elevated safety risk was flagged in this session, but the "
            "response should still include a gentle reminder that professional mental "
            "health support is available and recommended."
        )

        chat_prompt = f"""
            You are a compassionate, trauma-informed psychiatric assistant.
            You will be given a structured psychiatric assessment and your job is to
            communicate findings to the patient in clear, accessible, non-stigmatizing
            language.

            TONE RULES (follow strictly):
            - Start by acknowledging the person's experience and validating that reaching
              out takes courage
            - Use person-first language ("a person experiencing depression", not
              "a depressive")
            - Avoid clinical jargon where a plain equivalent exists
            - Never frame mental health symptoms as weaknesses or character flaws
            - Present conditions as "possibilities being considered" — never as diagnoses
            - End with a clear, actionable next step (e.g., speak with a GP, psychiatrist,
              or psychologist)
            {safety_instruction}

            ==== SHARED CONTEXT FROM PREVIOUS SESSIONS ====
            {shared_str}
            ==== WORKING MEMORY FROM CURRENT CONVERSATION ====
            {thread_str}

            Latest psychiatric assessment:
            {json.dumps(response_data, indent=1)}

            Generate a warm, clear natural language response that:
            1. Validates the patient's experience
            2. Explains the key findings in plain language
            3. Highlights the most important follow-up questions they should be prepared
               to answer with a mental health professional
            4. States clearly what next steps they can take
            5. If safety risk is elevated — leads with crisis resources
        """

        stream_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": chat_prompt},
                {"role": "user", "content": user_query}
            ],
            stream=True,
            options={"temperature": 0.4}
        )

        full_query_response = ""
        for chunk in stream_response:
            token = chunk['message']['content']
            full_query_response += token
            yield {"type": "chat_stream", "token": token}

        thread = threading.Thread(
            target=self.make_history_updates,
            args=(response_data, user_query, memory, shared_str, thread_str)
        )
        thread.start()
        response_data["query_response"] = full_query_response

        yield {"type": "done"}
        return response_data

    def make_history_updates(self, response_data, user_query, memory: ConversationMemory,
                             shared_str: str, thread_str: str):

        full_history = memory.fetch_thread_history()

        system_prompt_summarize_thread = f"""
            You are a psychiatric clinical scribe. Your only job is to summarize the
            conversation given to you.

            STRICT RULES:
            - Only include symptoms, experiences, and findings EXPLICITLY stated in the
              conversation below
            - If a symptom does not appear in the patient's messages, do not include it
            - Do not draw on general psychiatric knowledge to fill gaps
            - Never invent clinical details — a short accurate summary is better than a
              long fabricated one
            - Always include a risk status section even if risk was "none reported"

            {f"Previous session context (for continuity only): {thread_str}" if thread_str else ""}
        """

        messages = [{"role": "system", "content": system_prompt_summarize_thread}]

        for turn in full_history:
            if turn['role'] == 'user':
                messages.append({"role": "user", "content": turn['content']})
            else:
                assistant_summary = turn['content'].get('query_response', '')
                analysis = turn['content'].get('symptom_analysis', {})
                conditions = [c['name'] for c in analysis.get('possible_conditions', [])]
                risk = analysis.get('risk_assessment', {})
                if conditions:
                    assistant_summary += f"\n\nConditions considered: {', '.join(conditions)}"
                if risk:
                    assistant_summary += (
                        f"\nRisk — SI: {risk.get('suicidal_ideation', 'n/a')}, "
                        f"SH: {risk.get('self_harm_risk', 'n/a')}, "
                        f"HTO: {risk.get('harm_to_others', 'n/a')}"
                    )
                messages.append({"role": "assistant", "content": assistant_summary})

        messages.append({"role": "user", "content": user_query})

        condensed_response = response_data.get('query_response', '')
        analysis = response_data.get('symptom_analysis', {})
        conditions = [c['name'] for c in analysis.get('possible_conditions', [])]
        risk = analysis.get('risk_assessment', {})
        red_flags = [r['symptom'] for r in analysis.get('red_flags', [])]
        tests = [t['test'] for t in analysis.get('recommended_tests', [])]
        formulation = analysis.get('psychiatric_formulation', {})

        if conditions:
            condensed_response += f"\n\nConditions considered: {', '.join(conditions)}"
        if risk:
            condensed_response += (
                f"\nRisk — SI: {risk.get('suicidal_ideation', 'n/a')}, "
                f"SH: {risk.get('self_harm_risk', 'n/a')}, "
                f"HTO: {risk.get('harm_to_others', 'n/a')}"
            )
        if red_flags:
            condensed_response += f"\nRed flags: {', '.join(red_flags)}"
        if tests:
            condensed_response += f"\nTests/scales recommended: {', '.join(tests)}"
        if formulation.get('precipitating'):
            condensed_response += f"\nPrecipitating factors: {', '.join(formulation['precipitating'])}"

        messages.append({"role": "assistant", "content": condensed_response})
        messages.append({
            "role": "user",
            "content": (
                "Using ONLY the conversation above, write a psychiatric clinical summary covering:\n"
                "1. Patient's reported symptoms and experiences (exact, as stated)\n"
                "2. Conditions considered and their likelihood\n"
                "3. Risk assessment (suicidal ideation, self-harm, harm to others)\n"
                "4. Biopsychosocial formulation elements mentioned\n"
                "5. Red flags identified\n"
                "6. Assessments and tests recommended\n\n"
                "Do not include any symptom or finding not explicitly present in the messages above."
            )
        })

        logger.info(f"Generating psychiatric thread summary")
        summary_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.5}
        )
        thread_summary = summary_response['message']['content'].strip()
        memory.save_to_memory("summary", thread_summary, shared=False)

        # ── Shared Knowledge Base Update (Psychiatric Principles) ─────────────
        system_prompt_summarize_shared = f"""
            You are maintaining a psychiatric clinical reasoning reference — equivalent to a
            senior psychiatrist's internal knowledge base built up over years of practice.
            This is NOT a record of what happened in any session. It is a structured
            collection of general, timeless psychiatric and psychological knowledge.

            Think of it as a DSM-5/ICD-11 companion that gets smarter over time, not a
            log that accumulates cases.

            ==== CURRENT KNOWLEDGE BASE ====
            {shared_str if shared_str else "Empty — begin building it."}

            ==== WHAT THIS SESSION'S REASONING REVEALED ====
            {thread_str if thread_str else "None."}

            Your task: review what psychiatric reasoning strategies, clinical rules, or
            evidence-based principles were implicitly used or validated this session, then
            update the knowledge base to capture ONLY those — stated as general, reusable
            psychiatric principles.

            WHAT BELONGS (concrete examples):
            ✓ "Anhedonia lasting ≥2 weeks, present most of the day, is a core A-criterion
               for Major Depressive Episode under DSM-5 and must be directly screened."
            ✓ "Trauma history is a key predisposing factor for both PTSD and BPD; the
               distinction often lies in the pervasiveness of affective dysregulation
               across contexts vs. reactivity tied to trauma cues."
            ✓ "PHQ-9 score ≥10 indicates at least moderate depression and warrants formal
               clinical interview; scores ≥20 require urgent safety assessment."
            ✓ "In suspected bipolar disorder, a full longitudinal history is required —
               a current depressive episode alone cannot exclude hypomania or mania."
            ✓ "Safety screening must precede differential reasoning in every psychiatric
               encounter regardless of chief complaint."

            WHAT DOES NOT BELONG (hard exclusions):
            ✗ Anything phrased as "a patient presented with..." or "this case showed..."
            ✗ Frequency claims derived from a single session
            ✗ Symptom descriptions copied from the current patient's profile
            ✗ Differential lists that mirror the current session's output

            REFRAME TEST — before adding any statement, ask:
            "Would this sentence appear verbatim in the DSM-5, ICD-11, UpToDate, or a
            psychiatric textbook, completely independent of any patient encounter?"
            If NO → discard it.

            Update the knowledge base by merging new validated principles with existing ones.
            Deduplicate. Keep entries terse and precise — one clinical principle per sentence.

            Output only the updated knowledge base as plain running text. No JSON, no headers,
            no bullet points. Write it as a single cohesive paragraph that reads like a dense
            psychiatric reference note a senior clinician wrote for themselves.
        """

        messages = [{"role": "system", "content": system_prompt_summarize_shared}]
        shared_summary_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.5}
        )

        shared_summary = shared_summary_response['message']['content'].strip()
        logger.info(f"Thread summary: {thread_summary}")
        logger.info(f"Shared summary: {shared_summary}")
        memory.save_to_memory("shared_summary", shared_summary, shared=True)
        memory.save_turn(user_query, response_data)

        # ── Thread Title ──────────────────────────────────────────────────────
        title_update_prompt = f"""
            You are a thread title generator for a psychiatric symptom analysis tool.
            Based on the conversation summary below, generate a concise and descriptive
            thread title that captures the main clinical theme (e.g., the primary
            presenting concern or suspected condition area). The title should be no more
            than 5 words and help the user quickly identify this thread.
            Do NOT include a patient name. Do NOT use stigmatizing labels.

            Conversation summary:
            {thread_summary}

            Generate an appropriate thread title:
        """
        title_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": title_update_prompt}],
            options={"temperature": 0.5}
        )
        new_title = title_response['message']['content'].strip()
        if new_title:
            logger.info(f"Updating thread title to: {new_title}")
            memory.update_thread_title(new_title)
