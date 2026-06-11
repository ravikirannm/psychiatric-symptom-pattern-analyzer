# Use Ollama qwen3.5:9b for symptom analysis
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
class SymptomAnalyzer:
    def __init__(self):
        # Initialize any necessary variables or configurations here
        self.client = ollama.Client(host=OLLAMA_URL)
        self.db_manager = DBManager()
        

    def analyze(self, user_query,user_id, thread_id=None):
        # 1. Initialize Memory
        yield {"type": "progress", "thread_id": thread_id, "message": "Initializing reasoning engine..."}

        memory = ConversationMemory(self.db_manager, user_id, thread_id)
        memory._ensure_thread() # Ensure thread exists in DB
        ctx = memory.get_working_context()
        # Build context string for prompts
        history_str = ctx['history']
        shared_str = json.dumps(ctx['shared_memory'], indent=1)
        thread_str = json.dumps(ctx['thread_memory'], indent=1)
        
        # 2. Pass 1: Clinical Reformulation
        yield {"type": "progress", "thread_id": thread_id, "message": "Structuring clinical presentation..."}

        system_prompt = f"""
           You are a clinical language formatter. Your only job is to convert 
            a patient's natural language symptom description into 5 structured 
            clinical reformulations. Each variant must use a different 
            clinical framing style to maximize medical entity coverage.
            ==== SHARED CONTEXT FROM OTHER CONVERSATIONS ====
            {shared_str}
            ==== WORKING MEMORY CONTEXT FROM CURRENT CONVERSATION ====
            {thread_str}
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
                {{
                    "variant": 1,
                    "style": "Standard Medical",
                    "clinical_presentation": "",
                    "duration": "",
                    "severity": "",
                    "location": "",
                    "onset_pattern": "",
                    "associated_symptoms": [],
                    "patient_context": ""
                }},
                ... repeat for variants 2-5
            ]
        """
        messages = [
            {"role": "system", "content": system_prompt},
           
        ]
        for turn in history_str:
            messages.append({"role": 'user', "content": turn['query']})
            messages.append({"role": "assistant", "content": json.dumps(turn['analysis'], indent=1)})
        messages.append({"role": "user", "content": user_query})
        logger.info(f"Analyzing symptoms: {user_query}")
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
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

        # 3. Pass 2: Query Generation
        yield {"type": "progress","thread_id":thread_id, "message": "Formulating PubMed & ICD-11 queries..."}
        

        system_prompt_pass_2 = f"""
            You are a medical search query specialist.
            ==== SHARED CONTEXT FROM OTHER CONVERSATIONS ====
            {shared_str}
            ==== WORKING MEMORY CONTEXT FROM CURRENT CONVERSATION ====
            {thread_str}
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

        yield {"type": "progress","thread_id":thread_id, "message": "Fetching external medical evidence..."}
        
        pubmed_results = verifier.fetch_pubmed(api_inputs.get("pubmed", {}))
        icd11_results = verifier.fetch_icd11(api_inputs.get("icd11", {}))

        # 4. Pass 3: Clinical Synthesis (Structured)
        yield {"type": "progress","thread_id":thread_id, "message": "Synthesizing clinical assessment..."}


        system_prompt_pass_3 = f"""
            You are an expert clinical diagnostic assistant performing a structured differential diagnosis.
            You will be given comprehensive medical information gathered from multiple authoritative sources
            about a patient's symptoms.

            Your job is to synthesize ALL inputs into a clear, evidence-based clinical assessment following
            this strict reasoning pipeline:

            STEP 1 — CONDITION IDENTIFICATION
            Identify all plausible conditions consistent with the presented symptoms. Rank them by likelihood.
            Use ICD-11 codes. Never give a definitive diagnosis — always qualify as "possible" or "likely".

            STEP 2 — SYMPTOM GAP ANALYSIS (critical)
            For each condition shortlisted in Step 1, retrieve its canonical symptom profile from your
            medical knowledge. Compare that profile against the symptoms ALREADY reported by the patient.
            Identify which hallmark or discriminating symptoms of each condition have NOT yet been confirmed
            or denied by the patient. These gaps are what drive the follow-up questions.

            STEP 3 — TARGETED FOLLOW-UP QUESTION GENERATION
            Generate follow-up questions EXCLUSIVELY from the symptom gaps identified in Step 2.
            Each question must:
            - Target a specific symptom that is either (a) strongly associated with one condition,
                helping confirm it, or (b) present in one condition but absent in another, helping
                discriminate between them.
            - Be phrased in plain, non-clinical language the patient can answer yes/no or describe.
            - Not ask about symptoms the patient has already reported.
            - Be tied explicitly to which condition(s) it helps rule in or rule out.

            STEP 4 — RED FLAGS & TESTS
            Flag specific urgent warning signs with immediate actions.
            Recommend targeted diagnostic tests with clear rationale.

            ==== SHARED CONTEXT FROM OTHER CONVERSATIONS ====
            {shared_str}
            ==== WORKING MEMORY CONTEXT FROM CURRENT CONVERSATION ====
            {thread_str}

            Output strict JSON only. No explanation. No preamble. No markdown.
        """

        user_prompt_pass_3 = f"""
            === PATIENT SYMPTOM VARIANTS & KEYWORDS ===
            {json.dumps(processed_inputs, indent=1)}

            === MEDICAL KNOWLEDGE BASE (RAG) ===
            {json.dumps(retrieved_docs, indent=1)}

            === PUBMED EVIDENCE ===
            {json.dumps(pubmed_results, indent=1)}

            === ICD-11 MATCHED CONDITIONS ===
            {json.dumps(icd11_results, indent=1)}

            Based on ALL sources above, and following the STEP 1→4 reasoning pipeline in your instructions,
            generate the clinical assessment in this exact structure:

            {{
                "possible_conditions": [
                    {{
                        "name": "condition name",
                        "icd11_code": "ICD-11 code",
                        "likelihood": "high | medium | low",
                        "reasoning": "why this condition fits the currently reported symptoms",
                        "supporting_evidence": "which source(s) support this",
                        "unconfirmed_hallmark_symptoms": [
                            "symptom A not yet reported by patient",
                            "symptom B not yet reported by patient"
                        ]
                    }}
                ],
                "follow_up_questions": [
                    {{
                        "question": "plain-language question to ask the patient",
                        "targets_condition": "condition name this question probes",
                        "symptom_being_probed": "the specific clinical symptom being asked about",
                        "rules_in_if_yes": "condition(s) this positive answer would support",
                        "rules_out_if_no": "condition(s) this negative answer would help exclude",
                        "discriminates_between": ["condition A", "condition B"]
                    }}
                ],
                "red_flags": [
                    {{
                        "symptom": "specific warning sign",
                        "associated_condition": "condition this flag is tied to",
                        "action": "what to do immediately"
                    }}
                ],
                "recommended_tests": [
                    {{
                        "test": "test name",
                        "reason": "what it rules in or out",
                        "targets_condition": "condition this test helps confirm or exclude"
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
        # Convert to dict for easier downstream use
        symptom_analysis = symptom_analysis_output.model_dump()
        response_data = {
            "symptom_variants": symptom_dicts,
            "generated_queries": api_inputs,
            "pubmed_results": pubmed_results,
            "icd11_results": icd11_results,
            "symptom_analysis": symptom_analysis
        }

        chat_prompt = f"""
            You are an expert clinical diagnostic assistant.
            You will be given comprehensive medical information gathered from multiple 
            authoritative sources about a patient's symptoms.

            Your job is to synthesize ALL inputs into a clear, evidence-based clinical assessment.
            Keep language clear enough for a patient to understand, but do not dumb down the medical accuracy. Use a compassionate tone.

            ==== SHARED CONTEXT FROM OTHER CONVERSATIONS ====
            {shared_str}
            ==== WORKING MEMORY CONTEXT FROM CURRENT CONVERSATION ====
            {thread_str}

            
            Current shared memory context:
            {shared_str}

            Current thread memory context:
            {thread_str}

            Latest analysis result:
            {json.dumps(response_data, indent=1)}

            Please generate a natural language response to the patient that explains the findings in an easy-to-understand way. Use a compassionate tone and avoid medical jargon where possible. The response should help the patient understand their symptoms and the next steps they can take.
        """
        # Stream the natural language response token-by-token
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
            # Yield token for the typewriter effect in the Angular middle panel
            yield {"type": "chat_stream", "token": token}
            
        thread = threading.Thread(target=self.make_history_updates, args=(response_data, user_query, memory, shared_str, thread_str))
        # Start the thread
        thread.start()
        response_data["query_response"] = full_query_response
        
        yield {"type": "done"}
        return response_data

    def make_history_updates(self,response_data, user_query, memory: ConversationMemory, shared_str: str, thread_str: str):
        
        full_history = memory.fetch_thread_history()
        system_prompt_summarize_thread = f"""
        You are a medical scribe. Your only job is to summarize the conversation given to you.

        STRICT RULES:
        - Only include symptoms, findings, and assessments that are EXPLICITLY stated in the conversation below.
        - If a symptom does not appear in the patient's messages, it DOES NOT exist. Do not infer it. Do not add it.
        - Do not draw on general medical knowledge to fill gaps. If information is absent, omit it entirely.
        - Never invent clinical details. A fabricated summary is worse than a short one.

        {f"Previous session context (for continuity only): {thread_str}" if thread_str else ""}
        """

        messages = [
            {"role": "system", "content": system_prompt_summarize_thread},
        ]

        for turn in full_history:
            if turn['role'] == 'user':
                messages.append({"role": "user", "content": turn['content']})
            else:
                # Pass only the human-readable response, not the full JSON blob.
                # Local models lose track of conversation content when drowning in JSON tokens.
                assistant_summary = turn['content'].get('query_response', '')
                analysis = turn['content'].get('symptom_analysis', {})
                conditions = [c['name'] for c in analysis.get('possible_conditions', [])]
                if conditions:
                    assistant_summary += f"\n\nConditions considered: {', '.join(conditions)}"
                messages.append({"role": "assistant", "content": assistant_summary})

        messages.append({"role": "user", "content": user_query})

        # Condense response_data the same way before appending
        condensed_response = response_data.get('query_response', '')
        analysis = response_data.get('symptom_analysis', {})
        conditions = [c['name'] for c in analysis.get('possible_conditions', [])]
        red_flags = [r['symptom'] for r in analysis.get('red_flags', [])]
        tests = [t['test'] for t in analysis.get('recommended_tests', [])]
        if conditions:
            condensed_response += f"\n\nConditions considered: {', '.join(conditions)}"
        if red_flags:
            condensed_response += f"\nRed flags identified: {', '.join(red_flags)}"
        if tests:
            condensed_response += f"\nTests recommended: {', '.join(tests)}"

        messages.append({"role": "assistant", "content": condensed_response})

        # Final instruction MUST be a user turn — Ollama ignores system role mid-conversation
        messages.append({
            "role": "user",
            "content": (
                "Using ONLY the conversation above, write a clinical summary covering:\n"
                "1. Patient's reported symptoms (exact, as stated)\n"
                "2. Conditions considered and their likelihood\n"
                "3. Red flags identified\n"
                "4. Tests recommended\n\n"
                "Do not include any symptom or finding not explicitly present in the messages above."
            )
        })
        logger.info(f"Generating thread summary with messages: {messages}")
        # Create basic model response for thread summary        
        summary_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.5}
        )
        thread_summary = summary_response['message']['content'].strip()
        memory.save_to_memory("summary", thread_summary, shared=False)
        system_prompt_summarize_shared = f"""
            You are maintaining a medical reasoning reference — equivalent to a clinician's internal
            knowledge base built up over years of practice. This is NOT a record of what happened in
            any session. It is a structured collection of general, timeless clinical knowledge.

            Think of it as a textbook that gets smarter over time, not a log that accumulates cases.

            ==== CURRENT KNOWLEDGE BASE ====
            {shared_str if shared_str else "Empty — begin building it."}

            ==== WHAT THIS SESSION'S REASONING REVEALED ====
            {thread_str if thread_str else "None."}

            Your task: review what reasoning strategies, clinical rules, or medical facts were implicitly
            used or validated this session, then update the knowledge base to capture ONLY those — stated
            as general, reusable medical principles.

            WHAT BELONGS IN THE KNOWLEDGE BASE (concrete examples):
            ✓ "Pain on eye movement is a key differentiator for optic neuritis vs other causes of
            transient visual loss — its presence sharply increases pre-test probability."
            ✓ "When visual symptoms co-occur with any neurological deficit, MRI brain + orbits with
            contrast is first-line imaging regardless of symptom duration."
            ✓ "VEP detects subclinical demyelination and is useful when clinical exam and MRI are
            ambiguous in suspected MS workup."
            ✓ "Morning stiffness lasting >1 hour is a discriminating feature between inflammatory
            arthropathy and mechanical/degenerative joint disease."

            WHAT DOES NOT BELONG (hard exclusions):
            ✗ Anything phrased as "a patient presented with..." or "this case showed..."
            ✗ Frequency claims derived from this session ("X is frequently reported") — one session
            is not an epidemiological signal
            ✗ Symptom descriptions copied or abstracted from the current patient's profile
            ✗ Differential lists that mirror the current session's output
            ✗ Any statement that only makes sense because of what happened in this specific session

            REFRAME TEST — before adding any statement, ask:
            "Would this sentence appear verbatim in a medical textbook or UpToDate article,
            completely independent of any patient encounter?"
            If NO → discard it.

            Update the knowledge base by merging new validated principles with existing ones.
            Deduplicate. Keep entries terse and precise — one clinical principle per sentence.

            Output only the updated knowledge base as plain running text. No JSON, no headers,
            no bullet points, no structure. Write it as a single cohesive paragraph that reads
            like a dense clinical reference note a senior physician wrote for themselves.
        """
        messages = [
            {"role": "system", "content": system_prompt_summarize_shared},
        ]        
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
        # Update thread title based on summary
        title_update_prompt = f"""
            You are a thread title generator for a medical symptom analysis tool.
            Based on the following conversation summary, generate a concise and descriptive thread title that captures the main clinical theme of the conversation. The title should be no more than 5 words and should help the user quickly identify the topic of this thread in the future.

            Conversation summary:
            {thread_summary}

            Generate an appropriate thread title:
        """
        logger.info(f"Generating thread title using prompt: {title_update_prompt}")
        title_response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": title_update_prompt},
            ],
            options={"temperature": 0.5}
        )
        new_title = title_response['message']['content'].strip()
        logger.info(f"Generated thread title: {new_title}")
        if new_title:
            logger.info(f"Updating thread title to: {new_title}")
            memory.update_thread_title(new_title)

    