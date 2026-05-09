import requests
from Bio import Entrez
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import logging
load_dotenv()
logger = logging.getLogger(__name__)
# --- CONFIGURATION ---
Entrez.email = os.getenv("BIO_EMAIL")  # Set your email for NCBI API access

class MedicalQueryVerifier:
    def __init__(self):
        # Using NLM mirror for ICD-11 to bypass complex OAuth overhead
        self.icd_url = "https://clinicaltables.nlm.nih.gov/api/icd11_codes/v3/search"

    # ─── PUBMED SEARCH ──────────────────────────────────
    def fetch_pubmed(self, pubmed_input: dict, limit=5):
        """Searches PubMed with MeSH terms and filters."""
        query = pubmed_input.get("primary_query")
        
        # Format filters as PubMed 'Filter' tags
        if pubmed_input.get("filters"):
            filter_tags = [f"{f}[Filter]" for f in pubmed_input["filters"]]
            filter_str = " AND ".join(filter_tags)
            query = f"({query}) AND {filter_str}"

        try:
            # 1. Execute Search
            handle = Entrez.esearch(db="pubmed", term=query, retmax=limit)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            # Fallback logic
            if not id_list and pubmed_input.get("fallback_query"):
                logger.info(f"No results for primary query. Attempting fallback: {pubmed_input['fallback_query']}")
                return self.fetch_pubmed({"primary_query": pubmed_input["fallback_query"]}, limit)

            if not id_list:
                return []

            # 2. Fetch metadata for found IDs
            handle = Entrez.esummary(db="pubmed", id=",".join(id_list))
            summaries = Entrez.read(handle)
            handle.close()

            return [{
                "id": s["Id"],
                "title": s["Title"],
                "source": s["Source"],
                "pub_date": s["PubDate"]
            } for s in summaries]

        except Exception as e:
            return [{"error": f"PubMed Exception: {str(e)}"}]

    # ─── ICD-11 SEARCH ──────────────────────────────────
    def fetch_icd11(self, icd_input: dict):
        """Searches ICD-11 terms using the NLM mirror."""
        all_results = []
        
        # Parallelize requests if searching multiple terms
        def search_term(term):
            try:
                response = requests.get(self.icd_url, params={"terms": term, "df": "code,title"})
                if response.status_code == 200:
                    data = response.json()
                    return [{"code": m[0], "description": m[1]} for m in data[3]]
            except Exception:
                return []
            return []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_term, t) for t in icd_input.get("search_terms", [])]
            for f in futures:
                all_results.extend(f.result())
        
        # Deduplicate results by code
        return [dict(t) for t in {tuple(d.items()) for d in all_results}]

# --- RUNTIME EXECUTION ---
if __name__ == "__main__":
    advanced_input = {
        'pubmed': {
            'primary_query': '(Low Back Pain [MeSH Terms] AND Chronic Pain [MeSH Terms]) OR (Neuropathic Pain [MeSH Terms] AND Sciatica [MeSH Terms])', 
            'fallback_query': 'Back Pain [Mesh Terms] OR Neuropathy [Mesh Terms]', 
            'filters': ['Journal Article', 'Review', 'Clinical Trial']
        }, 
        'icd11': {
            'search_terms': ['Low Back Pain', 'Sciatica', 'Neuropathic Pain']
        }
    }

    fetcher = MedicalDataFetcher()
    
    print("--- FETCHING PUBMED EVIDENCE ---")
    results = fetcher.fetch_pubmed(advanced_input['pubmed'])
    for idx, paper in enumerate(results, 1):
        print(f"{idx}. {paper.get('title')} ({paper.get('pub_date')})")

    print("\n--- FETCHING ICD-11 CLASSIFICATIONS ---")
    codes = fetcher.fetch_icd11(advanced_input['icd11'])
    for code in codes[:8]:
        print(f"[{code['code']}] {code['description']}")