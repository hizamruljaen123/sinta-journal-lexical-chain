from flask import Flask, render_template, request, stream_with_context, Response, jsonify
import json
import requests
import time
import pandas as pd
from collections import defaultdict
import spacy

# Try loading Indonesian model, fallback to English if not available
try:
    nlp = spacy.load("id_core_news_sm")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Warning: Using English model as fallback - Indonesian model not found")
    except OSError:
        nlp = spacy.blank("id")
        print("Warning: Using blank Indonesian model - no language model available")

class LexicalChainAnalyzer:
    def __init__(self):
        self.lexical_chains = defaultdict(list)
        self.term_frequencies = defaultdict(int)
        self.term_details = defaultdict(dict)

    def process_text(self, text):
        doc = nlp(text.lower())
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    def build_chains(self, query, snippets):
        query_terms = self.process_text(query)
        snippet_terms = []
        
        for snippet in snippets:
            snippet_terms.extend(self.process_text(snippet))
        
        all_terms = query_terms + snippet_terms
        
        # Build detailed lexical chains
        for i in range(len(all_terms) - 1):
            current_term = all_terms[i]
            next_term = all_terms[i + 1]
            self.lexical_chains[current_term].append(next_term)
            self.term_frequencies[current_term] += 1
            
            # Track term positions and connections
            if current_term not in self.term_details:
                self.term_details[current_term] = {
                    'count': 0,
                    'connected_terms': defaultdict(int)
                }
            self.term_details[current_term]['count'] += 1
            self.term_details[current_term]['connected_terms'][next_term] += 1
        
        return query_terms

    def calculate_relevance(self, query_terms, snippet):
        snippet_terms = self.process_text(snippet)
        relevance_score = 0
        term_analysis = []
        
        for term in query_terms:
            if term in snippet_terms:
                term_score = self.term_frequencies.get(term, 0)
                relevance_score += term_score
                
                # Get connected terms and their frequencies
                connected = []
                for related_term, count in self.term_details.get(term, {}).get('connected_terms', {}).items():
                    if related_term in snippet_terms:
                        connected.append({'term': related_term, 'count': count})
                        relevance_score += 1
                
                term_analysis.append({
                    'term': term,
                    'score': term_score,
                    'connected_terms': connected
                })
        
        return {
            'score': relevance_score,
            'term_analysis': term_analysis
        }

class GoogleSearcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def search_google(self, query, journal_info):
        search_query = f"{query} site:{journal_info['link']} filetype:pdf"
        params = {
            "engine": "google",
            "q": search_query,
            "hl": "id",
            "num": 20,
            "api_key": self.api_key
        }
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            results = response.json()
            articles = []
            if "organic_results" in results:
                for result in results["organic_results"]:
                    link = result.get('link', '')
                    if not link.lower().endswith('.pdf'):
                        continue
                    article = {
                        'judul': result.get('title', ''),
                        'link': link,
                        'snippet': result.get('snippet', ''),
                        'jurnal': journal_info['nama_jurnal'],
                        'peringkat_sinta': journal_info.get('sinta_rank', ''),
                        'website_jurnal': journal_info['link']
                    }
                    articles.append(article)
            return articles
        except Exception as e:
            print(f"Error saat mencari di {journal_info['nama_jurnal']}: {str(e)}")
            return []

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    @stream_with_context
    def generate():
        topic = request.form.get('topic')
        sinta_rank = request.form.get('sinta_rank')
        sources_json = request.form.get('sources_json')

        api_key = "d3d775e783819ad347d88e1f236ff3a8a6e883e171e836525df0bd7607bfe995"
        searcher = GoogleSearcher(api_key)
        analyzer = LexicalChainAnalyzer()

        # Prefer generated JSON from list.html, fallback to original
        journals_data = None
        for filename in ("list_journal_from_list_html.json", "list_journal.json"):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    journals_data = json.load(f)
                break
            except FileNotFoundError:
                continue

        if journals_data is None:
            yield "data: ERROR: Tidak ada file JSON daftar jurnal ditemukan.\n\n"
            return

        all_results = []
        snippets = []

        # If sources_json provided, parse it and use those explicit journals (array of {nama_jurnal, link, sinta_rank?})
        if sources_json:
            try:
                selected = json.loads(sources_json)
                if not isinstance(selected, list):
                    raise ValueError("invalid")
            except Exception:
                yield "data: ERROR: format sources_json tidak valid\n\n"
                return

            for journal in selected:
                # ensure minimal keys
                if not journal.get("nama_jurnal") and not journal.get("link"):
                    continue
                journal.setdefault("nama_jurnal", journal.get("link", "Tidak ada nama"))
                journal.setdefault("link", journal.get("link", ""))
                journal.setdefault("sinta_rank", journal.get("sinta_rank", "SELECTED"))
                yield f"data: Mencari di {journal['nama_jurnal']} ({journal.get('sinta_rank')})...\n\n"
                results = searcher.search_google(topic, journal)
                all_results.extend(results)
                snippets.extend([r.get('snippet', '') for r in results])
                time.sleep(1)
        else:
            def group_matches(selected_rank, group_key):
                if not selected_rank:
                    return group_key.startswith("SINTA_") or group_key == "NON_SINTA"
                sr = selected_rank.strip().lower()
                if sr in ("non", "nonsinta", "non_sinta", "non-sinta"):
                    return group_key == "NON_SINTA"
                if sr.isdigit():
                    return group_key == f"SINTA_{sr}"
                return False

            # First pass to collect snippets for analysis
            for rank in journals_data:
                # only consider SINTA groups and NON_SINTA
                if not (rank.startswith("SINTA_") or rank == "NON_SINTA"):
                    continue
                if not group_matches(sinta_rank, rank):
                    continue
                for journal in journals_data[rank]:
                    # skip header-like or empty entries
                    if not journal.get('nama_jurnal'):
                        continue
                    journal['sinta_rank'] = rank
                    yield f"data: Mencari di {journal['nama_jurnal']} ({rank})...\n\n"
                    results = searcher.search_google(topic, journal)
                    all_results.extend(results)
                    snippets.extend([r.get('snippet', '') for r in results])
                    time.sleep(1)
        
        # Build lexical chains and calculate relevance
        query_terms = analyzer.build_chains(topic, snippets)
        for result in all_results:
            analysis = analyzer.calculate_relevance(query_terms, result['snippet'])
            result['relevance_score'] = analysis['score']
            result['lexical_analysis'] = analysis['term_analysis']
        
        df = pd.DataFrame(all_results)
        if len(df) > 0:
            # Sort by relevance score
            df = df.sort_values('relevance_score', ascending=False)
            articles = df.to_dict(orient='records')
            yield f"data: {json.dumps(articles)}\n\n"
        else:
            yield "data: DONE\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/journals', methods=['GET'])
def journals():
    # prefer generated JSON from list.html, fallback to original
    data = None
    for filename in ('list_journal_from_list_html.json', 'list_journal.json'):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            break
        except FileNotFoundError:
            continue

    if data is None:
        return jsonify({'error': 'No journal JSON found'}), 404

    ordered = []
    for i in range(1, 7):
        k = f'SINTA_{i}'
        if k in data:
            ordered.append(k)
    if 'NON_SINTA' in data:
        ordered.append('NON_SINTA')

    result = {'catatan': data.get('catatan', '')}
    for k in ordered:
        result[k] = data.get(k, [])

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
