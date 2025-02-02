#FLASK IMPORTS
from flask import Flask, request, jsonify
from flask_cors import CORS

#ADJECTIVE IMPORTS
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

#SENTENCE IMPORTS
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

#SUGGESTIONS IMPORTS
import spacy
from nltk.corpus import wordnet, words, gutenberg
from nltk.probability import FreqDist
from transformers import pipeline, AutoTokenizer, AutoModel

nltk.download('wordnet')
nltk.download('words')
nltk.download('omw-1.4')
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
gutenberg_words = nltk.corpus.gutenberg.words()
fdist = FreqDist(w.lower() for w in gutenberg_words)
common_words = set(words.words())

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

app = Flask(_name_)
CORS(app)

# This method checks every word in the user inputted text for gendered language 
# and flags it if the word is overly gendered. 
@app.route('/update_word_checks', methods=['POST','OPTIONS'])
def update_word_checks():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight test"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response, 200
    text = request.json.get('text', '')

    all_adjectives = ["polite", "friendly"]
    gendered_adjectives = get_gender(all_adjectives)
    to_flag = gendered_adjectives[0]
    to_suggest = gendered_adjectives[1]
    adjectives_and_sentences = find_sentences_with_words(to_suggest, text)
    word_checks = {
        'flagged': {},
        'suggestions': {}
    }

    # find each adjective in the sentence
    for adjective in adjectives_and_sentences:
        for sentence in adjectives_and_sentences[adjective]:
            word_checks['suggestions'][adjective] = identify_gendered_adjectives(sentence, adjective)
    # flag adjectives if they are overly gendered
    for flag in to_flag:
        word_checks['flagged'][flag] = "Consider removing."

    # manual CORS setup
    response = jsonify(word_checks)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

def best_suggestion(sentence, word, replacements):
    # IMPLEMENT THIS
    return replacements[0]

# This method checks if a sentence contains words.
def find_sentences_with_words(words, text):
    sentences = sent_tokenize(text)
    word_sentences = {word: [] for word in words}
    for sentence in sentences:
        for word in words:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE):
                word_sentences[word].append(sentence)
    return word_sentences

# This method identifies which words in the sentence are adjectives.
def identify_adjectives(text):
    # load tagger using custom load function
    tagger = SequenceTagger.load("flair/pos-english")
    print(tagger)
    # create sentence
    sentence = Sentence(text)
    # predict part of speech tags
    tagger.predict(sentence)
    # extract adjectives
    adjectives = [entity.text for entity in sentence.get_spans('pos') if entity.tag == 'JJ']
    return adjectives

nlp = spacy.load("en_core_web_sm")

# This function identifies which adjectives are gendered.
def identify_gendered_adjectives(text, word):
    doc = nlp(text)
    new_text = []
    gendered_dict = {word:""}

    for token in doc:
        # if a word is a gendered adjective, provide alternative gender-neutral suggestions
        if token.pos_ == "ADJ" and token.text.lower() in gendered_dict:
          bert_suggestions = bert_suggest_alternative(text, token.text)
          synonyms = find_neutral_synonym(token.text.lower())
          all_suggestions = bert_suggestions + synonyms
          print("all suggestions",all_suggestions)
          return best_suggestion(text, word, all_suggestions)

# This method finds all neutral synonyms for a word.
def find_neutral_synonym(word):
    # find all synonyms for word given by user in input
    synonyms = wordnet.synsets(word, pos = wordnet.ADJ)
    neutral_options = set()

    # for each synonym in the list of all synonym sets for the given word
    # synset = adjective groups with the same meaning
    for syn in synonyms:
        # find other definitions in the set of synonyms
        for lemma in syn.lemmas():
          # data cleaning
            clean_word = lemma.name().replace("_", " ").lower()
            if clean_word in common_words and clean_word not in nltk.corpus.stopwords.words('english'):
              # to prevent obscure/archaic words from showing up
              if fdist[clean_word] > 6:
                neutral_options.add(clean_word)
    neutral_options.discard(word)
    return list(neutral_options)[:10]

# This function uses the BERT language model to suggest gender-neutral alternatives.
# Best used in longer contexts (ie. >1 word inputs).
def bert_suggest_alternative(sentence, gendered_word):
    masked_sentence = sentence.replace(gendered_word, "[MASK]")
    predictions = fill_mask(masked_sentence)

    filtered_predictions = [pred["token_str"] for pred in predictions[:10]
        if pred["token_str"].isalpha() and len(pred["token_str"]) > 2 
        and pred["token_str"] != gendered_word]

    # sort and rank all suggestions by semantic similarity to original word
    ranked_suggestions = [(suggested_word,
                           find_semantic_similarity
                            (gendered_word,suggested_word)) 
    for suggested_word in filtered_predictions]
    ranked_suggestions = sorted(ranked_suggestions,key = lambda x: x[1],
                                reverse = True)
    return [suggestion[0] for suggestion in ranked_suggestions[:10]]

# This method compares the semantic similarities of the original word to the possible alternatives 
# using a cosine similarity function.
def find_semantic_similarity(word,new_word):
  embedded_word = tokenizer.encode(word,return_tensors = "pt")
  suggested_embedded_word = tokenizer.encode(new_word,return_tensors = "pt")

  with torch.no_grad():
    word_embedding = model(embedded_word)[0][:,0,:]
    suggested_embedding = model(suggested_embedded_word)[0][:,0,:]

  cos_sim = torch.nn.functional.cosine_similarity(word_embedding,suggested_embedding)
  return cos_sim.item()

def get_gender(adjective_list):
    ## FIX THIS
    to_flag = []
    to_suggest = adjective_list
    return(to_flag, to_suggest)

if __name__ == '__main__':
    app.run(debug=True)

