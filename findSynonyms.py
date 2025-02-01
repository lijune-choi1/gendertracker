import spacy
from nltk.corpus import wordnet, words, gutenberg
from nltk.probability import FreqDist
from transformers import pipeline, AutoTokenizer, AutoModel
import nltk
import torch


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

gendered_dict = {
    "handsome": "",
    "beautiful": "",
    "manly": "",
    "motherly": "",
    "fatherly": "",
    "effeminate": "",
    "masculine": "",
    "feminine": "",
    "bossy": "",
    "outspoken": "",
    "polite": ""
}

nlp = spacy.load("en_core_web_sm")

def identify_gendered_adjectives(text):
    doc = nlp(text)
    new_text = []

    for token in doc:
        if token.pos_ == "ADJ" and token.text.lower() in gendered_dict:
          # if word is an adjective in list of gendered words
          print(token.text, "is a gendered word.")
          
          bert_suggestions = bert_suggest_alternative(text, token.text)
          print("bert suggestions",bert_suggestions)
          synonyms = find_neutral_synonym(token.text.lower())
          print("neutral synonyms",synonyms)
          
          all_suggestions = bert_suggestions + synonyms
          print("Here are some synonyms you can use instead:",all_suggestions)

        else:
          print(token.text,"is not a gendered word.")

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
              if fdist[clean_word] > 6:
                neutral_options.add(clean_word)

    neutral_options.discard(word)
    return list(neutral_options)[:10]

def bert_suggest_alternative(sentence, gendered_word):
    masked_sentence = sentence.replace(gendered_word, "[MASK]")
    predictions = fill_mask(masked_sentence)

    filtered_predictions = [pred["token_str"] for pred in predictions[:10]
        if pred["token_str"].isalpha() and len(pred["token_str"]) > 2 
        and pred["token_str"] != gendered_word]

    ranked_suggestions = [(suggested_word,
                           find_semantic_similarity
                            (gendered_word,suggested_word)) 
    for suggested_word in filtered_predictions]

    ranked_suggestions = sorted(ranked_suggestions,key = lambda x: x[1],
                                reverse = True)
    return [suggestion[0] for suggestion in ranked_suggestions[:10]]


def find_semantic_similarity(word,new_word):
  embedded_word = tokenizer.encode(word,return_tensors = "pt")
  suggested_embedded_word = tokenizer.encode(new_word,return_tensors = "pt")

  with torch.no_grad():
    word_embedding = model(embedded_word)[0][:,0,:]
    suggested_embedding = model(suggested_embedded_word)[0][:,0,:]

  cos_sim = torch.nn.functional.cosine_similarity(word_embedding,suggested_embedding)
  return cos_sim.item()


testText = input()
neutral_text = identify_gendered_adjectives(testText)