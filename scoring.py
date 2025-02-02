import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine
import numpy as np

from sentence_transformers import SentenceTransformer

from scipy.spatial.distance import cosine



from IPython.display import clear_output
clear_output()

import os
import openai
from openai import OpenAI

#os.environ['OPENAI_API_KEY'] = ""
#api_key = os.environ['OPENAI_API_KEY']
#openai.api_key = api_key
#api_key = ""

#client = OpenAI(api_key = api_key)

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=[text],  # Must be a list
        model="text-embedding-3-small",
    )
    return response.data[0].embedding  # Access the new structure

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

masculine_words = ["man", "male", "boy", "uncle", "grandpa", "father", "dad", "patriarch", "manly", "masculine", "king", "prince", "lord"]
masculine_embeddings = [get_embedding(word) for word in masculine_words]
masculine_sum = masculine_embeddings[0]
for i in range(1, len(masculine_embeddings)):
    masculine_sum = np.add(masculine_sum, masculine_embeddings[i])
masculine_average = masculine_sum / len(masculine_words)

feminine_words = ["woman", "female", "girl", "aunt", "grandma", "mother", "mom", "matriarch", "womanly", "feminine", "queen", "princess", "lady"]
feminine_embeddings = [get_embedding(word) for word in feminine_words]
feminine_sum = feminine_embeddings[0]
for i in range(1, len(feminine_embeddings)):
    feminine_sum = np.add(feminine_sum, feminine_embeddings[i])
feminine_average = feminine_sum / len(feminine_words)

def get_score(word):
  return (cosine_similarity(get_embedding(word), masculine_average) - cosine_similarity(get_embedding(word), feminine_average))

def threshold_divisions(adjectives: list):
    scores = []
    for a in adjectives:
        s = get_score(a)
        scores.append(s)
    adjective_scores = list(zip(adjectives, scores))

    flags = []
    replacements = []

    for tu in adjective_scores:
        score = tu[1] * 10
        if (score <= -0.7):
            flags.append(tu)
        if (score >= -0.7) and (score <= -0.2):
            replacements.append(tu)

    return flags, replacements


def sentence_similarity(sentence, adjective, synonyms):
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  new_sentences = []

  for s in synonyms:
    new = sentence.replace(adjective, s)
    new_sentences.append(new)


  cosine_similarities = []

  for n in new_sentences:
    embeddings = model.encode([sentence, n ])
    embedding1 = embeddings[0]
    embedding2 = embeddings[1]
    cosine_sim = 1 - cosine(embedding1, embedding2)
    cosine_similarities.append(cosine_sim)

  max_index = np.argmax(cosine_similarities)
  return new_sentences[max_index]


print(sentence_similarity("She is very outspoken", "outspoken", ["intelligent", "involved"]))

