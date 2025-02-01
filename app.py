from flask import Flask, jsonify, request
from flask_cors import CORS

#ADJECTIVE IMPORTS
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

app = Flask(__name__)
CORS(app)

@app.route('/update_word_checks', methods=['POST'])
def update_word_checks():
    text = request.json.get('text', '')
    all_adjectives = identify_adjectives(text)
    print(all_adjectives)
    word_checks = {
        'flagged': {
            'witch': 'Delete Immediately',
            'hag': '',
            'mankind': ''
        },
        'suggestions': {
            'handsome': '[insert better synonym]',
            'womanly': '[insert better synonym]',
        }
    }
    return jsonify(word_checks)

def identify_adjectives(text):
    # load tagger using custom load function
    tagger = SequenceTagger.load("flair/pos-english")
    print(tagger)
    # create sentence
    sentence = Sentence(text)
    # predict POS tags
    tagger.predict(sentence)
    # extract adjectives
    adjectives = [entity.text for entity in sentence.get_spans('pos') if entity.tag == 'JJ']
    return adjectives


# def identify_adjectives(text):
#   # load tagger
#   tagger = SequenceTagger.load("flair/pos-english", weights_only=False)

#   # make example sentence
#   sentence = Sentence(text)

#   # predict NER tags
#   tagger.predict(sentence)
#   adjectives = [entity.text for entity in sentence.get_spans('pos') if entity.tag == 'JJ']
#   return adjectives


#   # Print each word with its POS tag
#   adjectives = []
#   print("Word-wise POS Tags:")
#   for token in sentence:
#       pos_tag = token.annotation_layers["pos"][0].value  # Correct way to access POS tags
#       if (pos_tag == "JJ"):
#         adjectives.append(token.text)

#   return adjectives


if __name__ == '__main__':
    app.run(debug=True)