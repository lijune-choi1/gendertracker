from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/update_word_checks', methods=['GET'])
def update_word_checks():
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

if __name__ == '__main__':
    app.run(debug=True)