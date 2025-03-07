# Illuminate

Illuminate scans a user-inputted body of text for adjectives that skew stereotypically male or female and offers users alternative, more gender-neutral phrases to use. For example, "aggressive" typically skews male whereas "bossy" typically skews female ([Bolukbasi et. al, 2016](https://arxiv.org/pdf/1607.06520)). Our goal in developing this web application is to illuminate and neutralize gender bias for word embeddings in natural language, ensuring more fair hiring practices that do not perpetuate gender stereotypes.

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Installing Requirements

Requirements for the software and other tools to build, test and push 

Install spaCy library

    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl

Install Natural Language Toolkit library

    pip install nltk

Install Torch library

    pip install torch

Example: "She is very outspoken in meetings."

    outspoken is a gendered word.
    Here are some synonyms you can use instead: ['candid', 'blunt', 'frank']
    
## Contributors

  - **Hailey Coval** - *Backend Developer*
    ([hailstorm24](https://github.com/hailstorm24))
  - **Li June Choi** - *Frontend Developer*
    ([lijune-choi1](https://github.com/lijune-choi1))
  - **Navya Sahay** - *Backend Developer*
    ([nsahay2004](https://github.com/nsahay2004))
  - **Yuyuan Huang** - *Backend Developer*
    ([yhuang2024](https://github.com/yhuang2024))

## Acknowledgments

Thank you to Hack @ Brown for giving us the opportunity to work on this project! We had a great time and hope to be back next year.
