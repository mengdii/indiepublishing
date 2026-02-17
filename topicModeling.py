# Conduct topic modeling based on publications' content summaries 
# References: 
#     https://github.com/NanGinger/Valio_news_propaganda
#     https://stackoverflow.com/questions/42995073/displaying-topics-associated-with-a-document-query-in-gensim

! pip install gensim
! pip install pyLDAvis

import json
import gensim
import string
import re
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, utils
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import webbrowser
nltk.download('stopwords')
nltk.download('punkt_tab')


def topics(dataset: str, topic_no: int, output_vis: str):

    with open(dataset, "r", encoding="utf-8") as new_file:
        data = json.load(new_file)

    books = []

    for book in data:
        for date, book_text in book.items():
            text_content = book_text
            books.append(text_content)

    preprocessed_texts = []

    # Regular expression to remove punctuations
    pattern = r'[^\w\s]' 

    stop_words = nltk.corpus.stopwords.words('english')

    # Based on the dataset, added stopwords that are 
    # common concepts related to publishing, 
    # concepts related to format or genre, 
    # concepts that are too frequent or generic, 
    # concepts that are too specific, and other irrelevant words
    
    new_stopwords = ['book','books','collection','collections','collective','publication','published','history','histories','social','culture','cultural','space','work','works','text','texts','stories','new','art','visual','archive','archives','archiving','publishing','concept','exhibition','contemporary','museum','research','essay','essays','words','image','form','forms','idea','ideas','graphic','design','poem','poems','poetry','written','writing','drawing','drawings','comics','photos','photographs','photography','photographic','photographers','practice','practices','process','narrative','concept','features','project','series','artist','artists','life','three','print','way','ways','point','various','sometimes','sometime','something','time','times','includes','world','people','includes','alongside','patterns','artistic','years','based','like','s','one','two','first','second','also','presents','many','make','takes','us','zdanevich','zaum','singapore','use','using','jim','year','well','day','offers','part','author','within','days','things','objects','farid','across','zine','others','tree','even','around','looks','beech','together','made','contains','brings','rather','making','table','could','another','aráuz','joey','archie','without','motutapu','ralph','paul','skagit','yet','de','marx','still','may','take','riis','including','genderfail','man','edition','images','issue','often','always','every','see','look','lara','would','find','banda','back','along','since','printed','chalayan','along','want','roma','red','delta','created','sohrab','different','become','found','place','create','must','experience','experiences','featuring','malay','body','explores','behind','much','english','four','nt','linhart','set','anita','alina','kristian','bora','san','msf','mcdonald','songwat','mau','lee','australia','luis','might']
    stop_words.extend(new_stopwords)

    # Lowercase
    for text in books:
        text = text.lower()

    # Tokenize the text using NLTK's word_tokenize()
        tokens = word_tokenize(text)

    # Remove stopwords and numbers, and join words back into a string
        filtered_text = [word for word in tokens if word not in stop_words and not word.isdigit()]
        preprocessed_text = ' '.join(filtered_text)

    # Remove remaining punctuation
        preprocessed_text = re.sub(pattern, '', preprocessed_text)

    # Append preprocessed text to the list
        preprocessed_texts.append(preprocessed_text)

    # Tokenize each string into a list of words
    tokenized_texts = [word_tokenize(text) for text in preprocessed_texts]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    dictionary.save('text_dictionary')

    # LDA model
    lda_model = models.LdaModel(corpus, alpha='auto', num_topics = topic_no, id2word = dictionary)

    '''
    # Coherence
    cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print(f'Coherence: {coherence}')
    '''
    
    for topic_id, topic_words in lda_model.print_topics():
        print(f'Topic {topic_id + 1}: {topic_words}')

    
    lda_model.save('lda_model')


    # Visualize the topics

    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics = False)
    pyLDAvis.save_html(lda_display, output_vis)
    webbrowser.open(output_vis)

    return lda_model


# Check the topics a document belong to

def check_topics(dataset: str):

    with open(dataset, "r", encoding="utf-8") as new_file:
        data = json.load(new_file)

    books = []

    for book in data:
        for date, book_text in book.items():
            text_content = book_text
            books.append(text_content)

    preprocessed_texts = []

    # Preprocessing
    pattern = r'[^\w\s]' 
    stop_words = nltk.corpus.stopwords.words('english')
    new_stopwords = ['book','books','collection','collections','collective','publication','published','history','histories','social','culture','cultural','space','work','works','text','texts','stories','new','art','visual','archive','archives','archiving','publishing','concept','exhibition','contemporary','museum','research','essay','essays','words','image','form','forms','idea','ideas','graphic','design','poem','poems','poetry','written','writing','drawing','drawings','comics','photos','photographs','photography','photographic','photographers','practice','practices','process','narrative','concept','features','project','series','artist','artists','life','three','print','way','ways','point','various','sometimes','sometime','something','time','times','includes','world','people','includes','alongside','patterns','artistic','years','based','like','s','one','two','first','second','also','presents','many','make','takes','us','zdanevich','zaum','singapore','use','using','jim','year','well','day','offers','part','author','within','days','things','objects','farid','across','zine','others','tree','even','around','looks','beech','together','made','contains','brings','rather','making','table','could','another','aráuz','joey','archie','without','motutapu','ralph','paul','skagit','yet','de','marx','still','may','take','riis','including','genderfail','man','edition','images','issue','often','always','every','see','look','lara','would','find','banda','back','along','since','printed','chalayan','along','want','roma','red','delta','created','sohrab','different','become','found','place','create','must','experience','experiences','featuring','malay','body','explores','behind','much','english','four','nt','linhart','set','anita','alina','kristian','bora','san','msf','mcdonald','songwat','mau','lee','australia','luis','might']
    stop_words.extend(new_stopwords)
    for text in books:
        text = text.lower()
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word not in stop_words and not word.isdigit()]
        preprocessed_text = ' '.join(filtered_text)
        preprocessed_text = re.sub(pattern, '', preprocessed_text)

    # Check the topic probability distribution for the document

        check_text = preprocessed_text
        lda = models.LdaModel.load('lda_model')
        dictionary = Dictionary.load('text_dictionary')
        tkn_doc = utils.simple_preprocess(check_text)
        doc_bow = dictionary.doc2bow(tkn_doc)
        doc_vec = lda[doc_bow] 
        print(f'{doc_vec}')
   
    return 


topics("book_catalog.json", 6, "books_lda_vis.html")

check_topics("book_catalog_check.json")

