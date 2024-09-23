#in terminal run pip install gensim, nltk created conda env venv python 3.10
# pip install pypdf2
#  Creating example documents
# doc_1 = "A whopping 96.5 percent of water on Earth is in our oceans, covering 71 percent of the surface of our planet. And at any given time, about 0.001 percent is floating above us in the atmosphere. If all of that water fell as rain at once, the whole planet would get about 1 inch of rain."

# doc_2 = "One-third of your life is spent sleeping. Sleeping 7-9 hours each night should help your body heal itself, activate the immune system, and give your heart a break. Beyond that--sleep experts are still trying to learn more about what happens once we fall asleep."

# doc_3 = "A newborn baby is 78 percent water. Adults are 55-60 percent water. Water is involved in just about everything our body does."

# doc_4 = "While still in high school, a student went 264.4 hours without sleep, for which he won first place in the 10th Annual Great San Diego Science Fair in 1964."

# doc_5 = "We experience water in all three states: solid ice, liquid water, and gas water vapor."

# # Create corpus
# corpus = [doc_1, doc_2, doc_3, doc_4, doc_5]


# Import necessary libraries
import PyPDF2
import os

# Function to read PDF and extract text
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# List of your PDF files
pdf_files = ["A_Memoir_of_Anorexia_and_Bulimia_-_Marya_Hornbacher.pdf", "Agorafabulous_-_Sara_Benincasa.pdf", "Beautiful_Boy_A_Fathers_Journey_Through_His_Sons_Addiction_-_David_Sheff.pdf", "Blackout_Remembering_the_Things_I_Drank_to_Forget_-_Sarah_Hepola.pdf", "Blood_Orange_Night_-_Melissa_Bond.pdf"]

# Extract text from each PDF and store in a list
corpus = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# Preprocessing steps: cleaning, tokenization, and lemmatization
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')  
nltk.download('omw-1.4')  
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# remove stopwords, punctuation, and normalize the corpus
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

clean_corpus = [clean(doc).split() for doc in corpus]


#A term-document matrix is merely a mathematical representation of a set of documents and the terms contained within them.
#It’s created by counting the occurrence of every term in each document and then normalizing the counts to create a matrix of 
# values that can be used for analysis.


from gensim import corpora

# Creating document-term matrix 
dictionary = corpora.Dictionary(clean_corpus)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]


# Running LDA model
from gensim.models import LdaModel

# LDA model
lda = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary)

# Results
print(lda.print_topics(num_topics=3, num_words=3))

"""
[
(0, '0.071*"water" + 0.025*"state" + 0.025*"three"'), 
(1, '0.030*"still" + 0.028*"hour" + 0.026*"sleeping"'), 
(2, '0.073*"percent" + 0.069*"water" + 0.031*"rain"')
]
"""