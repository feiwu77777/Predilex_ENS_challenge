import re

def clean2(raw_text, lemmatizer, stemmer, stop_words = []):
    # remove special characters such as the "'" in "it's".
    text = re.sub(r'\W', ' ', raw_text)
    
    # remove digits
    text = re.sub("\d+", "", text)

    # remove single character such as the "s" in "it s".
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # remove stop words if it's not none.
    for w in stop_words:
        text = re.sub(r"\b" + w + r"\b", '', text)

    # unify successive blank space as one blank space.
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    
    text = text.split(' ')
    # lemmatize ('went' -> 'go') and stem ('photographers' -> 'photographer') each word in the text.
    stemmed_text = []
    for w in text:
        if w != '':
            lemmatized_word = lemmatizer.lemmatize(w, pos='v')
            stemmed_word = stemmer.stem(lemmatized_word)
            stemmed_text.append(stemmed_word)

    text = ' '.join(stemmed_text)
    
    return text

def clean1(raw_text):    
    text = raw_text.replace('\xa0', ' ').replace('\n', ' ').lower()

    text = re.sub(r'x\.\.\.', 'xxx', text)
    text = re.sub(r'y\.\.\.', 'yyy', text)    
    
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text
