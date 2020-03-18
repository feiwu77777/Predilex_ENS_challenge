import re
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np

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

    #enlever les répétitions de '.' et '-'   
    text = re.sub(r"[.]{2,}", "", text)
    text = re.sub(r"[-]{2,}", "", text)

    # reformatter les dates 12.12.2002 en 12/12/2002
    text = re.sub(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", r"\1/\2/\3", text)
    text = re.sub(r"(\d{1,2})\. (\d{1,2})\. (\d{2,4})", r"\1/\2/\3", text)

    # reformatter les a.r.t.p. en artp et 600.000 en 6000000
    r = [(r"(\d{1,3})\.(\d{3})", r"\1\2"),
        (r"(\d{4,6})\.(\d{3})", r"\1\2"),
        (r"(\d{7,9})\.(\d{3})", r"\1\2"),
        (r"(\d{10,12})\.(\d{3})", r"\1\2"),
        (r"(\w)\.(\w)\.(\w).(\w).(\w).(\w).", r"\1\2\3\4\5\6"),
        (r"(\w)\.(\w)\.(\w).(\w).(\w).", r"\1\2\3\4"),
        (r"(\w)\.(\w)\.(\w).(\w).", r"\1\2\3\4"),
        (r"(\w)\.(\w)\.(\w).", r"\1\2\3"),
        (r"(\w)\.(\w)\.", r"\1\2")]
    
    for pair in r:
        text = re.sub(pair[0], pair[1], text)
    
    # enlever double espace
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


def find_opti_thresh(y, proba, plot = True):
    ###
    # Given true labels y_test and predicted probabilities y_proba, determine the auroc score
    # and find the threshold that maximize the recall for both classes
    ###

    # calculate auroc score roc_auc
    fpr, tpr, thresholds = roc_curve(y, proba)
    roc_auc = auc(fpr,tpr)

    # find the threshold that maximize recall of both classes specified with condition "s = 1 - fpr[i] + tpr[i]"
    maxi = 0
    max_ind = 0
    for i in range(len(fpr)):
        s = 1 - fpr[i] + tpr[i]
        if s > maxi:
            maxi = s
            max_ind = i

    best_thresh = thresholds[max_ind]

    # plot the ROC curve 
    if plot == True:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot([fpr[max_ind]], [tpr[max_ind]], marker='o', markersize=10, color="red")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - recall of 0')
        plt.ylabel('recall of 1')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        print(f'optimal threshold is {best_thresh}')

    return best_thresh, roc_auc

def cross_val(x, y, classifier, cv = 10, plot = False):

    ###
    # cross validation on self.x_train with default 5 folds.
    # evaluation metric is set to area under the roc curve instead of accuracy.
    # also look for the threshold that minimize difference between recall for both classes of the validation sets.
    ###

    # split into folds and iterate over them
    kf = KFold(n_splits=cv)
    splits = kf.split(x)
    results = []
    threshs = []
    for train_ind, val_ind in splits:
        x_train, x_val = x[train_ind], x[val_ind], 
        y_train, y_val = y[train_ind], y[val_ind]

        classifier.fit(x_train, y_train)
        
        #pred = classifier.predict(x_val)
        #acc = 0
        #for i in range(len(pred)):
        #    if pred[i] == y_val[i]: acc += 1
        #acc = acc/len(pred)
        # evaluating the auroc of the prediction of x_val
        # find the thresh that minimize recall difference between both classes on x_val
        proba_val = classifier.predict_proba(x_val)[:,1]
        best_thresh, auroc = find_opti_thresh(y_val, proba_val, plot = plot)

        threshs.append(best_thresh)
        results.append(auroc)
    return results, threshs

def grid_search(x, y, params, plot = False):
    ###
    # custom grid search for logistic regression with minimal hyper-parameters
    # also search for the threshold that minimize recall score on classes of the validation sets.
    ###
    count = 0
    max_score = 0
    best_thresh = 0
    for p1 in params['max_depth']:
        for p2 in params['learning_rate']:
            for p3 in params['subsample']:
                for p4 in params['colsample_bytree']:

                    count += 1
                    classifier = XGBClassifier(max_depth = p1,
                                               learning_rate = p2,
                                               subsample = p3,
                                               colsample_bytree = p4,
                                               random_state = 42)
                                               

                    # threshs are optimal thresh that minimize recall difference on
                    # the corresponding validation set.
                    aurocs, threshs = cross_val(x, y, classifier,cv = 5, plot = plot)
                    score = np.mean(aurocs)
                    print('step {}/{}, '.format(count, 36), end = '\r')
                    if score > max_score:
                        print('step {}/{},  current best auroc: {}'.format(count, 36, score), end = '\r')
                        max_score = score
                        best_thresh = np.mean(threshs)
                        best_params = {'max_depth': p1, 
                                       'learning_rate': p2, 
                                       'subsample':  p3, 
                                       'colsample_bytree': p4,
                                       'random_state': 42} 
    
    # set the learner with the best parameters.
    classifier = XGBClassifier(**best_params)
    
    return classifier, best_thresh, best_params