# coding: utf-8
"""CS585: Assignment 4

See README.md
"""

###  You may add to these imports for gensim and nltk.
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import brown
import urllib.request
import gensim
import nltk
#####################################


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')

def read_data(filename):
    """
    Read the data file into a list of lists of tuples.

    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag

    For example, the first two entries in the
    returned result for 'train.txt' are:

    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    ###TODO
    result = []
    with open(filename,'r') as f:
        data = f.read()
        sentences = data.split("\n\n")
        for i in range(0,len(sentences),1):
            if not sentences[i].startswith("-DOCSTART-"):
                sent_temp = []
                tokens_with_details = sentences[i].strip().split("\n")
                for token in tokens_with_details:
                    token_tags = tuple(token.strip().split())
                    sent_temp.append(token_tags)
                result.append(sent_temp)
    f.close()
    return result

def make_feature_dicts(data,
                       w2v_model,
                       token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       context=True,
                       w2v=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.

    See a3_test.

    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    ###TODO
    pass
    result = []
    ner_tags = []

    def general_update(dic_1,dic_2,flag):
        """
        This method updates the dictionary for the feature context=true

        params:     dic1- either the current dic or previous dic depending on flag value
                    dic2- either the next dic or the curr dic depending on the flag value
        returns:    Nothing
        """
        if flag=="prev":
            # dic_1 = prev
            # dic_2 = current
            for key in dic_1.keys():
                if not key.startswith("prev_"):
                    dic_2["prev_"+key] = dic_1[key]

        if flag=="next":
            # dic_1 = current
            # dic_2 = next
            for key in dic_2.keys():
                if not key.startswith("prev_"):
                    dic_1["next_"+key] = dic_2[key]


    for sent_index in range(len(data)):
        result_len = 0
        for tok_index in range(len(data[sent_index])):
            token_dic={}
            token_tuple = data[sent_index][tok_index]
            if token:
                token_dic['tok='+token_tuple[0].lower()]= 1

            if caps:
                if token_tuple[0][0].isupper():
                    token_dic['is_caps'] = 1

            if pos:
                token_dic['pos='+token_tuple[1]]=1

            if chunk:
                token_dic['chunk='+token_tuple[2]]=1

            if w2v:
                if token_tuple[0] in w2v_model:
                    w2v_vector = w2v_model[token_tuple[0]]
                    for i in range(len(w2v_vector)):
                        token_dic['w2v_'+str(i+1)] = w2v_vector[i]

            if context:
                # for update next we can call the function making :
                # next = curr and curr = previous
                if result_len !=0 :
                    # update previous for current
                    general_update(result[-1],token_dic,flag="prev")
                    general_update(result[-1],token_dic,flag="next")
                    # update next for the previous dictionary where next for it will be the current one:
            ner_tags.append(token_tuple[3])
            result.append(token_dic)
            result_len+=1

    return result,np.array(ner_tags)


def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.

    Params:
      true_labels....numpy array of true NER labels, one per token
      pred_labels....numpy array of predicted NER labels, one per token
    Returns:
    A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
    See Log.txt for an example.
    """
    index = Counter(true_labels).keys()
    df = pd.DataFrame(0,index=sorted(list(index)),columns=sorted(list(index)))
    # Use ix or you will get inverted values
    for i in range(len(true_labels)):
        df.ix[true_labels[i],pred_labels[i]]+=1
    return df

def evaluate(confusion_matrix):
    """
    Compute precision, recall, f1 for each NER label.
    The table should be sorted in ascending order of label name.
    If the denominator needed for any computation is 0,
    use 0 as the result.  (E.g., replace NaNs with 0s).

    NOTE: you should implement this on your own, not using
        any external libraries (other than Pandas for creating
        the output.)
    Params:
    confusion_matrix...output of confusion function above.
    Returns:
    A Pandas DataFrame. See Log.txt for an example.
    """
    ###TODO
    df = pd.DataFrame(0.0,index=["precision","recall","f1"],columns=list(confusion_matrix))
    labels = list(confusion_matrix)

    confusion_matrix["row_sum"] = confusion_matrix.sum(axis=1)
    for label in labels:
        tp = confusion_matrix.ix[label,label]
        tp_fn = confusion_matrix.ix[label,"row_sum"]
        tp_fp = sum(confusion_matrix.ix[:,label].tolist())
        precision = 0.0 if np.isnan(tp_fp) or tp<=0.0 else tp/tp_fp
        recall = 0.0 if np.isnan(tp_fn) or tp<=0.0 else tp/tp_fn
        f1 = 0.0 if precision + recall <=0.0 else (2 * precision * recall )/(precision + recall)
        df.ix["precision",label] = precision
        df.ix["recall",label] =  recall
        df.ix["f1",label] =  f1
    return df

def average_f1s(evaluation_matrix):
    """
    Returns:
    The average F1 score for all NER tags,
    EXCLUDING the O tag.
    """
    ###TODO
    sum_f1 = np.sum(evaluation_matrix.ix["f1",evaluation_matrix.columns != "O"].tolist())
    return sum_f1/(len(evaluation_matrix.columns)-1)

def evaluate_combinations(train_data, test_data,w2vModel):
    """
    Run 16 different settings of the classifier,
    corresponding to the 16 different assignments to the
    parameters to make_feature_dicts:
    caps, pos, chunk, context
    That is, for one setting, we'll use
    token=True, caps=False, pos=False, chunk=False, context=False
    and for the next setting we'll use
    token=True, caps=False, pos=False, chunk=False, context=True

    For each setting, create the feature vectors for the training
    and testing set, fit a LogisticRegression classifier, and compute
    the average f1 (using the above functions).

    Returns:
    A Pandas DataFrame containing the F1 score for each setting,
    along with the total number of parameters in the resulting
    classifier. This should be sorted in descending order of F1.
    (See Log.txt).

    Note1: You may find itertools.product helpful for iterating over
    combinations.

    Note2: You may find it helpful to read the main method to see
    how to run the full analysis pipeline.
    """
    ###TODO
    settings = []
    combos = sorted(list(product([True,False],repeat=5)))
    result = pd.DataFrame(0,index = [x for x in range(0,len(combos))],columns=["f1","n_params","caps","pos","chunk","context","w2v"])
    # brown_sentences = brown.sents()
    # w2vModel = gensim.models.Word2Vec(sentences=brown_sentences,size=50,window=5,min_count=5)
    for i in range(len(combos)):
        X_train_dic,X_train_labels = make_feature_dicts(data=train_data,w2v_model=w2vModel,token=True,caps=combos[i][0],pos=combos[i][1],chunk=combos[i][2],context=combos[i][3],w2v=combos[i][4])
        vectorizer = DictVectorizer()
        X_TRAIN = vectorizer.fit_transform(X=X_train_dic)
        X_test_dic,X_test_labels = make_feature_dicts(data=test_data,w2v_model=w2vModel,token=True,caps=combos[i][0],pos=combos[i][1],chunk=combos[i][2],context=combos[i][3],w2v=combos[i][4])
        X_TEST = vectorizer.transform(X=X_test_dic)
        clf = LogisticRegression()
        clf.fit(X_TRAIN,X_train_labels)
        pred = clf.predict(X_TEST)
        confusion_matrix = confusion(true_labels=X_test_labels,pred_labels=pred)
        evaluation_matrix = evaluate(confusion_matrix=confusion_matrix)
        params = clf.coef_.shape[0] * clf.coef_.shape[1]
        result.ix[i,"f1"] = average_f1s(evaluation_matrix=evaluation_matrix)
        result.ix[i,"n_params"] = params
        result.ix[i,"caps"] = combos[i][0]
        result.ix[i,"pos"] = combos[i][1]
        result.ix[i,"chunk"] = combos[i][2]
        result.ix[i,"context"]=combos[i][3]
        result.ix[i,"w2v"] = combos[i][4]

    with open('output1.txt','w') as fp:
        fp.write(str(result.sort_values(by="f1",ascending=False)))
    fp.close()
    return result.sort_values(by="f1",ascending=False)


if __name__ == '__main__':
    """
        You'll have to modify this based on the instructions.
    """
    pass
    download_data()
    train_data = read_data('train.txt')
    brown_sentences = brown.sents()
    # w2vModel is a vocab dictionary
    w2vModel = gensim.models.Word2Vec(sentences=brown_sentences,size=50,window=5,min_count=5)
    # w2vModel.save('brown_Model')
    dicts, labels = make_feature_dicts(train_data,
                                   token=True,
                                   caps=True,
                                   pos=True,
                                   chunk=True,
                                   context=True,
                                   w2v=True,
                                   w2v_model=w2vModel)
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)

    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,
                                                token=True,
                                                caps=True,
                                                pos=True,
                                                chunk=True,
                                                context=True,
                                                 w2v=True,
                                                 w2v_model=w2vModel)
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))
    preds = clf.predict(X_test)
    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
    print('average f1s: %f\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data,w2vModel=w2vModel)
    print('combination results:\n%s' % str(combo_results))
