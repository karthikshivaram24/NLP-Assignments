import tensorflow as tf
import numpy as np
import gensim
from copy import deepcopy
from a4 import average_f1s,confusion,evaluate
import pandas as pd
from nltk.corpus import brown

######### STEP 1 : Data Preparation ##########################
def read_data(filename):
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

def s_l_lists(sentence_tuples):
    sent_sequence = []
    labels_sequence =[]
    for sentence in sentence_tuples: # List
        words = []
        labels = []
        for word in sentence:  # Tuple
            words.append(word[0])
            labels.append(word[-1])
        sent_sequence.append(words)
        labels_sequence.append(labels)

    return sent_sequence,labels_sequence

def vocabdict(sent_sequence,sent=True):
    vocab_words = set([y for x in sent_sequence for y in x])
    vocabdic = dict()
    i = 0
    for word in sorted(vocab_words):
        vocabdic[word]=i
        i+=1
    if sent==True:
        vocabdic["-UNK-"] = i+1
    return vocabdic


def max_sentLength(sent_sequences):
    c = max([len(x) for x in sent_sequences])
    return c

def convertSeqtoNum(sent_sequences,vocab2dic):
    result = deepcopy(sent_sequences)
    for sent_index in range(len(result)):
        for word_index in range(len(result[sent_index])):
            result[sent_index][word_index] = vocab2dic[result[sent_index][word_index]]

    return result

###############################################################

############################# STEP 2: PADDING #########################################

def padd_data(sequences,pad_tok, max_length):

    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

# Word EMbed
def w2v(w2vModel,vocabdict,dim):
    """
    """
    embeddings = np.zeros(shape=(len(vocabdict)+1,dim),dtype=np.float32)

    for key in vocabdict:
        row_id = vocabdict[key]
        if key in w2vModel:
            embeddings[row_id] = np.asarray(w2vModel[key])

    return embeddings

########################## STEP 3: Get Data as Batches ################################

def get_batches(X,Y,seqlength,batchsize):
    # Data - [[s1],[]]
    x_batch, y_batch,seqlen = [], [],[]
    for i  in range(len(X)):
        if len(x_batch) == batchsize:
            return x_batch, y_batch,seqlen
            x_batch, y_batch,seqlen = [],[],[]
        x_batch += [X[i]]
        y_batch += [Y[i]]
        seqlen.append(seqlength[i])

    if len(x_batch) != 0:
        return  x_batch, y_batch, seqlen

def calculate_Accuracy(Y_True,Y_pred,seq_length):
    correct_no = 0
    total =0
    for lab, lab_pred, length in zip(Y_True, Y_pred, seq_length):
        lab = lab[:length]
        lab_pred = lab_pred[:length]
        for i in range(len(lab)):
            # print(str(lab[i])+"--->"+str(lab_pred[i]))
            if lab[i] == lab_pred[i]:
                correct_no+=1
            total+=1
    acc = correct_no/total
    return acc

def reverseDictionary(key2Value):
    rev = {}
    for key,value in key2Value.items():
        rev[value]=key
    return rev

def convertResult(Y_true,Y_pred,seq_len,int2label):
    result = []
    for t,pr,le in zip(Y_true,Y_pred,seq_len):
        x = []
        y = []
        for i in range(le):
            x.append(int2label[t[i]])
            y.append(int2label[pr[i]])
        result.append(y)
    return result

def outputWriter(pred_labels,true_labels):
    """
    This method writes our RNN's Output of the Test data to the output file
    output2.txt
    """
    # Unflatten List
    pred_labels = [x for y in pred_labels for x in y]
    true_labels = [x for y in true_labels for x in y]

    cf_dataframe = confusion(true_labels=true_labels,pred_labels=pred_labels)
    evaluate_dataframe = evaluate(confusion_matrix=cf_dataframe)
    avg_f1 = average_f1s(evaluation_matrix=evaluate_dataframe)

    with open("output2.txt","w") as fp:
        fp.write(str(cf_dataframe))
        fp.write("\n")
        fp.write(str(evaluate_dataframe))
        fp.write("\n")
        fp.write("avg_f1 : ")
        fp.write(str(avg_f1))

    fp.close()

class RNNModel():
    def __init__(self,lr,embeddings,ntags,hidden,iterations,batchsize):
        self.lr = lr
        self.embeddings = embeddings
        self.ntags = ntags
        self.hidden = hidden
        self.Iterations = iterations
        self.batch_size = batchsize

    def buildAndRun(self,padded_word_ids,padded_labels,seq_length,padded_word_id_test,padded_label_test,seq_length_test):
        # Step 1 Placeholders
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        #Decoder Weights
        W = tf.get_variable("W", shape=[2*self.hidden, self.ntags],dtype=tf.float32)

        b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())
        # Step2:
        # Embedding setup
        L = tf.Variable(self.embeddings, dtype=tf.float32, trainable=False)
        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(L, self.word_ids)

        #Step3
        def RNN_biDirec(pretrained_embeddings,sequence_lengths):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden,activation=tf.sigmoid)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                                                                        lstm_cell,
                                                                        pretrained_embeddings,
                                                                        sequence_length=sequence_lengths,
                                                                        dtype=tf.float32)
            context_rep = tf.concat([output_fw, output_bw], axis=-1)
            ntime_steps = tf.shape(context_rep)[1]
            context_rep_flat = tf.reshape(context_rep, [-1, 2*self.hidden])
            pred = tf.nn.softmax(tf.matmul(context_rep_flat, W) + b)
            return pred,ntime_steps

        pred,ntime_steps = RNN_biDirec(pretrained_embeddings=pretrained_embeddings,sequence_lengths=self.sequence_lengths)

        logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        # shape = (batch, sentence, nclasses)
        mask = tf.sequence_mask(self.sequence_lengths)
        # apply mask
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)

        # Step5 Optimizer:
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss)

        labels_pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        init = tf.global_variables_initializer()

        # Part 2: Running The Built Graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Now we iterate through the batches:
            while step * self.batch_size < self.Iterations:
                batch_X, batch_Y , batch_seq_length = get_batches(X=padded_word_ids,
                                                                  Y=padded_labels,
                                                                  seqlength=seq_length,
                                                                  batchsize=self.batch_size)

                sess.run(train_op,feed_dict={self.word_ids:batch_X,
                                             self.labels:batch_Y,
                                             self.sequence_lengths:batch_seq_length})

                if step % 100 == 0:

                    prediction = sess.run(labels_pred,feed_dict={self.word_ids:batch_X,
                                                                 self.labels:batch_Y,
                                                                 self.sequence_lengths:batch_seq_length})
                    acc = calculate_Accuracy(Y_True=batch_Y,Y_pred=prediction,seq_length=batch_seq_length)
                    # We calculate the Batch Loss
                    loss_b = sess.run(loss,feed_dict={self.word_ids:batch_X,
                                                 self.labels:batch_Y,
                                                 self.sequence_lengths:batch_seq_length})
                    # Print Batch Data
                    print("Iteration --> "+str(step*self.batch_size)+","
                          +"Accuracy-Batch --> " + str(acc)
                          +", "+"Loss-Batch --> " +str(round(loss_b,8)))
                step+=1

            pred_test_labels = sess.run(labels_pred,feed_dict={self.word_ids:padded_word_id_test,
                                                         self.labels:padded_label_test,
                                                         self.sequence_lengths:seq_length_test})
            print("Testing Accuracy : " + str(calculate_Accuracy(Y_True=padded_label_test,Y_pred=pred_test_labels,seq_length=seq_length_test)))
            return pred_test_labels

if __name__== "__main__":

    tf.reset_default_graph()

    brown_sentences = brown.sents()
    w2vModel = gensim.models.Word2Vec(sentences=brown_sentences,size=300,window=5,min_count=5)

    train = read_data("train.txt")
    test = read_data("test.txt")
    train_x_raw,train_y_raw = s_l_lists(train)
    test_x_raw,test_y_raw = s_l_lists(test)
    max_sent_length_train = max_sentLength(sent_sequences=train_x_raw)
    max_sent_length_test = max_sentLength(sent_sequences=test_x_raw)

    pad_d,seq_d_l = padd_data(sequences=train_x_raw,pad_tok="*UNKNOWN-TOKEN*",max_length=max_sent_length_train)
    pad_l,seq_l_l = padd_data(sequences=train_y_raw,pad_tok="*UNKNOWN-LABEL*",max_length=max_sent_length_train)
    data_dic = vocabdict(sent_sequence=pad_d)
    label_dic = vocabdict(sent_sequence=pad_l)

    pad_d_t,seq_d_l_t = padd_data(sequences=test_x_raw,pad_tok="*UNKNOWN-TOKEN*",max_length=max_sent_length_test)
    pad_l_t,seq_l_l_t = padd_data(sequences=test_y_raw,pad_tok="*UNKNOWN-LABEL*",max_length=max_sent_length_test)
    data_dic_t = vocabdict(sent_sequence=pad_d_t)

    # pad_d,seq_l = padd_data(sequences=train_x_raw,pad_tok="-UNK-",max_length=54)
    x = convertSeqtoNum(sent_sequences=pad_d,vocab2dic=data_dic)
    y = convertSeqtoNum(sent_sequences=pad_l,vocab2dic=label_dic)

    x_t = convertSeqtoNum(sent_sequences=pad_d_t,vocab2dic=data_dic_t)
    y_t = convertSeqtoNum(sent_sequences=pad_l_t,vocab2dic=label_dic)

    # Embeddings Matrix:
    em = w2v(w2vModel=w2vModel,vocabdict=data_dic, dim=300)

    rnn = RNNModel(lr=0.01,embeddings=em,ntags=6,hidden=20,iterations=100000,batchsize=32)
    predicted_Labels = rnn.buildAndRun(padded_word_ids=x ,
                                       padded_labels=y,
                                       seq_length=seq_d_l,
                                       padded_word_id_test = x_t,
                                       padded_label_test = y_t,
                                       seq_length_test =seq_d_l_t)
    int2labels = reverseDictionary(key2Value=label_dic)
    results = convertResult(Y_true=y_t,
                               Y_pred=predicted_Labels,
                               seq_len=seq_d_l_t,
                               int2label=int2labels)
    outputWriter(pred_labels=results,true_labels=test_y_raw)
