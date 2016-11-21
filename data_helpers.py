import numpy as np
import re
import itertools
from collections import Counter
import cPickle

# file address
fname='/home/zjj/distant_s/model/GoogleNews-vectors-negative300.bin'
# fname="E:/code/MIMLme/back/GoogleNews-vectors-negative300.bin"

data_pos = 'data/CDR.pos'
data_neg = 'data/CDR.neg'
pre_data = 'CDR.p'

# parameters
max_length = 340

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    '''
    preprocess data
    return: split sentences 
    '''
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    """
    # Load data from files
    positive_examples = list(open(data_pos).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(data_neg).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    # one hot label
    positive_labels = [[0, 1] for _ in positive_examples] # one-hot label, argmax 1
    negative_labels = [[1, 0] for _ in negative_examples] # argmax 0
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    '''
    sentences+pad
    '''
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = max_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence[:max_length])
    return padded_sentences


def build_vocab(sentences):
    '''
    vocabulary: dictionary, key is word, value is index
    vocabulary_inv: list, elem is word( only common word)
    '''
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    # every word in all sentences
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    # list, elem is common word. x is the elem of Counter. x[0] is word, x[1] is number
    # get the common word list
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def load_bin_vec(fname, vocab_inv):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    word_vecs: dic, key is word, value is vector
    vocab:
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split()) #3000000 300
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab_inv:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def build_input_data(sentences, labels, word_vecs):
    '''
    x: numpy array, elem is sentences, word is replaced by index, equal length by pad
    y: numpy array, elem is labels, one hot vectors
    '''
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[word_vecs[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    return [x, y]


def load_data():
    """
    master function for data
    """
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    word_vecs = load_bin_vec(fname, vocabulary_inv)
    add_unknown_words(word_vecs, vocabulary_inv)
    x, y = build_input_data(sentences_padded, labels, word_vecs)
    print "x shape:\n"
    print x.shape
    print "y shape:\n"
    print y.shape

    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    if len(data)%batch_size == 0:
        num_batches_per_epoch = int(len(data)/batch_size)
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__=="__main__":
    x,y,voca,voca_inv=load_data()
    cPickle.dump([x,y,voca,voca_inv], open(pre_data, "wb"))
