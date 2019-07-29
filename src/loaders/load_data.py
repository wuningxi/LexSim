import importlib
import os
import pickle

import numpy as np

from src.loaders.Quora.build import build
from src.loaders.augment_data import create_large_train, double_task_training_data
from src.preprocessing.Preprocessor import Preprocessor

def get_filenames(opt):
    filenames = []
    for s in opt['subsets']:
        for t in opt['tasks']:
            prefix = ''
            if opt['dataset'] == 'Quora':
                if s.startswith('p_'):
                    prefix = ''
                else:
                    prefix = 'q_'
            if opt['dataset'] == 'PAWS':
                prefix = 'p_'
            if opt['dataset'] == 'MSRP':
                prefix = 'm_'
            if opt['dataset'] == 'STS':
                prefix = 's_'
            filenames.append(prefix+s+'_'+t)
    return filenames

def get_filepath(opt):
    filepaths = []
    for name in get_filenames(opt):
        if 'quora' in name:
            filepaths.append(os.path.join(opt['datapath'], 'Quora', name + '.txt'))
            print('quora in filename')
        else:
            filepaths.append(os.path.join(opt['datapath'], opt['dataset'], name + '.txt'))
    return filepaths

def load_file(filename,onehot=True):
    """
    Reads file and returns tuple of (ID1, ID2, D1, D2, L) if ids=False
    """
    # todo: return dictionary
    ID1 = []
    ID2 = []
    D1 = []
    D2 = []
    L = []
    with open(filename,'r',encoding='utf-8') as read:
        for i,line in enumerate(read):
            # print(line.split('\t'))
            id1, id2, d1, d2, label = line.rstrip().split('\t')
            ID1.append(id1)
            ID2.append(id2)
            D1.append(d1)
            D2.append(d2)
            if 's_' in filename:
                if float(label)>=4:
                    label = 1
                elif float(label)<4:
                    label = 0
                else:
                    ValueError()
            L.append(int(label))
    L = np.array(L)
    # L = L.reshape(len(D1),1)
    if onehot:
        classes = L.shape[1] + 1
        L = get_onehot_encoding(L)
        print('Encoding labels as one hot vector.')
    return (ID1, ID2, D1, D2, L)

def get_dataset_max_length(opt):
    '''
    Determine maximum number of tokens in both sentences, as well as highes max length for current task
    :param opt: 
    :return: [maximum length of sentence in tokens,should first sentence be shortened?]
    '''
    tasks = opt['tasks']
    if opt['dataset'] in ['Quora','PAWS']:
        cutoff = opt.get('max_length', 24)
        if cutoff == 'minimum':
            cutoff = 24
        s1_len, s2_len = cutoff, cutoff
    elif opt['dataset']=='MSRP':
        cutoff = opt.get('max_length', 40)
        if cutoff == 'minimum':
            cutoff = 40
        s1_len, s2_len = cutoff, cutoff
    elif 'B' in tasks:
        cutoff = opt.get('max_length', 100)
        if cutoff == 'minimum':
            cutoff = 100
        s1_len, s2_len = cutoff, cutoff
    elif 'A' in tasks or 'C' in tasks:
        cutoff = opt.get('max_length', 200)
        if cutoff == 'minimum':
            s1_len = 100
            s2_len = 200
        else:
            s1_len, s2_len = cutoff,cutoff
    return s1_len,s2_len,max([s1_len,s2_len])

def reduce_examples(matrices, m):
    '''
    Reduces the size of matrices
    :param matrices: 
    :param m: 
    :return: 
    '''
    return [matrix[:m] for matrix in matrices]

def create_missing_datafiles(opt,datafile,datapath):
    if not os.path.exists(datapath) and 'large' in datafile:
        create_large_train()
    if not os.path.exists(datapath) and 'double' in datafile:
        double_task_training_data()
    if not os.path.exists(datapath) and 'quora' in datafile:
        quora_opt = opt
        quora_opt['dataset'] = 'Quora'
        build(quora_opt)

def get_cache_folder(opt):
    return opt['datapath'] + 'cache/'

def load_cache_or_process(opt, cache, onehot, vocab):
    ID1 = []
    ID2 = []
    R1 = []
    R2 = []
    T1 = []
    T2 = []
    L1 = []
    L2 = []
    L = []
    replace_ngrams = opt.get('n_gram_embd',False)
    lemmatize = opt.get('lemmatize',False)
    assert not (replace_ngrams and lemmatize)
    filenames = get_filenames(opt)
    print(filenames)
    filepaths = get_filepath(opt)
    print(filepaths)
    for datafile,datapath in zip(filenames,filepaths):
        #create_missing_datafiles(opt,datafile,datapath) # if necessary
        cache_folder = get_cache_folder(opt)
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        # separate cache for n-gram / no n-gram replacement
        if lemmatize:
            suffix = '_lemma'
        elif replace_ngrams:
            suffix = '_ngram'
        else:
            suffix = ''
        cached_path = cache_folder + datafile + suffix + '.pickle'
        # load preprocessed cache
        print(cached_path)
        if cache and os.path.isfile(cached_path):
            print("Loading cached input for " + datafile + suffix)
            try:
                with open(cached_path, 'rb') as f:
                    if lemmatize:
                        id1, id2, r1, r2, l1, l2, l = pickle.load(f)
                    else:
                        id1, id2, r1, r2, t1, t2, l = pickle.load(f)
            except ValueError:
                Warning('No ids loaded from cache: {}.'.format(cached_path))
                with open(cached_path, 'rb') as f:
                    r1, r2, l = pickle.load(f)
                    id1 = None
                    id2 = None

        # do preprocessing if cache not available
        else:
            print('Creating cache...')
            load_ids = opt.get('load_ids',True)
            if not load_ids:
                DeprecationWarning('Load_ids is deprecated setting. Now loaded automatically.')
            id1, id2, r1, r2, l = load_file(datapath,onehot)
            t1 = Preprocessor.basic_pipeline(r1)
            t2 = Preprocessor.basic_pipeline(r2)
            if lemmatize:
                l1 = Preprocessor.lemmatizer_pipeline(r1)
                l2 = Preprocessor.lemmatizer_pipeline(r2)
            if cache: # don't overwrite existing data if cache=False
                if lemmatize:
                    pickle.dump((id1, id2, r1, r2, l1, l2, l), open(cached_path, "wb")) # Store the new data as the current cache
                else:
                    pickle.dump((id1, id2, r1, r2, t1, t2, l), open(cached_path, "wb")) # Store the new data as the current cache
        # reduce number of examples in each file if needed for testing
        # if 'max_m' in opt:
        #     r1, r2, l = reduce_examples([r1, r2, l], opt['max_m'])
        #     if not id1 is None:
        #         id1, id2, l = reduce_examples([id1, id2, l], opt['max_m'])
        #     if lemmatize:
        #         l1,l2 = reduce_examples([l1, l2], opt['max_m'])
        ID1.append(id1)
        ID2.append(id2)
        R1.append(r1)
        R2.append(r2)
        L.append(l)
        if lemmatize:
            L1.append(l1)
            L2.append(l2)
        else:
            L1 = None
            L2 = None
            T1.append(t1)
            T2.append(t2)
    return {'ID1': ID1, 'ID2': ID2, 'R1': R1, 'R2': R2,'T1': T1, 'T2': T2, 'L1': L1, 'L2': L2,'L': L}


def load_data(opt,cache=True,numerical=True,onehot=False, write_vocab=False):
    """
    Reads data and does preprocessing based on options file and returns a data dictionary.
    Tokens will always be loaded, other keys depend on settings and will contain None if not available.
    :param opt: option dictionary, containing task and dataset info
    :param numerical: map tokens to embedding ids or not
    :param onehot: load labels as one hot representation or not
    :param write_vocab: write vocabulary to file or not
    :param cache: try to use cached preprocessed data or not
    :return: 
        { # essential:
        'ID1': ID1, 'ID2': ID2, # doc ids
        'R1': R1, 'R2': R2, # raw text
        'L': L, # labels
          # optional for word embds:
        'E1': E1, 'E2': E2, # embedding ids
        'embd': embd, # word embedding matrix
        'mapping_rates': mapping_rates,  
          # optional for topics:
        'D_T1':D_T1,'D_T2':D_T2, # document topics
        'word_topics':word_topics, # word topic matrix
        'topic_keys':topic_key_table} # key word explanation for topics
    """
    E1 = None
    E2 = None
    L1 = None
    L2 = None
    D_T1 = None
    D_T2 = None
    W_T1 = None
    W_T2 = None
    topic_key_table = None
    mapping_rates = None
    embd = None
    word_topics = None
    vocab = []
    word2id = None
    id2word = None

    # get options
    dataset = opt['dataset']
    module_name = "src.loaders.{}.build".format(dataset)
    my_module = importlib.import_module(module_name)
    my_module.build(opt) # download and reformat if not existing
    topic_scope = opt.get('topic','')
    topic_update = opt.get('topic_update', False)
    assert topic_update in [True,False] # no  backward compatibility
    assert topic_scope in ['', 'word', 'doc', 'word+doc']
    recover_topic_peaks = opt['unflat_topics'] =opt.get('unflat_topics', False)
    pretrained = opt.get('pretrained_embeddings', None)  # [GoogleNews,GoogleNews-reduced,Twitter,SemEval]
    w2v_limit = opt.get('w2v_limit', None)
    assert w2v_limit is None # discontinued
    calculate_mapping_rate = opt.get('mapping_rate', False)
    dim = opt.get('embedding_dim', 300)
    padding = opt.get('padding', False)
    simple_padding = opt.get('simple_padding', True)
    if padding:
        Warning('L_R_padding is discontinued. Using simple_padding instead.')
        simple_padding = True
    L_R_unk = opt.get('unk_sub', False)
    tasks = opt.get('tasks', '')
    assert len(tasks)>0
    num_topics = opt.get('num_topics',None)
    unk_topic = opt['unk_topic'] = opt.get('unk_topic', 'uniform')
    assert unk_topic in ['uniform','zero','min','small']
    lemmatize = opt.get('lemmatize',False)
    # ToDo: prevent using too large matrices when max_length > longest sentence
    s1_max_len,s2_max_len,max_len = get_dataset_max_length(opt) #maximum number of tokens in sentence
    max_m = opt.get('max_m',None) # maximum number of examples

    # load or create cache
    cache = load_cache_or_process(opt, cache, onehot, vocab) # load max_m examples
    ID1 = cache['ID1']
    ID2 = cache['ID2']
    R1 = cache['R1']
    R2 = cache['R2']
    T1 = cache['T1']
    T2 = cache['T2']
    L1 = cache['L1']
    L2 = cache['L2']
    L = cache['L']

    # map words to embedding ids
    if numerical:
        print('Mapping words to embedding ids...')
        processor_output = Preprocessor.map_files_to_ids(T1,T2, max_len, calculate_mapping_rate,
                                               simple_padding=simple_padding, L_R_unk=L_R_unk)
        print('Finished word id mapping.')
        E1 = processor_output['E1']
        E2 = processor_output['E2']
        word2id = processor_output['word2id']
        id2word = processor_output['id2word']

        # load embeddings and assign
        if simple_padding:
            padding_tokens = 1
        else:
            padding_tokens = 0
        if L_R_unk:
            unk_tokens = 2
        else:
            unk_tokens = 1
        if not pretrained is None:
            vocab,embd = Preprocessor.load_embd_cache(pretrained,dataset,tasks[0], word2id, w2v_limit, dim,padding_tokens,unk_tokens)

        mapping_rates = processor_output['mapping_rates']
        # if write_vocab:
        #     vocab_name = str(pretrained) + '_' + str(dim) + 'dim_' + str(w2v_limit) + 'lim'
        #     write_vocabulary(vocab,opt,vocab_name)
        #  reduce embd id length for questions to save computational resources
        if not s1_max_len==max_len:
            E1 = reduce_embd_id_len(E1, tasks, cutoff=s1_max_len)
        if not s2_max_len==max_len:
            E2 = reduce_embd_id_len(E2, tasks, cutoff=s2_max_len)

    # reduce number of examples after mapping words to ids to ensure static mapping regardless of max_m
    if not ID1 is None:
        ID1 = reduce_examples(ID1, max_m)
        ID2 = reduce_examples(ID2, max_m)
    R1 = reduce_examples(R1, max_m)
    R2 = reduce_examples(R2, max_m)
    T1 = reduce_examples(T1, max_m)
    T2 = reduce_examples(T2, max_m)
    if not E1 is None:
        E1 = reduce_examples(E1, max_m)
        E2 = reduce_examples(E2, max_m)
    if not L1 is None:
        L1 = reduce_examples(L1, max_m)
        L2 = reduce_examples(L2, max_m)
    L = reduce_examples(L, max_m)

    # load topic related data
    if 'word' in topic_scope:
        word_topics = load_word_topics(opt,recover_topic_peaks=recover_topic_peaks)
        word2id_dict = word_topics['word_id_dict']
        # word_topics.get('topic_matrix', None)
        # todo: ensure mapping between tokenized and lemmatised text (so that embd at pos i corresponds to topic distribution at pos i)
        # map lemmatised tokens to word distribution
        if lemmatize:
            doc1 = L1
            doc2 = L2
        else:
            doc1 = T1
            doc2 = T2
        # map word to topic id
        # if topic_update is None:
        #     # for backward compatibility
        #     print('Mapping words to topic distributions...')
        #     W_T1 = [Preprocessor.lookup_word_dist(r,id2topic_dist,num_topics,s1_max_len,unk_topic) for r in doc1]
        #     W_T2 = [Preprocessor.lookup_word_dist(r,id2topic_dist,num_topics,s2_max_len,unk_topic) for r in doc2]
        # else:
        print('Mapping words to topic ids...')
        W_T1 = [Preprocessor.map_topics_to_id(r,word2id_dict,s1_max_len) for r in doc1]
        W_T2 = [Preprocessor.map_topics_to_id(r,word2id_dict,s2_max_len) for r in doc2]

    if 'doc' in topic_scope:
        doc_topics = load_document_topics(opt,recover_topic_peaks=recover_topic_peaks)
        D_T1 = doc_topics['D_T1']
        D_T2 = doc_topics['D_T2']
    if ('word' in topic_scope) or ('doc' in topic_scope):
        topic_key_table = read_topic_key_table(opt)

    print('Done.')
    return {'ID1': ID1, 'ID2': ID2, # doc ids
            'R1': R1, 'R2': R2, # raw text
            'T1': T1, 'T2': T2,  # tokenized text
            'L1': L1, 'L2': L2, # lemmatized text
            'E1': E1, 'E2': E2, # embedding ids
            'W_T1': W_T1, 'W_T2': W_T2, # word topic ids ()
            'D_T1':D_T1,'D_T2':D_T2, # document topics
            'L': L, # labels
            # misc
            'mapping_rates': mapping_rates,  # optional
            'embd': embd,
            'id2word':id2word,
            'word2id':word2id,
            'word_topics':word_topics,
            'topic_keys':topic_key_table}

if __name__ == '__main__':

    # Example usage
    opt = {'dataset': 'MSRP', 'datapath': 'data/',
           'tasks': ['B'],'max_length':'minimum',
           'subsets': ['train','dev','test'],
           'model': 'affinity_cnn', 'load_ids':True,'lemmatize':False,'cache':True,
           'w2v_limit': None,'embedding_dim': 200,#'pretrained_embeddings': 'Deriu',
           'topic':'word','num_topics':70,'topic_type':'ldamallet'
           }

    data_dict = load_data(opt, cache=True, numerical=True, onehot=False)

    # data_dict
    data_dict['ID1']
    data_dict['embd']

    # R1 = data_dict['R1']
    # R2 = data_dict['R2']

    if opt.get('lemmatize',False):
        L1 = data_dict['L1']
        L2 = data_dict['L2']
    else:
        R1 = data_dict['R1']
        R2 = data_dict['R2']

    E1 = data_dict['E1']
    E2 = data_dict['E2']

    W_T1 = data_dict['W_T1']
    W_T2 = data_dict['W_T2']
    data_dict['word_topics']['topic_matrix']

    # # word topics
    # W_T1 = data_dict['W_T1']
    # default_vector =
    # for split in range(len(W_T1)):
    #     # print(split)
    #     for batch in range(len(W_T1[split])):
    #         # print(batch)
    #         for example in range(len(W_T1[split][batch])):
    #             # print(example)
    #             for word in range(len(W_T1[split][batch][example])):
    #                 print(word)
    #                 break

    # i = 200
    # if not len(L1[0][i])==len(R1[0][i].split(' ')):
    #     print(R1[0][i].split(' '))
    #     print(L1[0][i])
    #
    # E1 = data_dict['E1']
    # E2 = data_dict['E2']
    # E1[0].shape #(3169, 100)
    # E2[0].shape #(3169, 100)
    #
    # if opt.get('topic','')=='doc':
    #     data_dict['D_T1'][0].shape #(3169, 20)
    #     data_dict['D_T2'][0].shape #(3169, 20)
    #     data_dict['topic_keys'] # key word explanation for topics
    #
    # if opt.get('topic','')=='word':
    #     W_T1 = data_dict['W_T1']
    #     data_dict['word_topics']['topic_dict']
    #     data_dict['word_topics']['topic_matrix']
    #     data_dict['topic_keys'] # key word explanation for topics
    #
    #
