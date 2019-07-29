import tensorflow as tf
from tensorflow.python.framework import ops

from src.evaluation.visualise_attention import plot_affinity_matrix
from src.loaders.load_data import load_data
from src.logs.training_logs import read_all_training_logs, find_entry
from src.models.save_load import load_model_and_graph, get_model_dir, run_restored_tensor
from src.preprocessing.Preprocessor import Preprocessor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
import argparse

# read logs

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--debug",
    type="bool",
    nargs="?",
    const=True,
    default=False,
    help="Use debugger to track down bad values during training. ")
parser.add_argument('-model', action="store", dest="model", type=str, default=None)
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.model is None:
    models =     [# Quora Task B
    # 3586,
    4151,4156
    # 3606, 3624, 3655, 3663, 3633, 3639, 3607,3625,3652,3662,3634,3638
    # 130
    # 73,74,75,76,78,79,80,81
    # 3585
    # 3583,3584,
    # 3579,3581,3580,  # separate topic affinities
    # 3564,3565,3567,3566,3569,3572,3574,3573,3568,3545,3571,3570, 3576,3577# best tuned model ablation
    # 3511,3512,3520,3515,3554,3556, 3555, 3557,3532,3533,3526,3535 # fixed grid search
                 ]
else:
    print('input: "{}"'.format(FLAGS.model))
    models = [int(m) for m in FLAGS.model.split(',')]
print(models)

for model_id in models:

    # for alpha in [0.1,1,10,50]:
        # read log
    vm_path = model_id>3000
    if vm_path:
        local_log_path = 'data/VM_logs'
    else:
        local_log_path = 'data/logs'
    log = read_all_training_logs(local_log_path)

    # find model in log via id
    opt = find_entry(model_id,log)
    # opt['num_epochs']=15
    # opt['num_topics']=20
    # opt['unk_topic'] = 'zero'
    # opt['unflat_topics'] = True
    # opt['topic_alpha']=alpha
    print(opt)

    if opt['affinity'] is None:
        print('No affinity in architecture to plot. Shutting down.')

    else:
        model_path = get_model_dir(opt,VM_copy=vm_path)
        print(model_path)
        epochs = opt['num_epochs']

        # load graph and restore weights
        ops.reset_default_graph()
        sess = tf.Session()
        load_model_and_graph(opt, sess, epochs, vm_path)

        # load original data to pass through network for inspection
        if opt['dataset']=='Quora':
            examples = [441,  # Po
                        89,  # Pn
                        5874,  # No
                        396]  # Nn
        elif opt['dataset'] == 'Semeval':
            examples =[441,
                      89,
                      396]
        elif opt['dataset'] == 'MSRP':
            examples =[256] # No
        else:
            ValueError('Example ids for {} not defined'.format(opt['dataset']))
        opt['max_m'] = max(examples)+1

        #  prepare input to be passed through network
        data_dict = load_data(opt)
        print(data_dict['embd'].shape)
        subset = 1 # 0 for train, 1 for dev, 2 for test
        ID1 = data_dict['ID1'][subset]
        ID2 = data_dict['ID2'][subset]
        R1 = data_dict['R1'][subset]
        R2 = data_dict['R2'][subset]
        E1 = data_dict['E1'][subset]
        E2 = data_dict['E2'][subset]
        W_T1 = data_dict['W_T1']
        W_T2 = data_dict['W_T2']
        D_T1 = data_dict['D_T1']
        D_T2 = data_dict['D_T2']
        T1 = data_dict['T1'][subset]
        T2 = data_dict['T2'][subset]
        L = data_dict['L'][subset]

        # specific examples
        for i in examples:
            id_left = ID1[i]
            id_right = ID2[i]
            x1 = E1[i]
            x2 = E2[i]
            if not W_T1 is None:
                w_t1 = W_T1[subset][i]
                w_t2 = W_T2[subset][i]
            if not D_T1 is None:
                d_t1 = D_T1[subset][i]
                d_t2 = D_T2[subset][i]
            true_label = L[i]
            left_sentence_split = [data_dict['id2word'][w] for w in x1]
            right_sentence_split = [data_dict['id2word'][w] for w in x2]

            # execute TF graph
            A_mask = 'attention_{}/masked/A'.format(opt['encoder'])
            A_topic = 'attention_{}/masked/A'.format('topic')
            A_topic_softmax = 'attention_{}/softmax/A'.format('topic')
            A_softmax = 'attention_{}/softmax/A'.format(opt['encoder'])

            if '_col' in opt['affinity'] or '_row' in opt['affinity']:
                tensornames = [A_softmax, 'evaluation_metrics/predict', 'cost/main_cost/cost', 'softmax_layer/fully_connected/BiasAdd']
            else:
                tensornames = [A_mask, 'evaluation_metrics/predict', 'cost/main_cost/cost', 'softmax_layer/fully_connected/BiasAdd']

            if FLAGS.debug:
                E, left, right = run_restored_tensor(['embedding/W', 'embedding_lookup_L', 'embedding_lookup_R'], [x1], [x2], sess,[true_label])
                print(E)
                print(right)

            if 'separate' in opt['model']:
                # get attention matrix after softmax if necessary for topics, too
                if ('topic_affinity' in opt.keys()) and (not opt['topic_affinity'] is None) and ('_col' in opt['topic_affinity'] or '_row' in opt['topic_affinity']):
                    tensornames = [A_topic_softmax] + tensornames  # also get affinity matrix for topics
                else:
                    tensornames = [A_topic]+tensornames # also get affinity matrix for topics
                if W_T1 is None and D_T1 is None:
                    a_topic, a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames,[x1], [x2], sess, [true_label])
                # doc topic
                elif W_T1 is None and (not D_T1 is None):
                    a_topic,a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                    left=[x1], right=[x2], sess=sess, label=[true_label], d_tl=[d_t1], d_tr=[d_t2])
                # word topic
                elif (not W_T1 is None) and (D_T1 is None):
                    # tensornames.append('embedding_lookup_2')
                    # tensornames.append('embedding_lookup_3')
                    # a_topic, a_emb, predicted, main_loss, logits,left,right = run_restored_tensor(tensornames=tensornames,
                    # left=[x1], right=[x2], sess=sess, label=[true_label], w_tl=[w_t1], w_tr=[w_t2])
                    a_topic, a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                    left=[x1], right=[x2], sess=sess, label=[true_label], w_tl=[w_t1], w_tr=[w_t2])
                # word+doc topic
                elif (not W_T1 is None) and (not D_T1 is None):
                    a_topic, a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                    left=[x1], right=[x2], sess=sess, label=[true_label], d_tl=[d_t1], d_tr=[d_t2], w_tl=[w_t1], w_tr=[w_t2])
            else:
                if W_T1 is None and D_T1 is None:
                    a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames, [x1], [x2], sess,[true_label])
                # doc topic
                elif W_T1 is None and (not D_T1 is None):
                    a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                                                                              left=[x1], right=[x2], sess=sess,
                                                                              label=[true_label], d_tl=[d_t1],
                                                                              d_tr=[d_t2])
                # word topic
                elif (not W_T1 is None) and (D_T1 is None):
                    a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                                                                              left=[x1], right=[x2], sess=sess,
                                                                              label=[true_label], w_tl=[w_t1],
                                                                              w_tr=[w_t2])
                # word+doc topic
                elif (not W_T1 is None) and (not D_T1 is None):
                    a_emb, predicted, main_loss, logits = run_restored_tensor(tensornames=tensornames,
                                                                              left=[x1], right=[x2], sess=sess,
                                                                              label=[true_label], d_tl=[d_t1],
                                                                              d_tr=[d_t2], w_tl=[w_t1], w_tr=[w_t2])
            conf_score = sess.run(tf.nn.softmax(logits[0]))

            if opt['dataset']=='MSRP':
                limit = 31
            else:
                limit = 20

            print(main_loss)
            plot_affinity_matrix(a_emb[0][0:limit, 0:limit], left_sentence_split[:limit], right_sentence_split[:limit], True,
                                 predicted[0], true_label, conf_score, opt, id_left, id_right,
                                 fig_folder='/Users/nicole/Documents/Latex/Papers/2019-02_ACL/fig/affinity_plots/',  # '/Users/nicole/Documents/Latex/Papers/2018-PhD/fig/affinity_plots/'
                                 alpha=opt["aux_loss"])
            attention_sum = a_emb[0].sum()
            print('Attention sum: {}'.format(attention_sum))

            if 'separate' in opt['model']:
                # recover sentence from topic model
                left_sentence_split = R1[i].split(' ')
                right_sentence_split = R2[i].split(' ')
                # add padding
                if len(left_sentence_split)<limit:
                    padding_tokens = ['<PAD>']*(limit - len(left_sentence_split))
                    left_sentence_split = left_sentence_split + padding_tokens
                if len(right_sentence_split)<limit:
                    padding_tokens = ['<PAD>']*(limit - len(right_sentence_split))
                    right_sentence_split = right_sentence_split + padding_tokens
                # substitute non-topic
                if not data_dict['word_topics'] is None:
                    if opt.get('topic_update', None) is None:
                        # substitute non-topic words with UNK [backward compatability]
                        vocab = data_dict['word_topics']['complete_topic_dict'].keys()
                        left_sentence_split = [t if t in vocab or t == '<PAD>' else '<UNK>' for t in left_sentence_split]
                        right_sentence_split = [t if t in vocab or t == '<PAD>' else '<UNK>' for t in right_sentence_split]
                    else:
                        # use topic dictionary to reconstruct sentence
                        left_sentence_split = [data_dict['word_topics']['topic_dict'][t] for t in w_t1]
                        right_sentence_split = [data_dict['word_topics']['topic_dict'][t] for t in w_t2]
                # plot separate affinity matrix for topics
                # print(a_topic[0][0:limit, 0:limit])
                plot_affinity_matrix(a_topic[0][0:limit, 0:limit], left_sentence_split[:limit],
                                     right_sentence_split[:limit], True,
                                     predicted[0], true_label, conf_score, opt, id_left, id_right,
                                     fig_folder='/Users/nicole/Documents/Latex/Papers/2019-02_ACL/fig/affinity_plots/',
                                     # '/Users/nicole/Documents/Latex/Papers/2018-PhD/fig/affinity_plots/'
                                     alpha=opt["aux_loss"],topic=True,fixed_scale=True,alpha_in_filename=True)
        break
        sess.close()

