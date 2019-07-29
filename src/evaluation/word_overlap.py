import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.difficulty.difficulty_cases import load_subset_pred_overlap
from src.loaders.load_data import load_data
from src.models.save_load import get_model_dir


def calculate_word_overlap_ratio(opt):
    '''
    Loads dataset defined by opt, calculates word overlap ratio for each sentence pair and returns nested list with overlap 
    ratios in each subset.
    :param opt: option dictionary to load dataset
    :return : nested list with overlap ratios
    '''
    data_dict = load_data(opt, numerical=True)
    E1 = data_dict['E1']
    E2 = data_dict['E2']
    t = opt['tasks'][0]
    d = opt['dataset']
    colors = ['g','b','r']
    subset_overlap = []
    for n,s in enumerate(opt['subsets']):
        identical_per_pair = []
        for i in range(len(E1[n])):
    #         print(line)
            len_1 = (E1[n][i] != 1).sum()
            len_2 = (E2[n][i] != 2).sum()
            total_len = len_1 +len_2
            identical_ids = np.intersect1d(E1[n][i], E2[n][i])
            left_overlap = np.isin(E1[n][i], identical_ids).sum()
            right_overlap = np.isin(E2[n][i], identical_ids).sum()
            overlap_ratio = (left_overlap + right_overlap)/total_len
    #         print(overlap_ratio)
            identical_per_pair.append(overlap_ratio)
        plt.hist(identical_per_pair, color=colors[n], alpha=0.5,bins=25, label=s,range=[0, 1])
        subset_overlap.append(identical_per_pair)
    plt.legend(loc='upper right')
    plt.ylabel('Number of document pairs')
    plt.xlabel('Lexical overlap ratio')
    plt.title('Lexical overlap in {} task {}'.format(d,t))
    plt.show()
    plt.close()
    return subset_overlap


def plot_overlap_violin(opt, overlapping_list, VM_path=True, plot_folder='',scaled=False,subsets='one',classes=4):
    # if VM_path:
    #     logs = read_all_training_logs(log_dir='data/VM_logs/')
    # else:
    #     logs = read_all_training_logs(log_dir='data/logs/')
    # opt = find_entry(model_id,logs)
    print('Loading predictions for '+opt['dataset']+' '+opt['tasks'][0])
    # dev_df = load_subset_pred_overlap(opt, 'dev', overlapping_list[1], VM_path)
    if subsets=='one':
        if type(opt['id'])==str:
            subsets=['primary']
        else:
            subsets = ['test']
    elif subsets=='all':
        if type(opt['id'])==str:
            subsets=['primary','contrastive1','contrastive2']
        else:
            subsets = ['dev','test']
    test_df = None
    for s in subsets:
        try:
            test_df = load_subset_pred_overlap(opt, s, overlapping_list[2], VM_path)
            test_df['model']=opt['id']
            plot_type = 'violin'
            my_pal = {True: "g", False: "r"}
            title = 'Predictions of model {}_{} on {} {} (test)'.format(opt['id'],s,opt['dataset'],opt['tasks'][0])
            if classes==4:
                if scaled:
                    ax = sns.violinplot(hue=test_df['correct_pred'], y=test_df['overlapping'], x=test_df['gold_label'], palette=my_pal, scale='count')
                    suffix = '_scaled'
                else:
                    ax = sns.violinplot(hue=test_df['correct_pred'], y=test_df['overlapping'],x=test_df['gold_label'], palette=my_pal)
                    suffix = ''
            elif classes==2:
                if scaled:
                    ax = sns.violinplot(x=test_df['correct_pred'], y=test_df['overlapping'],
                                        palette=my_pal, scale='count')
                    suffix = '_scaled'
                else:
                    ax = sns.violinplot(x=test_df['correct_pred'], y=test_df['overlapping'],
                                        palette=my_pal)
                    suffix = ''
            ax.set_ylabel("Lexical overlap ratio")
            ax.set_title(title)
#             plt.legend(loc='lower right')
            if plot_folder== '':
                plt.show()
            else:
                plot_path = plot_folder + plot_type + '_' + opt['tasks'][0] + '_' + s + suffix + ".png"
                plt.savefig(plot_path)
                plt.close()
        except FileNotFoundError:
            print('File not found for: {} - {}'.format(opt['id'],s))
    return test_df

def plot_difficulty_overlap_violin(opt, overlapping_list, VM_path=True, plot_folder='',scaled=False,subsets='one'):
    # if VM_path:
    #     logs = read_all_training_logs(log_dir='data/VM_logs/')
    # else:
    #     logs = read_all_training_logs(log_dir='data/logs/')
    # opt = find_entry(model_id,logs)
    print('Loading predictions for '+opt['dataset']+' '+opt['tasks'][0])
    # dev_df = load_subset_pred_overlap(opt, 'dev', overlapping_list[1], VM_path)
    if subsets=='one':
        if type(opt['id'])==str:
            subsets=['primary']
        else:
            subsets = ['test']
    elif subsets=='all':
        if type(opt['id'])==str:
            subsets=['primary','contrastive1','contrastive2']
        else:
            subsets = ['dev','test']
    test_df = None
    for s in subsets:
        try:
            test_df = load_subset_pred_overlap(opt, s, overlapping_list[2], VM_path)
            test_df['model']=opt['id']
            plot_type = 'violin'
            my_pal = {True: "g", False: "r"}
            title = 'Predictions of model {}_{} on {} {} (test)'.format(opt['id'],s,opt['dataset'],opt['tasks'][0])
            median = test_df['overlapping'].median()
            test_df['difficulty'] = ''
            test_df.loc[(test_df['overlapping'] <= median) & (test_df['gold_label'] == 0), 'difficulty'] = 'easy'
            test_df.loc[(test_df['overlapping'] > median) & (test_df['gold_label'] == 1), 'difficulty'] = 'easy'
            test_df.loc[(test_df['overlapping'] <= median) & (test_df['gold_label'] == 1), 'difficulty'] = 'difficult'
            test_df.loc[(test_df['overlapping'] > median) & (test_df['gold_label'] == 0), 'difficulty'] = 'difficult'
            my_pal = {True: "g", False: "r"}
            ax = sns.violinplot(y=test_df['overlapping'], hue=test_df['correct_pred'], x=test_df['difficulty'],
                           palette=my_pal, scale='count', split=True)
            plt.axhline(median, color='black', linewidth=0.5)
            ax.set_ylabel("Lexical overlap ratio")
            ax.set_title(title)
#             plt.legend(loc='lower right')
            if plot_folder== '':
                plt.show()
            else:
                suffix = 'difficulty'
                plot_path = plot_folder + plot_type + '_' + opt['tasks'][0] + '_' + s + suffix + ".png"
                plt.savefig(plot_path)
                plt.close()
        except FileNotFoundError:
            print('File not found for: {} - {}'.format(opt['id'],s))
    return test_df


def plot_multi_model_overlap_violin(opt, joined_dfs, plot_folder='', scaled=False, plot_type='violin_summary',suffix=''):
    if scaled:
        suffix += '_scaled'
        scale = 'count'
    else:
        suffix += ''
        scale='area'
    if plot_type=='violin_difficulty':
        my_pal = {'easy': "pink", 'difficult': "purple"}
        correct_df = joined_dfs[joined_dfs['correct_pred'] == True]
        ax = sns.violinplot(y=correct_df['overlapping'], x=correct_df['model'], hue=correct_df['difficulty'], palette=my_pal,
                       scale='count', split=True)
        median = joined_dfs['overlapping'].median()
        title = 'Correct model predictions vs. difficulty on {} {} (test)'.format(opt['dataset'], opt['tasks'][0])
        plt.axhline(median, color='black', linewidth=0.5)
    elif plot_type == 'violin_summary':
        my_pal = {True: "g", False: "r"}
        correct_df = joined_dfs[joined_dfs['correct_pred'] == True]
        ax = sns.violinplot(hue=correct_df['pred_label'], y=correct_df['overlapping'], x=correct_df['model'],
                            palette=my_pal, scale=scale, split=True)
        title = 'Correct model predictions on {} {} (test)'.format(opt['dataset'], opt['tasks'][0])
    ax.set_ylabel("Lexical overlap ratio")
    ax.set_title(title)
    if plot_folder== '':
        plt.show()
    else:
        plot_path = plot_folder + plot_type + '_' + opt['tasks'][0] + '_' + suffix + ".png"
        plt.savefig(plot_path)
        plt.close()
    # return test_df

def annotate_difficulty(test_df):
    median = test_df['overlapping'].median()
    test_df['difficulty'] =''
    test_df.loc[(test_df['overlapping']<=median) & (test_df['gold_label']==0), 'difficulty'] = 'easy'
    test_df.loc[(test_df['overlapping']>median) & (test_df['gold_label']==1), 'difficulty'] = 'easy'
    test_df.loc[(test_df['overlapping']<=median) & (test_df['gold_label']==1), 'difficulty'] = 'difficult'
    test_df.loc[(test_df['overlapping']>median) & (test_df['gold_label']==0), 'difficulty'] = 'difficult'
    return test_df

def plot_overlap(opt, overlapping_list, VM_path=True, plot_folder='', plot_type='gold_label',metric=''):
    # if VM_path:
    #     logs = read_all_training_logs(log_dir='data/VM_logs/')
    # else:
    #     logs = read_all_training_logs(log_dir='data/logs/')
    # opt = find_entry(model_id,logs)
    print('Loading predictions for '+opt['dataset']+' '+opt['tasks'][0])
    # dev_df = load_subset_pred_overlap(opt, 'dev', overlapping_list[1], VM_path)
    if type(opt['id'])==str:
        subsets=['primary','contrastive1','contrastive2']
    else:
        subsets = ['test']
        # subsets = ['dev','test']
    for s in subsets:
        try:
            test_df = load_subset_pred_overlap(opt, s, overlapping_list[2], VM_path)
            if plot_type== 'gold_label':
                title = 'Labels by word overlap in {} {} (test)'.format(opt['dataset'],opt['tasks'][0])
                test_df[test_df['gold_label']==1]['overlapping'].plot.hist(alpha=0.5,bins=50,color='green',range=[0, 1],label='Positive')
                ax = test_df[test_df['gold_label']==0]['overlapping'].plot.hist(alpha=0.5,bins=50,color='red',title=title,range=[0, 1],label='Negative')
                ax.set_xlabel(metric)
            elif plot_type== 'all_pred':
                title = 'Predictions by word overlap of model {} ({} {} test)'.format(opt['id'], opt['dataset'],opt['tasks'][0])
                test_df[test_df['correct_pred'] == True]['overlapping'].plot.hist(alpha=0.5, bins=50, color='green', range=[0, 1], label='Correct')
                ax = test_df[test_df['correct_pred'] == False]['overlapping'].plot.hist(alpha=0.5, bins=50, color='red',title=title, range=[0, 1],label='Incorrect')
                ax.set_xlabel(metric)
            elif plot_type== 'pos_case':
                title = 'Predictions of model {} by word overlap ({} {} test)'.format(opt['id'],opt['dataset'],opt['tasks'][0])
                try:
                    test_df[test_df['gold_label'] == 1][test_df['correct_pred'] == True]['overlapping'].plot.hist(alpha=0.5,bins=50,color='green',title=title,range=[0, 1],label='Correct')
                    ax = test_df[test_df['gold_label'] == 1][test_df['correct_pred'] == False]['overlapping'].plot.hist(alpha=0.5,bins=50,color='red',title=title,range=[0, 1],label='Incorrect')
                    ax.set_xlabel(metric)
                except TypeError:
                    pass
            elif plot_type== 'neg_case':
                title = 'Predictions of model {} by word overlap ({} {} test)'.format(opt['id'],opt['dataset'],opt['tasks'][0])
                try:
                    test_df[test_df['gold_label'] == 0][test_df['correct_pred'] == True]['overlapping'].plot.hist(alpha=0.5,bins=50,color='green',title=title,range=[0, 1],label='Correct')
                    ax = test_df[test_df['gold_label'] == 0][test_df['correct_pred'] == False]['overlapping'].plot.hist(alpha=0.5,bins=50,color='red',title=title,range=[0, 1],label='Incorrect')
                    ax.set_xlabel(metric)
                except TypeError:
                    pass
            # elif plot_type== 'all_case':
            #     title = 'Predictions of model {} by word overlap ({} {} test)'.format(opt['id'],opt['dataset'],opt['tasks'][0])
            #     try:
            #         test_df[test_df['gold_label'] == 1][test_df['correct_pred'] == True]['overlapping'].plot.hist(alpha=0.5,bins=50,color='green',title=title,range=[0, 1],label='TruePositive')
            #         test_df[test_df['gold_label'] == 1][test_df['correct_pred'] == False]['overlapping'].plot.hist(alpha=0.5,bins=50,color='blue',title=title,range=[0, 1],label='FalseNegative')
            #         test_df[test_df['gold_label'] == 0][test_df['correct_pred'] == True]['overlapping'].plot.hist(alpha=0.5,bins=50,color='red',title=title,range=[0, 1],label='TrueNegative')
            #         ax = test_df[test_df['gold_label'] == 0][test_df['correct_pred'] == False]['overlapping'].plot.hist(alpha=0.5,bins=50,color='yellow',title=title,range=[0, 1],label='FalsePositive')
            #         ax.set_xlabel("Lexical overlap ratio")
            #     except TypeError:
            #         pass
            plt.legend(loc='upper right')
            if plot_folder== '':
                plt.show()
            else:
                plot_path = plot_folder + plot_type + '_' + opt['tasks'][0] + '_' + s + ".png"
                plt.savefig(plot_path)
                plt.close()
        except FileNotFoundError:
            print('File not found for: {} - {}'.format(opt['id'],s))
    # return test_df

if __name__ == '__main__':

    VM_path = True
    for t in ['A']:
        opt = {'dataset': 'Semeval', 'datapath': 'data/',
               'tasks': [t],'n_gram_embd':False,
                'subsets': ['train_large','test2016', 'test2017'],
               'load_ids': False, 'cache':True,
               'simple_padding': True, 'padding': True}
        overlapping_list = calculate_word_overlap_ratio(opt)
        # 'TakeLab-QA'
        all_models = ['Beihang-MSRA', 'KeLP', 'SimBow', 'UPC-USMBA', 'ECNU', 'LS2N', 'SnowMan', 'EICA',
         'LearningToQuestion', 'SwissAlps',
         'FA3L', 'MoRS', 'FuRongWang', 'NLM_NIH', 'Talla', 'GW_QA', 'QU_BIGIR', 'TrentoTeam', 'IIT-UHH', 'SCIR-QA',
         'UINSUSKA-TiTech', 'bunji', 1577, 1378, 2233]

        if t =='A':
            best = ['KeLP','Beihang-MSRA','IIT-UHH','ECNU','bunji','EICA','SwissAlps','FuRongWang','FA3L','SnowMan']
            my = [3402]
        elif t =='B':
            best = ['SimBow','LearningToQuestion','KeLP','Talla','Beihang-MSRA','NLM_NIH','UINSUSKA-TiTech','IIT-UHH','SCIR-QA','FA3L','ECNU','EICA']
            my = [3391]
        elif t == 'C':
            best = ['IIT-UHH','bunji','KeLP','EICA','FuRongWang','ECNU']
            my = [3404]
        else:
            raise ValueError('Task not well defined.')
        comparison = ['perfect', 'random', 'truly_random']

        dfs = []
        for m in comparison : # best[5:10]+my
            opt['id'] = m
            model_folder = get_model_dir(opt, VM_path)
            # plot_overlap(opt, overlapping_list, VM_path=VM_path,plot_folder=model_folder,plot_type='pos_case')
            # plot_overlap(opt, overlapping_list, VM_path=VM_path,plot_folder=model_folder,plot_type='neg_case')
            # model_pred_df = plot_overlap(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder, plot_type='all_pred')
            model_pred_df = plot_overlap_violin(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder,
                                                scaled=False)
            # model_pred_df = plot_difficulty_overlap_violin(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder,scaled=True,subsets='one')
            annotated_df = annotate_difficulty(model_pred_df)
            dfs.append(annotated_df)
        joined_dfs = pd.concat(dfs)
        plot_multi_model_overlap_violin(opt, joined_dfs, scaled=True, plot_type='violin_difficulty',
                                        plot_folder='data/baseline_models/plots/', suffix='comparison')

        dfs = []
        for m in comparison : # best[5:10]+my
            opt['id'] = m
            model_folder = get_model_dir(opt,VM_path)
            # plot_overlap(opt, overlapping_list, VM_path=VM_path,plot_folder=model_folder,plot_type='pos_case')
            # plot_overlap(opt, overlapping_list, VM_path=VM_path,plot_folder=model_folder,plot_type='neg_case')
            # model_pred_df = plot_overlap(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder, plot_type='all_pred')
            model_pred_df = plot_overlap_violin(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder,scaled=False)
            # model_pred_df = plot_difficulty_overlap_violin(opt, overlapping_list, VM_path=VM_path, plot_folder=model_folder,scaled=True,subsets='one')
            annotated_df = annotate_difficulty(model_pred_df)
            dfs.append(annotated_df)
        joined_dfs = pd.concat(dfs)
        plot_multi_model_overlap_violin(opt, joined_dfs, scaled=True, plot_type='violin_summary',plot_folder='data/baseline_models/plots/',suffix='comparison')
        # plot_multi_model_overlap_violin(opt, joined_dfs, scaled=True, plot_type='violin_summary',plot_folder='data/baseline_models/plots/',suffix='comparison')
        # plot_multi_model_overlap_violin(opt, joined_dfs, scaled=False, plot_type='violin_summary', plot_folder='data/baseline_models/plots/',suffix='comparison')



