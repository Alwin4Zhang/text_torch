from collections import Counter
from os.path import dirname,join as join_path

data_root_path = dirname(dirname(__file__))

def positive_sample_percentage(filename):
    tot,pos = 0,0
    for line in open(filename):
        line = line.strip().split('\t')
        if line[-1] == '1':
            pos += 1
        tot += 1
    print(pos/float(tot))

def sentence_length_distribution(filename):
    tot,pos = 0,0
    pos_seq_len = []
    neg_seq_len = []
    tot_seq_len = []
    for line in open(filename):
        line = line.strip().split('\t')
        s1,s2 = line[1],line[2]
        tot_seq_len.extend([len(s1),len(s2)])
        tot += 2
        if line[-1] == '1':
            pos_seq_len.extend([len(s1),len(s2)])
            pos += 2
        else:
            neg_seq_len.extend([len(s1),len(s2)])    
    tot_counter = Counter(tot_seq_len)    
    pos_counter = Counter(pos_seq_len)
    neg_counter = Counter(neg_seq_len)
    tot_freq = sorted(map(lambda x: (x[0], round(x[1]/float(tot), 4)), tot_counter.items()))
    pos_freq = sorted(map(lambda x: (x[0], round(x[1]/float(pos), 4)), pos_counter.items()))
    neg_freq = sorted(map(lambda x: (x[0], round(x[1]/float(tot-pos), 4)), neg_counter.items()))
    print('Total sample length distribution: {}'.format(tot_freq))
    print('Positive sample length distribution: {}'.format(pos_freq))
    print('Negetive sample length distribution: {}'.format(neg_freq))

def pair_length_diff_distribution(filename):
    tot, pos = 0, 0
    tot_diff = []
    pos_diff = []
    neg_diff = []
    for line in open(filename):
        line = line.strip().split('\t')
        s1 = line[1]
        s2 = line[2]
        len_diff = abs(len(s1) - len(s2))
        tot_diff.append(len_diff)
        tot += 1
        if line[-1] == '1':
            pos_diff.append(len_diff)
            pos += 1
        else:
            neg_diff.append(len_diff)
    tot_counter = Counter(tot_diff)
    pos_counter = Counter(pos_diff)
    neg_counter = Counter(neg_diff)
    tot_freq = sorted(map(lambda x: (x[0], round(x[1] / float(tot), 4)), tot_counter.items()))
    pos_freq = sorted(map(lambda x: (x[0], round(x[1] / float(pos), 4)), pos_counter.items()))
    neg_freq = sorted(map(lambda x: (x[0], round(x[1] / float(tot - pos), 4)), neg_counter.items()))
    print('Total pair length diff distribution: {}'.format(tot_freq))
    print('-' * 100)
    print('Positive pair length diff distribution: {}'.format(pos_freq))
    print('-' * 100)
    print('Negetive pair length diff distribution: {}'.format(neg_freq))

if __name__ == '__main__':
    filename = join_path(data_root_path,'data/atec_nlp_sim_train.csv')
    # positive_sample_percentage(filename)
    # sentence_length_distribution(filename)
    pair_length_diff_distribution(filename)