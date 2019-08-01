import random
from os.path import dirname,join as join_path

data_root_path = dirname(dirname(__file__))

def train_test_split(infile,test_rate=0.2):
    train_path = join_path(data_root_path,'data/train.csv')
    test_path = join_path(data_root_path,'data/test.csv')

    with open(train_path,'w') as f_train,open(test_path,'w') as f_test:
        for line in open(infile):
            if random.random() > test_rate:
                f_train.write(line)
            else:
                f_test.write(line)

if __name__ == "__main__":
    data_dir = join_path(data_root_path,'data/atec_nlp_sim_train.csv')
    train_test_split(data_dir)
