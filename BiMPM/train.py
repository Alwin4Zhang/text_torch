import argparse
import copy
import os
from os.path import join as join_path, dirname
import torch
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import gmtime, strftime
import torch.nn.functional as F
from torchtext.vocab import Vectors

from model.BIMPM import BIMPM
from model.utils import DataSet
from test import test


def train(args, data):
    model = BIMPM(args, data)
    if args.cuda:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    for epoch in range(args.epoch):
        print("当前为训练第{%s}轮" % str(epoch + 1))
        iterator = data.train_iter
        for i, batch in enumerate(iterator):
            present_epoch = int(iterator.epoch)
            if present_epoch == args.epoch:
                break
            if present_epoch > last_epoch:
                print('epoch:', present_epoch + 1)
            last_epoch = present_epoch

            s1, s2, label = 'q1', 'q2', 'label'

            s1, s2, label = getattr(batch, s1), getattr(batch, s2), getattr(batch, label)

            # limit the lengths of input sentences up to max_sent_len
            if args.max_sent_len >= 0:
                if s1.size()[1] > args.max_sent_len:
                    s1 = s1[:, :args.max_sent_len]
                if s2.size()[1] > args.max_sent_len:
                    s2 = s2[:, :args.max_sent_len]

            if args.cuda:
                s1, s2, label = s1.cuda(), s2.cuda(), label.cuda()
            kwargs = {'p': s1, 'h': s2}

            if args.use_char_emb:
                char_p = Variable(torch.LongTensor(data.characterize(s1)))
                char_h = Variable(torch.LongTensor(data.characterize(s2)))

                if args.cuda:
                    char_p = char_p.cuda()
                    char_h = char_h.cuda()

                kwargs['char_p'] = char_p
                kwargs['char_h'] = char_h

            pred = model(**kwargs)

            optimizer.zero_grad()

            loss = F.cross_entropy(pred, label)
            loss += loss.data
            loss.backward()
            optimizer.step()

            if (i + 1) % args.print_freq == 0:
                dev_loss, dev_acc = test(model, args, data, mode='dev')
                test_loss, test_acc = test(model, args, data)
                c = (i + 1) // args.print_freq

                writer.add_scalar('loss/train', loss, c)
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('acc/dev', dev_acc, c)
                writer.add_scalar('loss/test', test_loss, c)
                writer.add_scalar('acc/test', test_acc, c)

                print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
                      f' / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                    max_test_acc = test_acc
                    best_model = copy.deepcopy(model)

                # loss = 0
                model.train()

    writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    # parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--use-char-emb', default=True, action='store_true')
    parser.add_argument('--word-dim', default=64, type=int)
    # device
    # parser.add_argument('-device', type=int, default=1,
    #                     help='device to use for iterable data,-1 mean cpu [default:-1]')
    args = parser.parse_args()

    model_path = join_path(dirname(__file__), 'data/word_vec')
    vectors = Vectors(model_path)
    setattr(args, 'vectors', vectors)
    data = DataSet(args)

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    args.cuda = True if torch.cuda.is_available() else False
    args.device = None if torch.cuda.is_available() else -1
    # args.cuda = False

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    args.data_type = 'atec'
    torch.save(best_model.state_dict(), f'saved_models/BIBPM_{args.data_type}_{args.model_time}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
