import argparse

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.BIMPM import BIMPM
from model.utils import DataSet
from os.path import join as join_path, dirname
from torchtext.vocab import Vectors
from time import gmtime, strftime


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        s1, s2, label = 'q1', 'q2', 'label'

        s1, s2, label = getattr(batch, s1), getattr(batch, s2), getattr(batch, label)
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
        batch_loss = F.cross_entropy(pred, label)
        loss += batch_loss.data

        # _, pred = pred.max(dim=1)
        # acc += (pred == batch.label).sum().float()
        # corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()

        acc += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
        size += len(pred)
    if args.cuda:
        acc = acc.item()
    acc /= size
    # acc = acc.cpu().data
    acc = 100 * acc
    return loss, acc


def load_model(args, data):
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--use-char-emb', default=True, action='store_true')
    parser.add_argument('--word-dim', default=64, type=int)

    parser.add_argument('--model-path', required=True)

    args = parser.parse_args()

    # if args.data_type == 'SNLI':
    #     print('loading SNLI data...')
    #     data = SNLI(args)
    # elif args.data_type == 'Quora':
    #     print('loading Quora data...')
    #     data = Quora(args)
    # data = DataSet(args)
    #
    # setattr(args, 'char_vocab_size', len(data.char_vocab))
    # setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    # setattr(args, 'class_size', len(data.LABEL.vocab))
    # setattr(args, 'max_word_len', data.max_word_len)
    #
    # print('loading model...')
    # model = load_model(args, data)

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
    # args.cuda = False

    model = load_model(args, data)

    _, acc = test(model, args, data)

    print(f'test acc: {acc:.3f}')
