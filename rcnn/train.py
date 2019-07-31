import os
import sys
import torch
import torch.nn.functional as F
from os.path import join as join_path,dirname

path_root = dirname(dirname(__file__))


def train(train_iter,dev_iter,model,args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1,args.epochs + 1): # epochs所有训练数据的训练次数
        print('Epoch:', epoch)
        for batch in train_iter:
            feature,target = batch.text,batch.label
            feature.t_(), target.sub_(1) # t_() 是转置  sub_(1) 将每个元素减去1
            # target = torch.autograd.Variable(target).long()
            if args.cuda:
                feature,target = feature.cuda(),target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits,1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size)) 

            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter,model,args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                else: #长时间没有提升
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt

def eval(data_iter,model,args):
    model.eval()
    corrects,avg_loss = 0,0
    with torch.no_grad():
        for batch in data_iter:
            feature,target = batch.text,batch.label
            feature.t_(),target.sub_(1)
            # target = torch.autograd.Variable(target).long()
            if args.cuda:
                feature,target = feature.cuda(),target.cuda()
            logits = model(feature)
            loss = F.cross_entropy(logits,target)
            avg_loss += loss.item()
            corrects += (torch.max(logits,1)[1].view(target.size()).data == target.data).sum()
        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = 100 * corrects / size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                        accuracy,
                                                                        corrects,
                                                                        size))
    return accuracy

def save(model,save_dir,save_prefix,steps):
    save_dir = join_path(path_root,save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir,save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(),save_path)