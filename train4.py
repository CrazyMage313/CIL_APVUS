import numpy as np
from model import ConGAT, SpConGAT, ConGraph, SiameseNetwork
import argparse
# import sys, os.path
# sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
import torch
import torch.nn
from chainer import cuda, optimizers
import pickle as pickle
import random
import torch.optim as optim
# from torch_geometric.data import DataLoader
from torch.autograd import Variable
from sklearn import tree, svm
# from torch_cluster import knn_graph
import torch.nn.functional as F
import sklearn
from sklearn.metrics import roc_auc_score
import time
# import utils
# from dgl.nn.pytorch.factory import KNNGraph
from utils import *
from main import *
import model
import scipy.sparse as sp
import h5py
from model import Net
from scipy import spatial
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import random


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A-B), 2)))

def get_model_optimizer(args):
    model = SiameseNetwork()

    if args.gpu >= 0:
        model.to_gpu()

    if args.optimizer == 'SGD':
        optimizer = optimizers.MomentumSGD(lr = args.lr, momentum = args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = optimizers.Adam()
    optimizer.setup(model)

    return model, optimizer

# def train(epochs):
#     print(epochs)
#     t = time.time()
#     # N = len(features[idx_train])
#     N = len(train_n1)
#     perm = np.random.permutation(N)
#     batchsize = 2500
#     loss_train1 = 0
#     loss_train = 0
#     model.train()
#
#     for i in range(0, N, batchsize):
#         optimizer.zero_grad()
#         x_batch1 = train_n1[perm[i:i+batchsize]]
#         x_batch2 = train_n2[perm[i:i+batchsize]]
#         x_batch3 = train_p1[perm[i:i+batchsize]]
#         # x_batch4 = train_p2[perm[i:i+batchsize]]
#         # label_batch = labels[perm[i:i+batchsize]]
#         output1 =  model.forward(x_batch2, x_batch1)
#         mean1 = output1.mean()
#         # print(mean1)
#         output2 =  model.forward(x_batch2, x_batch3)
#         mean2 = output2.mean()
#         # print(mean2)
#         result1 = torch.cat([output1, output2], 0)
#
#
#         # input = torch.cat([x_batch2, x_batch3], 0)
#         # output3 = - model.single_forward(input)
#         # output4 = - model.single_forward(x_batch3)
#
#
#         # output3 = - model.single_forward(x_batch2)
#         # mean3 = output3.mean()
#         # output4 = - model.single_forward(x_batch3)
#         # result2 = torch.cat([output3, output4], 0)
#
#         sim_np1 = result1.detach().numpy()
#         # # sim_pj = np.sum(sim_np1)/3300
#         # p = (mean1 + mean2).detach().numpy() /2
#         # p = 0.58
#         #
#         label1 = []
#         # # label2 = []
#         #
#         for i in range(5000):
#             if sim_np1[i]<(0.7275):
#                 label1.append(1)
#             else:
#                 label1.append(0)
#
#         # sim_np2 = result2.detach().numpy()
#         # # sim_pj = np.sum(sim_np1)/3300
#         # p = (mean3 + mean4).detach().numpy() /2
#         #
#         #
#         # for i in range(6000):
#         #     if sim_np2[i]<(p):
#         #         label2.append(0)
#         #     else:
#         #         label2.append(1)
#
#         # result = torch.nn.Sigmoid(result)
#         label_batch = []
#
#         # output4 = model(x_batch4)
#
#         label_batch = [1 for _ in range(2500)]
#         label_batch.extend([0 for _ in range(2500)])
#         label_batch = torch.FloatTensor(label_batch)
#         # loss_train = F.nll_loss(result, label_batch)
#         label1 = torch.FloatTensor(label1)
#         # label2 = torch.FloatTensor(label2)
#         loss_train1 = loss_train1 + nn.BCELoss()(label1,label_batch)
#         # preds = output3.max(1)[1].type_as(label_batch)
#         # loss_train2 = nn.BCELoss()(preds,label_batch)
#         # loss_train = loss_train1 + loss_train2
#
#         spec, sen = confusionMatrix(label1, label_batch)
#         auc_test = auc(label1, label_batch)
#
#         print("Train set results:",
#               'AUC: %.4f\t Spec: %.4f\t'
#               'Sen: %.4f\t' % (auc_test, spec, sen))
#
#     loss_train = loss_train1 / (N/batchsize)
#     # loss_train_f2 = loss_train2 / (N/batchsize)
#     # loss_train = loss_train_f1 + loss_train_f2
#
#     loss_train.requires_grad = True
#     loss_train.backward()
#     print("Loss: %.4f\t" % (loss_train))
#     optimizer.step()
#
#     return loss_train1.data.item()
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


def train(args, train_neg, train_pos, model, optimizer):
    train_neg1 = torch.stack(train_neg[0], 0).squeeze(1)
    # train_n1 = Variable(train_neg1, requires_grad = True)
    train_neg2 = torch.stack(train_neg[1], 0).squeeze(1)
    # train_n2 = Variable(train_neg2, requires_grad = True)
    train_pos1 = torch.stack(train_pos[0], 0).squeeze(1)
    # train_p1 = Variable(train_pos1, requires_grad = True)
    # train_pos2 = torch.stack(train_pos[1], 0).squeeze(1)
    # train_p2 = Variable(train_pos2, requires_grad = True)
    xp = cuda.cupy if args.gpu >= 0 else np
    N = len(train_neg1)
    losses = []
    data_set = []
    data = {}
    args.batchsize = N
    # args.batchsize = int(N/10)
    # data = np.hstack((train_data, train_label.reshape(N,1)))
    # t1 = xp.asarray(t1, dtype=np.float32)
    # # t1 = torch.from_numpy(t1)
    # label = xp.array(label, dtype=np.int32).flatten()
    # label = torch.from_numpy(label)

    for epoch in range(1, args.epoch + 1):
        print(epoch)
        mean_loss = 0
        perm = np.random.permutation(N)
        for i in range(0, N, args.batchsize):
            batch_n1 = np.array(train_neg1)[perm[i:i + args.batchsize]]
            batch_n2 = np.array(train_neg2)[perm[i:i + args.batchsize]]
            batch_p1 = np.array(train_pos1)[perm[i:i + args.batchsize]]
            # label = np.array(labels)[perm[i:i + args.batchsize]]

            # x0_batch = torch.cat([batch_n1, batch_n1], 0)
            # x1_batch = torch.cat([batch_n2, batch_p1], 0)
            x0_batch = np.concatenate((batch_n1, batch_p1), 0)
            x1_batch = np.concatenate((batch_n2, batch_n2), 0)

            label_batch = [1 for _ in range(args.batchsize)]
            label_batch.extend([0 for _ in range(args.batchsize)])

            label = xp.array(label_batch, dtype=np.int32).flatten()

            x0_batch = xp.asarray(x0_batch, dtype=np.float32)
            x1_batch = xp.asarray(x1_batch, dtype=np.float32)
            # x0_batch = torch.from_numpy(x0_batch)
            # label = torch.from_numpy(label)

            numIndex = args.batchsize * 2

            index = [i for i in range(numIndex)]
            random.shuffle(index)
            label = label[index]
            x0_batch = x0_batch[index]
            x1_batch = x1_batch[index]

            model.cleargrads()
            # optimizer.zero_grad()
            loss = model.forward(x0_batch, x1_batch, label)
            # loss = model.forward(x0_batch, label)
            # loss.requires_grad_(True)


            d1 = model.forward_dist(x0_batch, x1_batch).data
            # d1 = dist1(y0, y1)
            # y1 = xp.asarray(y1, dtype=np.float32)
            mean = np.mean(d1)
            #
            Y_pr = []

            for i in range(numIndex):
                if d1[i] < mean:
                    Y_pr.append(1)
                else:
                    Y_pr.append(0)

            # label2 = torch.FloatTensor(label)
            Y_pr2 = xp.array(Y_pr, dtype=np.int32).flatten()

            loss = 0.001 * loss + 0.001 * cross_entropy_error(label, Y_pr2)

            loss.backward()
            optimizer.update()

            conf_matrix = sklearn.metrics.confusion_matrix(y_true=label, y_pred=Y_pr)
            spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
            sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
            roc1 = (spec + sen) / 2

            auc = roc_auc_score(label, Y_pr, average='macro')
            print('AUC: %.4f\t Spec: %.4f\n'
                  'Sen: %.4f\t' % (auc, spec, sen))

            mean_loss += float(loss.data) * args.batchsize

        if args.optimizer == 'SGD':
            optimizer.lr = args.lr * (1 + args.gamma * epoch) ** -args.power
        losses.append(mean_loss / N)

        # plt.clf()
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.plot(losses)
        # plt.savefig('loss.png')



        print('epoch:{}/{}\tlr:{}\tloss:{}'.format(
            epoch, args.epoch, optimizer.lr, mean_loss / N
        ))

        if epoch % 10 == 0:
            pickle.dump(model, open('model_{}.pkl'.format(epoch), 'wb'), -1)

    return model

# def dist1(y0, y1):
#     # X = np.vstack([y0, y1])
#
#     # sk = np.var(X, axis = 0, ddof = 1)
#     d1 = np.sqrt(((y0 - y1)**2 ).sum())
#
#     return d1

# def test_knn(args, posData, negData, test_data, test_label):
#     xp = cuda.cupy if args.gpu >= 0 else np
#
#     posData = xp.asarray(posData, dtype = np.float32)
#     negData = xp.asarray(negData, dtype=np.float32)
#
#     train_data = np.concatenate((posData, negData), 0)
#     N_pos = posData.shape[0]
#     N_neg = negData.shape[0]
#     N = train_data.shape[0]
#
#     trainLabel = [0 for _ in range(N_pos)]
#     trainLabel.extend([1 for _ in range(N_neg)])
#
#     train_data = xp.asarray(train_data, dtype=np.float32)
#     test_data = xp.asarray(test_data, dtype=np.float32)
#     train_label = xp.array(trainLabel, dtype=np.int32).flatten()
#     test_label = xp.array(test_label, dtype=np.int32).flatten()
#
#     train_results = xp.empty((train_data.shape[0], 4))
#     for i in range(0, N, args.batchsize):
#         x_batch = train_data[i:i + args.batchsize]
#         x_batch = xp.asarray(x_batch, dtype=xp.float32)
#         y = model.forward_once(x_batch)
#         train_results[i:i + args.batchsize] = y.data
#
#     test_results = xp.empty((test_data.shape[0], 4))
#     for i in range(0, N, args.batchsize):
#         x_batch = test_data[i:i + args.batchsize]
#         x_batch = xp.asarray(x_batch, dtype=xp.float32)
#         y = model.forward_once(x_batch)
#         test_results[i:i + args.batchsize] = y.data
#
#     if args.gpu >= 0:
#         train_results = xp.asnumpy(train_results)
#         test_results = xp.asnumpy(test_results)
#
#     # knn classification
#     knn = KNeighborsClassifier(5)
#     knn.fit(train_results, train_label)
#     y_pr_p = knn.predict_proba(test_results)
#     y_pr = np.argmax(y_pr_p, axis=1)
#
#     # DecisionTreeClassification
#     # clf = tree.DecisionTreeClassifier()
#     # clf = clf.fit(train_results, train_label)
#     # y_pr = clf.predict(test_results)
#
#     # SVM classification
#     # classifier = svm.SVC(C=1, kernel='rbf', gamma = 1, decision_function_shape = 'ovr')
#     # classifier.fit(train_results, train_label)
#     # y_pr = classifier.predict(test_results)
#     #
#     conf_matrix = sklearn.metrics.confusion_matrix(y_true=test_label, y_pred=y_pr)
#     spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
#     sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
#     roc1 = (spec + sen) / 2
#
#     # acc = accuracy_score(test_label, y_pr)
#     # f1 = f1_score(test_label, y_pr, average='mac0=;.pro')
#     # g_mean = geometric_mean_score(test_label, y_pr, average='macro')
#     auc = roc_auc_score(test_label, y_pr, average='macro')
#     print('AUC: %.4f\t Spec: %.4f\n'
#           'Sen: %.4f\t' % (auc, spec, sen))


def compute_test(model, test, negData, test_label):

    # negData = negData.numpy()

    test = np.asarray(test, dtype=np.float32)



    test_pos = ([], [])
    label = []
    #
    for i in range(len(test)):
        for j in range(len(negData)):
            test_pos[0].append(test[i])
            test_pos[1].append(negData[j])
            label.append(test_label[i])
    N = len(test_pos[0])
    args.batchsize = int(N )
    #
    perm = np.random.permutation(N)
    for i in range(0, N, args.batchsize):
        batch_t1 = np.array(test_pos[0])[perm[i:i + args.batchsize]]
        batch_n1 = np.array(test_pos[1])[perm[i:i + args.batchsize]]
        batch_label = np.array(label)[perm[i:i + args.batchsize]]

        x_label = np.array(batch_label, dtype=np.int32).flatten()

        x0_batch = np.asarray(batch_t1, dtype=np.float32)
        x1_batch = np.asarray(batch_n1, dtype=np.float32)

        numIndex = args.batchsize

        index = [i for i in range(numIndex)]
        random.shuffle(index)
        x_label = x_label[index]
        x0_batch = x0_batch[index]
        x1_batch = x1_batch[index]

        d1 = model.forward_dist(x0_batch, x1_batch).data

        mean = np.mean(d1)

        Y_pr = []

        for i in range(numIndex):
            if d1[i] < mean:
                Y_pr.append(0)
            else:
                Y_pr.append(1)

        conf_matrix = sklearn.metrics.confusion_matrix(y_true=x_label, y_pred=Y_pr)
        spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
        sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
        roc1 = (spec + sen) / 2

        auc = roc_auc_score(x_label, Y_pr, average='macro')
        print('AUC: %.4f\t Spec: %.4f\n'
                'Sen: %.4f\t' % (auc, spec, sen))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default = -1, type = int)
    parser.add_argument('--epoch', default=100, type = int)
    parser.add_argument('--batchsize', default = 2500, type = int)
    parser.add_argument('--lr', default=0.05, type = float)
    parser.add_argument('--gamma', default=0.001, type = float)
    parser.add_argument('--power', default=0.75, type = float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--train', default=
                        0, type=int)
    parser.add_argument('--optimizer', default='SGD', type = str,
                        choices = ['SGD', 'Adam'])
    args = parser.parse_args()


    if args.train == 1:
        posData, negData, test_data, test_label = get_data()
        train_neg, train_pos, negKey= LinkSampleUndersample_2(posData, negData)
        model, optimizer = get_model_optimizer(args)
        model = train(args, train_neg, train_pos, model, optimizer)
        pickle.dump(model, open('model_{}.pkl'.format(args.epoch), 'wb'), -1)
    else:
        posData, negData, test_data, test_label = get_data()
        train_neg, train_pos, negData_g = LinkSampleUndersample_2(posData, negData)
        model = pickle.load(open('model_{}.pkl'.format(args.epoch), 'rb'))
        # test_knn(args, posData, negData, test_data, test_label)
        compute_test(model, test_data, negData_g, test_label)




