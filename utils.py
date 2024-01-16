import numpy as np
import torch
import scipy.sparse as ssp
import sklearn
# from numpy.core.tests.test_umath import simple, TestAbsoluteNegative, TestFPClass
from sklearn.cluster import KMeans
from sklearn import preprocessing
from torch.utils import data
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import h5py
import time
import random
import heapq
import multiprocessing as mp
from tqdm import tqdm
import networkx as nx
from sklearn import metrics
import warnings, sys, os.path
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
# from torch_geometric.transforms import LineGraph
# from torch_geometric.data import Data

def get_data():
    '''
    读取数据,返回正类样本(标号为0)和负类样本(标号为1)
    '''

    path1 = 'GCIL/data/flare-F_train.h5'
    path2 = 'GCIL/data/flare-F_val.h5'

    data_set = np.zeros((0, 12))

    with h5py.File(path1, 'r') as hf:
        posData = np.asarray(hf[str(0)])[:20]
        negData = np.asarray(hf[str(1)])[:200]

    with h5py.File(path2, 'r') as hf:
        for i in range(2):
            data = np.asarray(hf[str(i)])
            label = np.asarray([i for m in range(len(data))])
            label = label[np.newaxis, :]
            data = np.c_[data, label.T]
            data_set = np.concatenate((data_set, data), axis = 0)
    test_data, test_label = data_set[:, :-1], data_set[:, -1]

    posData = posData
    negData = negData

    return posData, negData, test_data, test_label

def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1,1)
    node_tag.scatter_(1, tags, 1)
    return node_tag

def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_s = edge_fea[sr[0,:], :]
    fea_s = fea_s.repeat(2,1)
    fea_r = edge_fea[sr[1,:], :]
    fea_r = fea_r.repeat(2,1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body

def random_torch(feature,label):
    index = [i for i in range(len(label))]
    random.shuffle(index)
    return feature[index],label[index]

def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))

def iter_data(data_arrays, batch_size, is_train=True):  #
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)    #data_arrays=(fetures,labels)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)    

def LinkSampleUndersample_2(posData, negData):
    min_max_scaler = preprocessing.MinMaxScaler()
    # po_num = posData.shape[0]
    # ne_num = negData.shape[0]
    po_num = posData.shape[0]
    ne_num = negData.shape[0]
    posData = min_max_scaler.fit_transform(posData)
    negData = min_max_scaler.fit_transform(negData)
    po_train = posData[:po_num] 
    ne_train = negData[:ne_num]

    train_neg, train_pos, negKey = balance_sample(po_train,ne_train)

    # test = torch.FloatTensor(test)

    # trainLabel = [0 for _ in range(30)]
    # trainLabel.extend([1 for _ in range(3000)])
    #
    # testLabel = [0 for _ in range(10)]
    # testLabel.extend([1 for _ in range(100)])

    return train_neg, train_pos, negKey

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(list(edge_features.values())[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}

    def helper(A, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in
                                                   zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end - start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)


def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def KeyNegdata(posData, negData):
    # 1. qiu de mei ge fu lei yang ben de zhu yi li quan zhong
    softmax = torch.nn.Softmax()
    posNumber = len(posData)
    posNumber2 = 2 * posNumber
    negSoftmax = []
    negSoftmax2 = []
    negFin = []
    for i in range(len(negData)):
        sim = cosine_similarity( np.array([negData[i,:]]),negData)
        softmax_sim =  softmax(torch.tensor(sim[0]))
        negSoftmax.append(softmax_sim[i])
    # 2. an quan zhong cong xiao dao da xuan ze qian 2*zheng lei shu liang ge

    arr_min = heapq.nsmallest(posNumber2, negSoftmax)
    index_min = map(negSoftmax.index, arr_min)
    list_min = list(index_min)

    # 3. jiang zhe 2n ge yu zheng lei yang beng die jia zai qiu zhu yi li quan zhong
    for i in range(posNumber2):
        allData = np.concatenate((np.array([negData[list_min[i],:]]), posData), axis=0)
        sim = cosine_similarity( np.array([negData[list_min[i],:]]),allData)
        softmax_sim2 =  softmax(torch.tensor(sim[0]))
        negSoftmax2.append(softmax_sim2[0])

    # 4. an cong xiao dao da xuan ze hou zheng lei shu liang ge  fu lei yang ben
    arr_max = heapq.nlargest(posNumber, negSoftmax2)
    index_max = map(negSoftmax2.index, arr_max)
    list_max = list(index_max)

    for i in range(posNumber):
        j = list_max[i]
        negFin.append(np.array([negData[list_min[j],:]]))

    return negFin

def random_sampler(data_0, scale):
    index = random.sample(range(0, 200), scale)
    data_0 = np.asarray(data_0)
    data_0_sample = data_0[index][:]
    return data_0_sample

def balance_sample(posData, negData):
    # min_max_scaler = preprocessing.MinMaxScaler()
    # posData = min_max_scaler.fit_transform(posData)
    # negData = min_max_scaler.fit_transform(negData)

    train_pos = ([], [])
    train_neg = ([], [])
    negFin = []

    posNumber = len(posData)
    negNumber = len(negData)

    # neg - neg
    negFin = KeyNegdata(posData, negData)

    # cluster-based undersampling
    # estimator = KMeans(n_clusters=posNumber)
    # estimator.fit(negData)
    # centroids = estimator.cluster_centers_
    # negFin = torch.FloatTensor(centroids)

    # random undersampling
    # negFin = random_sampler(negData, 20)

    posData = torch.FloatTensor(posData)
    negData1 = torch.FloatTensor(negData)
    negFin = torch.FloatTensor(negFin)
    negData_g = []
    for i in range(posNumber):
        for j in range(negNumber):

            if (eucliDist(negData1[j],negFin[i]).sum() == 0):
                continue

            if (eucliDist(negData1[j],negFin[i]).sum() > (1.2 * eucliDist(negData1[j],posData[i]).sum())):
                continue

            train_neg[0].append(negFin[i])
            train_neg[1].append(negData1[j])
            train_pos[0].append(posData[i])
            train_pos[1].append(negData1[j])
            negData_g.append(negData[j])

    # # pos - neg
    # for i in range(posNumber):
    #     for j in range(negNumber):
    #         train_pos[0].append(posData[i])
    #         train_pos[1].append(negData[j])
    train_pos = train_pos[:80000]
    train_neg = train_neg[:80000]
    negData_g = negData_g[:40]

    return train_neg, train_pos, negData_g

def balance_sample_2(posData, negData,p,q,test=False,pre=False):
    min_max_scaler = preprocessing.MinMaxScaler()
    # NewSample = []
    NewSample = torch.FloatTensor([])
    NewLabel = []

    posNumber = len(posData)
    negNumber = len(negData)

    posData = torch.FloatTensor(posData)
    negData = torch.FloatTensor(negData)

    # neg - neg
    if not test and not pre:
        estimator = KMeans(n_clusters=posNumber)
        estimator.fit(negData)
        centroids = estimator.cluster_centers_
        centroids = torch.FloatTensor(centroids)
        for i in range(negNumber):
            for j in range(posNumber):
                # tmp = centroids[j]*p + negData[i]
                # NewSample.append(tmp)
                tmp = (centroids[j])*0.01 + (negData[i])
                np.append(tmp, 1)
                np.append(tmp, 1)
                NewSample = torch.cat((NewSample,tmp.unsqueeze(0)), 0)
                NewLabel.append(1) 

    # pos - neg
    for i in range(posNumber):
        for j in range(negNumber):
            # tmp = posData[i] + negData[j]*q
            # NewSample.append(tmp)
            tmp = (posData[i]) + (negData[j])*0.01
            np.append(tmp, 0)
            np.append(tmp, 1)
            NewSample = torch.cat((NewSample,tmp.unsqueeze(0)), 0)
            NewLabel.append(0)

    NewLabel = torch.LongTensor(NewLabel)
    # print(NewLabel)
    if not test and not pre:
        NewSample, NewLabel = random_torch(NewSample, NewLabel)
    # NewSample = torch.FloatTensor([item.detach().numpy() for item in NewSample])
    # NewSample = min_max_scaler.fit_transform(NewSample)
    # print([item.detach().numpy()[0] for item in NewSample])
    # NewSample = min_max_scaler.fit_transform([item.detach().numpy() for item in NewSample])
    NewSample = torch.FloatTensor(NewSample)
    

    if not test:
        return NewSample,NewLabel  
    return NewSample




def LinkSampleUndersample(posData, negData,pre=False,ne_test_num = 100):
    '''
    先分割再采样,训练集为30:3000.
    验证集10:100,从训练集抽100个负类样本拼接
    测试集12:100,和整个训练集负类样本拼接
    '''
    po_num = 30
    ne_num = 3000
    val_num = 10
    po_train = posData[:po_num] 
    ne_train = negData[:ne_num]
    ne_val = ne_train[:ne_test_num]
    po_val = np.concatenate((posData[po_num : po_num + val_num],
                            negData[ne_num : ne_num + val_num*10]))
    po_test = np.concatenate((posData[po_num + val_num : po_num+ val_num + 10],
                            negData[ne_num + val_num*10 : ne_num + val_num*10 + 100]))
    # x_val = torch.FloatTensor(po_val)
    # x_te = torch.FloatTensor(po_test)

    y_val = [0 for _ in range(len(posData[po_num:po_num+val_num]))]
    y_val.extend([1 for _ in range(len(negData[ne_num:ne_num+val_num*10]))])
    y_val = torch.LongTensor(y_val)

    y_te = [0 for _ in range(len(posData[po_num + val_num : po_num+ val_num + 10]))]
    y_te.extend([1 for _ in range(len(negData[ne_num + val_num*10 : ne_num + val_num*10 + 100]))])
    y_te = torch.LongTensor(y_te)

    return po_train,ne_train,po_val,po_test,ne_val,y_val,y_te

#用训练好的模型来预测测试集，然后按照训练的方式来采样测试集
def train_sample(x_te,output):

    pos_idx = torch.nonzero(output==0)[:,-1]
    # pre_real_pos = torch.cat((output[pos_idx].unsqueeze(0),y_te[pos_idx].unsqueeze(0)),axis=0)
    neg_idx = torch.nonzero(output==1)[:,-1]
    # pre_real_neg = torch.cat((output[neg_idx].unsqueeze(0),y_te[neg_idx].unsqueeze(0)),axis=0)
    # pre_real = torch.cat((pre_real_pos,pre_real_neg),axis=1)
    idx = torch.cat((neg_idx,pos_idx))
    print(x_te[pos_idx].shape,x_te[neg_idx].shape)
    x,y = balance_sample_2(x_te[pos_idx].numpy(),x_te[neg_idx].numpy(),pre=True) 

    return x,idx


def test(x_te,y_te,model,ne_test_number,pre=False):

    out_label = []
    for j in range(len(y_te)):
        output = model(x_te[j*ne_test_number:(j+1)*ne_test_number])     
        preds = output.max(1)[1].type_as(y_te)
        for i in range(0,len(preds),ne_test_number):
            temp = preds[i:i+ne_test_number]
            # print(temp) 
            nozero_count = torch.count_nonzero(temp).item()
            if nozero_count >= ne_test_number/10:
                out_label.append(1)
            else:
                out_label.append(0)
    out_label = torch.LongTensor(out_label)

    # print(out_label)
    auc_test = roc_auc_score(y_te, out_label, average = 'macro')

    conf_matrix = sklearn.metrics.confusion_matrix(y_te, out_label)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0

    print("Test set results:",
        'AUC: %.4f\t Spec: %.4f\t'
        'Sen: %.4f\t' % (auc_test,spec,sen))
    if pre : 
        return out_label

    return auc_test

def test_total(x_te,y_te,model,ne_test_number,pre=False):
    out_label = []
    output = model(x_te)
    preds = output.max(1)[1].type_as(y_te)

    for i in range(0,len(preds),ne_test_number):
        temp = preds[i:i+ne_test_number]
        # print(temp) 
        nozero_count = torch.count_nonzero(temp).item()
        if nozero_count >= ne_test_number/5:
            out_label.append(1)
        else:
            out_label.append(0)
    out_label = torch.LongTensor(out_label)

    # print(out_label)
    auc_test = roc_auc_score(y_te, out_label, average = 'macro')

    conf_matrix = sklearn.metrics.confusion_matrix(y_te, out_label)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0

    print("Test set results:",
        'AUC: %.4f\t Spec: %.4f\t'
        'Sen: %.4f\t' % (auc_test,spec,sen))
    if pre : 
        return out_label

    return auc_test
    

def test_train_sample(x_te,y_te,model,ne_test_number):
   
    out_label = []
    output = model(x_te)
    preds = output.max(1)[1].type_as(y_te)
    pos_test_number = y_te.shape[0] - ne_test_number 
    for i in range(0,pos_test_number * ne_test_number,pos_test_number):
            temp = preds[i : i + pos_test_number]
            # print(temp)
            nozero_count = torch.count_nonzero(temp).item()
            if nozero_count >= ne_test_number/2:
                out_label.append(1)
            else:
                out_label.append(0)

    for i in range(0,pos_test_number * ne_test_number,ne_test_number):
        temp = preds[i : i + ne_test_number]
        # print(temp)
        nozero_count = torch.count_nonzero(temp).item()
        if nozero_count >= ne_test_number/5:
            out_label.append(1)
        else:
            out_label.append(0)

    out_label = torch.LongTensor(out_label)

    auc_test = roc_auc_score(y_te, out_label, average = 'macro')

    conf_matrix = sklearn.metrics.confusion_matrix(y_te, out_label)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0

    print("train_sample Test set results:",
        'AUC: %.4f\t Spec: %.4f\t'
        'Sen: %.4f\t' % (auc_test,spec,sen))

    return out_label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def auc(output, labels):
    # preds = output.max(1)[1].type_as(labels)
    auc = roc_auc_score(labels, output, average = 'macro')
    return auc

def confusionMatrix(output, labels):
    # preds = output.max(1)[1].type_as(labels)
    conf_matrix = sklearn.metrics.confusion_matrix(labels, output)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
    return spec, sen

# shuffle data and label
def random_shuffle(data, label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum) 
    np.random.shuffle(label)
    return data, label

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def train(epochs,pos_data,neg_data,pos_val,neg_val,y_val,model, optimizer,ne_test_number=100):
    t = time.time()
    # N = len(features[idx_train])
    np.random.shuffle(pos_data) 
    np.random.shuffle(neg_data)
    batch_num = 5
    pos_prm = len(pos_data)//batch_num
    neg_prm = len(neg_data)//batch_num
    model.train()

    for i in range(batch_num):
        optimizer.zero_grad()
        pos_batch = pos_data[i * pos_prm : (i+1) * pos_prm]
        neg_batch = neg_data[i * neg_prm : (i+1) * neg_prm]
        # label_batch = labels[perm[i:i+batchsize]]
        output,label = model(pos_batch,neg_batch)
        loss_train = F.nll_loss(output, label)
        loss_train.backward()
        optimizer.step()

# 验证集测试，未投票
    model.eval()
    # output = model(pos_val,neg_val)

    print('Epoch: {:04d}'.format(epochs),
        'loss_train: {:.4f}'.format(loss_train.data.item()),
        # 'acc_train: {:.4f}'.format(acc_train.data.item()),
        # 'loss_val: {:.4f}'.format(loss_val.data.item()),
        # 'acc_val: {:.4f}'.format(acc_val.data.item()),
        'time: {:.4f}s'.format(time.time() - t),
        # 'AUC: %.4f\t Spec: %.4f\n'
        # 'Sen: %.4f\t' % (auc_test, spec, sen)
        )

    # auc = test_total(x_val,y_val,model,ne_test_number=100)
    auc = test_new(pos_val,neg_val,y_val,model,ne_test_number=ne_test_number)
    # return loss_val.data.item()
    return auc

def test_new(pos_data,neg_data,y_te,model,ne_test_number=100):
    out_label = []
    output = model(pos_data,neg_data,test=True)
    preds = output.max(1)[1].type_as(y_te)

    for i in range(0,len(preds),ne_test_number):
        temp = preds[i:i+ne_test_number]
        # print(temp) 
        nozero_count = torch.count_nonzero(temp).item()
        if nozero_count >= ne_test_number/2:
            out_label.append(1)
        else:
            out_label.append(0)
    out_label = torch.LongTensor(out_label)

    # print(out_label)
    auc_test = roc_auc_score(y_te, out_label, average = 'macro')

    conf_matrix = sklearn.metrics.confusion_matrix(y_te, out_label)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0

    print("Test set results:",
        'AUC: %.4f\t Spec: %.4f\t'
        'Sen: %.4f\t' % (auc_test,spec,sen))
    
    return auc_test
    