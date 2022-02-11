import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def load_data(file_path):
    '''
        argument:
            file_path: ./data/FB15k-237
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entities.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relations.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid_500.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)
    fair_triplets = read_high_triplets(os.path.join(file_path, 'gen2prof_fair'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))
    print('num_fair_triples: {}'.format(len(fair_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets, fair_triplets

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def read_high_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail, relation2, tail2 = line.strip().split('\t')
            triplets.append([(entity2id[head], relation2id[relation], entity2id[tail]), ( entity2id[tail], relation2id[relation2], entity2id[tail2])])

    return np.array(triplets)

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target):
    k, indices = torch.sort(score, dim=1, descending=True)
    #print("O=: ", score)
    #print('R=: ', k)
    #print("==: ", indices)
    #print("=T: ", target.view(-1, 1))
    indices = torch.nonzero(indices == target.view(-1, 1))
    #print(indices)
    indices = indices[:, 1].view(-1)
    #print(indices)
    return indices

def sort_and_ret(score):
    k, indices = torch.sort(score, dim=1, descending=True)
    #print("O=: ", score)
    #print('R=: ', k)
    #print("==: ", indices)
    #print("=T: ", target.view(-1, 1))
    #indices = torch.nonzero(indices == target.view(-1, 1))
    #print(indices)
    indices = indices[:, 0].view(-1)
    #print(indices)
    return k, indices

def fair_metric1(output, idx, sens, target_pool):
    labels = sens
    val_y = labels[idx].cpu().numpy()

    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==int(target_pool[0])
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==int(target_pool[1])
    #print(idx_s1, sum(idx_s1))
    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)
    #print(idx_s0_y1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    #print(pred_y[idx_s1])
    pred_y0 = sum(pred_y)
    pred_y1 = len(idx) - pred_y0
    #print(pred_y0, sum(idx_s0))

    parity = abs(pred_y0/sum(idx_s0)-pred_y1/sum(idx_s1))
    equality = abs(pred_y0/sum(idx_s0_y1)-pred_y1/sum(idx_s1_y1))

    return parity,equality

def fair_metric(output, idx, sens, sens_pool, labels, labels_pool):
    #labels = sens

    labels = labels.cpu().numpy()
    labels = np.array([1 if x==int(labels_pool[0]) else 0 for x in labels])
    labels = torch.from_numpy(labels)

    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==int(sens_pool[0])
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==int(sens_pool[1])

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    #print(pred_y[idx_s0])
    #print(pred_y[idx_s1])

    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality

def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():

        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()

def calc_mrr_fair(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():

        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()

def calc_fair(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():

        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        #head_relation_triplets = all_triplets[:, :2]
        #tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)
        all_len = len(all_triplets)
        head_len = all_len #len(all_triplets[:, :, 0])
        all_head_index = np.array(list(set(np.arange(head_len))))
        all_head_index = torch.from_numpy(all_head_index)
        #print(test_triplets)
        target_list = all_triplets[:, :, 0][:, 0]
        #print(target_list)
        target_pool = torch.unique(all_triplets[:, :, 0][:, 0], return_counts=True)[0]
        count_pool = torch.unique(all_triplets[:, :, 0][:, 0], return_counts=True)[1]
        label_list = all_triplets[:, :, 2][:, 1]
        label_pool = torch.unique(all_triplets[:, :, 2][:, 1], return_counts=True)[0]
        label_count_pool = torch.unique(all_triplets[:, :, 2][:, 1], return_counts=True)[1]
        #print(label_pool)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            #print(test_triplet)
            subject = test_triplet[0][0]
            relation = test_triplet[0][1]
            object_ = test_triplet[0][2]
            relation2 = test_triplet[1][1]
            object2_ = test_triplet[1][2]

            #subject_relation = test_triplet[0] + test_triplet[1]
            #delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            #delete_index = torch.nonzero(delete_index == 2).squeeze()

            #delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            #perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            #perturb_entity_index = torch.from_numpy(np.array([4810, 5245]))
            #perturb_entity_index = torch.from_numpy(np.array(list(np.arange(1))))
            #rest = np.array(target_pool)
            #if int(object2_) == target_pool[0]:
            #    rest = np.array([5245, 4810])
            #    target = torch.tensor(0)
            #else:
            #    target = torch.tensor(1)
            #perturb_entity_index = torch.unique( torch.cat((perturb_entity_index, object2_.view(-1))) )
            perturb_entity_index = label_pool
            #target = torch.tensor(len(perturb_entity_index) - 1)
            #print(perturb_entity_index)

            emb_ar = embedding[subject] * w[relation] * embedding[object_] * w[relation2]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            #target = torch.tensor(len(perturb_entity_index) - 1)
            #print(target)
            ranks_s.append(sort_and_ret(score))

            # Perturb subject
            #object_ = test_triplet[2]
            #relation = test_triplet[1]
            #subject = test_triplet[0]

            #object_relation = torch.tensor([object_, relation])
            #delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            #delete_index = torch.nonzero(delete_index == 2).squeeze()

            #delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            #perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            #perturb_entity_index = torch.from_numpy(perturb_entity_index)
            #perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            #emb_ar = embedding[object_] * w[relation]
            #emb_ar = emb_ar.view(-1, 1, 1)

            #emb_c = embedding[perturb_entity_index]
            #emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            #out_prod = torch.bmm(emb_ar, emb_c)
            #score = torch.sum(out_prod, dim = 0)
            #score = torch.sigmoid(score)

            #target = torch.tensor(len(perturb_entity_index) - 1)
            #ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        #print(ranks_s)
        #ranks_o = torch.cat(ranks_o)

        ranks = ranks_s
        p, e = fair_metric(ranks, all_head_index, target_list, target_pool, label_list, label_pool)
        print("parity: {:.6f} | equality: {:.6f}".format(p, e))

    #return mrr.item()
