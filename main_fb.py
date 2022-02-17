import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torchtool_fb import EarlyStopping

from test_fb import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr, calc_fair
from models_fair1 import RGCN

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.all_edge, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding) #+ fair_sum

    return loss

def valid(valid_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr

def test(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])
    #with open("log/fb_layer_" + {str(args.nlayer)} + "_link_"+str(args.regularization), 'a') as f:
    with open(f"log/fb_layer_{args.nlayer}_mu_{args.a}_reg_{args.regularization}", 'a') as f:
        f.write("Model: {:d}\n".format(args.b1))
        f.write("MRR (filtered): {:.6f}\n".format(mrr))

    return mrr

def fair_test(fair_triplets, model, test_graph, fair_all_triplets):
    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    p1, e1, p2, e2 = calc_fair(entity_embedding, model.relation_embedding, fair_triplets, fair_all_triplets, hits=[1, 3, 10])
    #with open("log/fb_layer_" + {str(args.nlayer)} + "_link_"+str(args.regularization), 'a') as f:
    with open(f"log/fb_layer_{args.nlayer}_mu_{args.a}_reg_{args.regularization}", 'a') as f:
        f.write("Task1: parity: {:.6f} | equality: {:.6f}\n".format(p1, e1))
        f.write("Task2: parity: {:.6f} | equality: {:.6f}\n".format(p2, e2))

def main(args):

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0
    _train = int(args.tran)
    evalu = int(args.eval)
    infer = int(args.infer)

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets, fair_triplets = load_data('./data/FB15k-237')
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    fair_all_triplets = torch.LongTensor(fair_triplets)
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)

    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    fair_triplets = torch.LongTensor(fair_triplets)

    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout, a = args.a, b1 = args.b1, b2 = args.b2, n_layer = args.nlayer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)

    if use_cuda:
        model.cuda()

    if _train:
        early_stopping = EarlyStopping(patience=10, verbose=True, path=f'best_mrr_model_fb_reg{args.regularization}_mu{args.a}_layer{args.nlayer}.pth')

        for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):

            model.train()
            optimizer.zero_grad()

            loss = train(train_triplets, model, use_cuda, batch_size=args.graph_batch_size, split_size=args.graph_split_size,
                negative_sample=args.negative_sample, reg_ratio = args.regularization, num_entities=len(entity2id), num_relations=len(relation2id))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            if epoch % args.evaluate_every == 0:

                tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

                if use_cuda:
                    model.cpu()

                model.eval()
                valid_mrr = valid(valid_triplets, model, test_graph, all_triplets)
                #early_stopping(loss.item(),  model, path='best_mrr_model_fb_reg{}_mu{}_layer{}.pth'.format(args.regularization, args.a, args.nlayer))
                #early_stopping(loss.item(),  model)
                early_stopping(valid_mrr,  model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if use_cuda:
                    model.cuda()
    if evalu:
        if use_cuda:
             model.cpu()

        model.eval()

        checkpoint = torch.load('best_mrr_model_fb_reg{}_mu{}_layer{}.pth'.format(args.regularization, args.a, args.nlayer))
        model.load_state_dict(checkpoint)#['state_dict'])

        test(test_triplets, model, test_graph, all_triplets)
        fair_test(fair_triplets, model, test_graph, fair_all_triplets)

    if infer:
        if use_cuda:
             model.cpu()

        model.eval()

        checkpoint = torch.load('weight/best_mrr_model_fb_reg{}_mu{}_layer{}.pth'.format(args.regularization, args.a, args.nlayer))
        model.load_state_dict(checkpoint)#['state_dict'])

        test(test_triplets, model, test_graph, all_triplets)
        fair_test(fair_triplets, model, test_graph, fair_all_triplets)


if __name__ == '__main__':
    SEED = 2021
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    parser = argparse.ArgumentParser(description='Fair-KGNN')

    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=200)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b1", type=int, default=1)
    parser.add_argument("--b2", type=float, default=1.0)
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--tran", type=int, default=1)
    parser.add_argument("--eval", type=int, default=1)
    parser.add_argument("--infer", type=int, default=0)

    args = parser.parse_args()
    print(args)

    main(args)
