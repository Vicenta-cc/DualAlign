import json
import torch
import argparse
import numpy as np
import random
import os
import sys
import optuna
import models
import torch.optim as optim
sys.path.append('..')
from Utils.pre_data import datasets
from Utils import utils, edge_rw_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    #utils.parse_optimizer(parser)
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='dblp_acm')
    parser.add_argument('-m', '--method', type=str, help='method used', default='DANN_rw')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='GCN')
    parser.add_argument('--seed', type=int, help='note this should be number of seeds', default=5)
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)

    parser.add_argument('--batch_size', type=int, help='Training batch size', default=1)
    parser.add_argument('--num_layers', type=int, help='Number of embedding layers', default=0)
    parser.add_argument('--dc_layers', type=int, help='Number of domain classification layers', default=2)
    parser.add_argument('--mlp_layers', type=int, help='Number of MLP layers', default=1)
    parser.add_argument('--class_layers', type=int, help='Number of classification layers', default=2)
    parser.add_argument('--K', type=int, help='Number of GNN layers', default=2)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension for GNN', default=128)
    parser.add_argument('--gcn_dim', type=int, help='hidden dimension for GNN', default=128)
    parser.add_argument('--mlp_embed_dim', type=int, help='output dimension for MLP', default=128)
    parser.add_argument('--conv_dim', type=int, help='output dimension for GNN layers', default=16)
    parser.add_argument('--cls_dim', type=int, help='hidden dimension for feature extractor layer', default=16)
    parser.add_argument('--bn', type=bool, help='if use batch normalization', default=False)
    parser.add_argument("--resnet", type=bool, help='if we want to use resnet', default=False)
    parser.add_argument('--best_model', type=bool,help='if use best model from validation results', default=True)
    parser.add_argument('--valid_data', type=str, help='src means use source valid to select model, tgt means use target valid', default='tgt')


    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=300)
    parser.add_argument('--opt', type=str, help='optimizer', default='adam')
    parser.add_argument('--opt_scheduler', type=str, help='optimizer scheduler', default='step')
    parser.add_argument('--opt_decay_step', type=int, default=50)
    parser.add_argument('--opt_decay_rate', type=int, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.007)

    parser.add_argument('--num_nodes', type=int, help='number of nodes for each block of Stochastic block model', default=1000)
    parser.add_argument("--ps", type=float, help="the source intraclass connection probability for SBM dataset", default=0.2)
    parser.add_argument("--qs", type=float, help="the source interclass connection probability for SBM dataset", default=0.02)
    parser.add_argument("--pt", type=float, help="the target intraclass connection probability for SBM dataset", default=0.2)
    parser.add_argument("--qt", type=float, help="the target interclass connection probability for SBM dataset", default=0.016)
    parser.add_argument('--sigma', type=float, help='sigma(std) in the stochastic block model', default=0.8)
    parser.add_argument('--theta', type=float, help='theta in the stochastic block model', default=0)
    parser.add_argument('--sbm_alpha', type=float, help='alpha in the stochastic block model', default=0)

    parser.add_argument('--num_events', type=int, help='number of events for Pileup dataset', default=100)
    parser.add_argument('--edge_feature', type=bool, help='if we want to use edge feature during convolution', default=False)

    parser.add_argument('--reweight', type=bool, help='if we want to reweight the source graph', default=False)
    parser.add_argument('--rw_freq', type=int, help='every number of epochs to compute the new weights based on psuedo-labels', default=20)
    parser.add_argument('--start_epoch', type=int, help='starting epoch for reweighting', default=200)
    parser.add_argument("--rw_lmda", type=float, help='lambda to control the rw', default=0.5)
    parser.add_argument("--use_valid_label", type=bool, help='if we want to use target validation ground truth label in rw', default=True)
    parser.add_argument('--pseudo', type=bool, help='if use pseudo label for reweighting', default=False)
    parser.add_argument("--rw_average", type=bool, help='if we want to average the edge weights when reweighting for multiple graphs', default=False)
    parser.add_argument("--alphatimes", type=float, help='constant in front of the alpha for DANN', default=1.5)
    parser.add_argument("--alphamin", type=float, help='min of alpha for DANN', default=1)
    parser.add_argument("--alphatimes_mlp", type=float, help='constant in front of the alpha for DANN(for FS)', default=1.5)
    parser.add_argument("--alphamin_mlp", type=float, help='min of alpha for DANN(for FS)', default=1)
    parser.add_argument("--alphatimes_gnn", type=float, help='constant in front of the alpha for DANN(for FCSS)', default=1.5)
    parser.add_argument("--alphamin_gnn", type=float, help='min of alpha for DANN(for FCSS)', default=1)

    parser.add_argument('--dir_name', type=str, default='../../DualAlign/main/result')
    parser.add_argument("--src_name", type=str, help='specify for the source dataset name',default='acm')
    parser.add_argument("--tgt_name", type=str, help='specify for the target dataset name',default='dblp')
    parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2005)
    parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2007)
    parser.add_argument("--balanced", type=bool, help='if we keep the label balanced in pileup dataset', default=True)
    parser.add_argument('--plt', type=str, help='plot using tsne or pca', default='pca')
    parser.add_argument("--num_bins", type=int, help='num_bins of calculate_reweight_based_on_distance', default=10)

    return parser.parse_args()





def train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed):
    directory = args.dir_name
    parent_dir = "../Utils/pre_data/"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)
    sys.stdout = utils.Logger(path)

    if isinstance(source_dataset, list):
        input_dim = source_dataset[0].num_node_features
        output_dim = source_dataset[0].num_classes
    else:
        input_dim = source_dataset.num_node_features
        output_dim = source_dataset.num_classes

        # Load the saved MLP model
    checkpoint = torch.load(path + '/best_freeze_model5.pth')
    MLP_encoder = models.MLPs(input_dim, args).to(device)
    MLP_advesarial = models.Adv_DANN(args.mlp_embed_dim, args.cls_dim, 1, args.num_layers, args.dc_layers, args).to(
        device)
    MLP_encoder.load_state_dict(checkpoint['MLP_encoder_state_dict'])
    MLP_advesarial.load_state_dict(checkpoint['MLP_advesarial_state_dict'])


    GNN_input = models.GCN_reweight(args.mlp_embed_dim, args.gcn_dim).to(device)
    GNN_hidden = models.GCN_reweight(args.gcn_dim, args.gcn_dim).to(device)
    GNN_out = models.GCN_reweight(args.gcn_dim, args.conv_dim).to(device)
    GNNs = []
    GNNs.append(GNN_input)
    for i in range(args.K - 2):
        GNNs.append(GNN_hidden)
    GNNs.append(GNN_out)
    GNN_adversarial = models.Adv_DANN(args.conv_dim, args.cls_dim, 1, args.num_layers, args.dc_layers,
                                          args).to(device)
    classifier = models.Classifier(args.conv_dim, args.cls_dim, output_dim, args).to(device)
    best_MLP_encoder = models.MLPs(input_dim, args).to(device)
    # best_GNN = models.GNN_simple(args.mlp_embed_dim, args).to(device)
    best_GNN_input = models.GCN_reweight(args.mlp_embed_dim, args.gcn_dim).to(device)
    best_GNN_hidden = models.GCN_reweight(args.gcn_dim, args.gcn_dim).to(device)
    best_GNN_out = models.GCN_reweight(args.gcn_dim, args.conv_dim).to(device)
    best_GNNs = []
    best_GNNs.append(best_GNN_input)
    for i in range(args.K - 2):
        best_GNNs.append(best_GNN_hidden)
    best_GNNs.append(best_GNN_out)
    best_classifier = models.Classifier(args.conv_dim, args.cls_dim, output_dim, args).to(device)
    best_GNN_state = None
    best_classifier_state = None
    GNN_params = []
    for layer in range(args.K):
        GNN_params.extend(list(GNNs[layer].parameters()))
    total_params = (
            list(MLP_encoder.parameters()) +
            list(MLP_advesarial.parameters()) +
            GNN_params +
            list(GNN_adversarial.parameters()) +
            list(classifier.parameters())
    )
    optimizer = optim.Adam(total_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    # epochs_train = []
    loss_src_train = []
    loss_src_valid = []
    loss_src_test = []
    loss_tgt_valid = []
    loss_tgt_test = []
    edge_weights = [source_dataset.edge_weight.to(device).clone() for _ in range(args.K)]
    best_valid_score = 0
    best_epoch = 0
    rw_rec = 0
    for epoch in range(args.epochs):
        MLP_encoder.train()
        MLP_advesarial.train()
        for layer in range(args.K):
            GNNs[layer].train()
        GNN_adversarial.train()
        classifier.train()

        # Get the MLP features for source and target graphs (use frozen MLP encoder)
        src_data = source_dataset.to(device)
        tgt_data = target_dataset.to(device)

        mlp_src = MLP_encoder(src_data)
        mlp_tgt = MLP_encoder(tgt_data)

        gnn_src = None
        gnn_tgt = None

        if args.reweight and epoch >= (args.start_epoch - 1):
            for layer in range(args.K):
                if layer == 0:
                    embed_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer])
                    embed_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight)
                else:
                    embed_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer])
                    embed_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight)

                # Perform distance-based reweighting at a specified frequency (every rw_freq epochs)
                if (epoch - args.start_epoch - 1) % args.rw_freq == 0:
                    edge_weight = edge_rw_distance.calculate_reweight_return(src_data, tgt_data,
                                                                             embed_src.clone().detach(),
                                                                             embed_tgt.clone().detach(),
                                                                             num_bins=args.num_bins, device=device)
                    edge_weights[layer] = edge_weight
                    # Pass MLP features through GNNs
                    if layer == 0:
                        gnn_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer])
                        gnn_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight)
                    else:
                        gnn_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer])
                        gnn_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight)
                else:
                    gnn_src = embed_src
                    gnn_tgt = embed_tgt
        else:
            for layer in range(args.K):
                if layer == 0:
                    gnn_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer])
                    gnn_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight)
                else:
                    gnn_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer])
                    gnn_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight)

        alpha_gnn = min((args.alphatimes_gnn * (epoch + 1) / args.epochs), args.alphamin_gnn)
        alpha_mlp = min((args.alphatimes_mlp * (epoch + 1) / args.epochs), args.alphamin_mlp)


        final_mlp_src, pred_mlp_domain_src = MLP_advesarial(mlp_src, alpha_mlp)
        final_mlp_tgt, pred_mlp_domain_tgt = MLP_advesarial(mlp_tgt, alpha_mlp)

        final_src, pred_domain_src = GNN_adversarial(gnn_src, alpha_gnn)
        final_tgt, pred_domain_tgt = GNN_adversarial(gnn_tgt, alpha_mlp)

        pred_src = classifier(gnn_src)
        pred_tgt = classifier(gnn_tgt)

        mask_src = src_data.source_training_mask
        label_src = src_data.y[mask_src]
        pred_src = pred_src[mask_src]
        pred_domain_src = pred_domain_src[src_data.source_mask]
        pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]
        pred_mlp_domain_src = pred_mlp_domain_src[src_data.source_mask]
        pred_mlp_domain_tgt = pred_mlp_domain_tgt[tgt_data.target_mask]

        # Construct domain labels: source domain as 0, target domain as 1
        domain_label_src = torch.zeros_like(pred_domain_src)
        domain_label_tgt = torch.ones_like(pred_domain_tgt)

        # Compute source domain classification loss (cross-entropy loss)
        cls_loss_src = utils.CE_loss(pred_src, label_src)
        # Compute domain classification loss from MLP embeddings(using binary cross-entropy)
        domain_loss_src = utils.BCE_loss(pred_domain_src, domain_label_src)
        domain_loss_tgt = utils.BCE_loss(pred_domain_tgt, domain_label_tgt)

        # Compute domain classification loss from GNN embeddings (using binary cross-entropy)
        mlp_domain_loss_src = utils.BCE_loss(pred_mlp_domain_src, domain_label_src)
        mlp_domain_loss_tgt = utils.BCE_loss(pred_mlp_domain_tgt, domain_label_tgt)

        # Compute total loss as the sum of source classification loss and adversarial domain loss
        loss = cls_loss_src + 1 * (domain_loss_src + domain_loss_tgt) + 1 * (mlp_domain_loss_src + mlp_domain_loss_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1 == 0:
            inter_report, loss_dict = evaluate(source_dataset, target_dataset, MLP_encoder, GNNs, classifier,
                                               edge_weights, args, )

            loss_src_train.append(loss_dict['loss_src_train'])
            loss_src_valid.append(loss_dict['loss_src_valid'])
            loss_src_test.append(loss_dict['loss_src_test'])
            loss_tgt_valid.append(loss_dict['loss_tgt_valid'])
            loss_tgt_test.append(loss_dict['loss_tgt_test'])

            if args.valid_data == "src":
                valid_score = inter_report['acc_tgt_test']
            else:
                valid_score = inter_report['acc_tgt_test']
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_epoch = epoch
                # !Note: To train the model yourself rather than using the provided one, please uncomment the following code.

                # for layer in range(args.K):
                #     torch.save(GNNs[layer].state_dict(),
                #                path + '/best_free_GNN_layer_' + str(layer) + '_seed_' + str(seed) + '.pt')
                # torch.save(MLP_encoder.state_dict(), path + '/best_free_H_l_encoder_' + str(seed) + '.pt')
                # torch.save(classifier.state_dict(), path + '/best_free_H_l_classifier_' + str(seed) + '.pt')

    best_MLP_encoder.load_state_dict(torch.load(path + '/best_free_H_l_encoder_' + str(seed) + '.pt'))
    for layer in range(args.K):
        best_GNNs[layer].load_state_dict(
            torch.load(path + '/best_free_GNN_layer_' + str(layer) + '_seed_' + str(seed) + '.pt'))
    best_classifier.load_state_dict(torch.load(path + '/best_free_H_l_classifier_' + str(seed) + '.pt'))

    print("best_valid_score: " + str(best_valid_score))
    final_report, loss_dict = evaluate(source_dataset, target_dataset, best_MLP_encoder, best_GNNs, best_classifier,
                                       edge_weights, args)
    print("rw times: " + str(rw_rec))
    print("best_epoch: " + str(best_epoch))
    print("best_valid_score: " + str(final_report['acc_tgt_test']))
    return final_report




def evaluate(source_dataset, target_dataset, MLP_encoder, GNN, classifier,edge_weights, args):

    loss_src_train, acc_src_train, auc_src_train = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier,edge_weights, args, "src_train"
    )
    loss_src_valid, acc_src_valid, auc_src_valid = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"src_valid"
    )
    loss_src_test, acc_src_test, auc_src_test = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"src_test"
    )
    loss_tgt_valid, acc_tgt_valid, auc_tgt_valid = evaluate_dataset(
        target_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"tgt_valid"
    )
    loss_tgt_test, acc_tgt_test, auc_tgt_test = evaluate_dataset(
        target_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"tgt_test"
    )

    report = {
        "acc_tgt_valid": acc_tgt_valid,
        "auc_tgt_valid": auc_tgt_valid,
        "acc_tgt_test": acc_tgt_test,
        "auc_tgt_test": auc_tgt_test,
        "acc_src_valid": acc_src_valid,
        "auc_src_valid": auc_src_valid,
        "acc_src_train": acc_src_train,
        "auc_src_train": auc_src_train,
        "acc_src_test": acc_src_test,
        "auc_src_test": auc_src_test,
        "default": acc_tgt_valid,
    }
    loss_dict = {
        "loss_src_train": loss_src_train,
        "loss_src_valid": loss_src_valid,
        "loss_src_test": loss_src_test,
        "loss_tgt_valid": loss_tgt_valid,
        "loss_tgt_test": loss_tgt_test,
    }

    return report, loss_dict


def evaluate_dataset(data, MLP_encoder, GNNs, classifier, edge_weights, args, phase):
    """
    Evaluate on the specified dataset type.
    """
    data = data.to(device)
    MLP_encoder.eval()
    for layer in range(args.K):
        GNNs[layer].eval()
    classifier.eval()
    with torch.no_grad():
        # Determine the mask to use based on the phase.
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            mask = data.target_testing_mask

        label = data.y
        mlp_features = MLP_encoder(data)

        if "src" in phase:
            current_edge_weight = edge_weights
        else:
            current_edge_weight = [data.edge_weight.to(device).clone() for _ in range(args.K)]

        for layer in range(args.K):
            if layer == 0:
                gnn_features = GNNs[layer](mlp_features, data.edge_index, current_edge_weight[layer])
            else:
                gnn_features = GNNs[layer](gnn_features, data.edge_index, current_edge_weight[layer])
        pred = classifier(gnn_features)

        pred = pred[mask]
        label = label[mask]

        cls_loss = utils.CE_loss(pred, label).item()

        acc, auc = utils.get_scores(pred, label)

    return cls_loss, acc, auc



def get_avg_std_report(reports):
    all_keys = {k: [] for k in reports[0]}
    avg_report, avg_std_report = {}, {}
    for report in reports:
        for k in report:
            if report[k]:
                all_keys[k].append(report[k])
            else:
                all_keys[k].append(0)

    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.5f} +/- {np.std(v):.5f}' for k, v in all_keys.items()}
    return avg_report, avg_std_report

def objective(trial):
    # Define the search space
    optimized_params = {
        'alphamin_gnn': trial.suggest_categorical('alphamin_gnn', [0.1,0.5,1]),  # 替代 suggest_uniform
        'alphatimes_gnn': trial.suggest_categorical('alphatimes_gnn', [1.0,1.5,2]),
        'alphamin_mlp': trial.suggest_categorical('alphamin_mlp', [0.01,0.05,0.1,0.5]),  # 替代 suggest_uniform
        'alphatimes_mlp': trial.suggest_categorical('alphatimes_mlp', [0.1, 0.5, 1.0, 1.5]),
        'gcn_dim': trial.suggest_categorical('gcn_dim', [32,64,128,256]),
        'conv_dim': trial.suggest_categorical('conv_dim', [16, 32, 64,128]),
        'start_epoch': trial.suggest_categorical('start_epoch', [200]),
        'rw_freq': trial.suggest_categorical('rw_freq', [1,5,10,15]),
        'rw_lmda': trial.suggest_categorical('rw_lmda', [0.5,0.6,0.7,0.8,0.9,1]),
    }

    #print("Optimized Parameters:", optimized_params)

    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    if 'rw' in args.method:
        args.reweight = True
        args.rw_freq = optimized_params['rw_freq']
        args.start_epoch = optimized_params['start_epoch']


    args.conv_dim = optimized_params['conv_dim']
    args.gcn_dim = optimized_params['gcn_dim']
    args.rw_lmda = optimized_params['rw_lmda']
    args.alphamin_mlp = optimized_params['alphamin_mlp']
    args.alphatimes_mlp = optimized_params['alphatimes_mlp']
    args.alphamin_gnn = optimized_params['alphamin_gnn']
    args.alphatimes_gnn = optimized_params['alphatimes_gnn']





    print("Arguments:", args)

    num_seeds = args.seed
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    best_scores = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # prepare the dataset
        if args.dataset == "SBM":
            source_dataset = datasets.get_synthetic_source_data(args.num_nodes, args.sigma, args.ps, args.qs)
            target_dataset = datasets.get_synthetic_target_data(args.num_nodes, args.sigma, args.pt, args.qt,rotation_angle=30)
        elif args.dataset == "dblp_acm":
            source_dataset = datasets.prepare_dblp_acm("../../DualAlign/dataset", args.src_name)
            target_dataset = datasets.prepare_dblp_acm("../../DualAlign/dataset", args.tgt_name)
        elif args.dataset == "Airport":
            source_dataset = datasets.prepare_airport("/../../DualAlign/dataset/Airport",
                                                          args.src_name)
            target_dataset = datasets.prepare_airport("/../../DualAlign/dataset/Airport",
                                                          args.tgt_name)
        elif args.dataset == "Arxiv":
            source_dataset = datasets.prepare_Arxiv("../../DualAlign/dataset/", [1950, 2007])
            target_dataset = datasets.prepare_Arxiv("../../DualAlign/dataset/", [2014, 2016])
            print("arxiv")
            print(source_dataset.num_edges)
        else:
            print("Invalid dataset name!")
            source_dataset = None
            target_dataset = None
        report_dict = train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)
    print("ggg")

    return avg_report['acc_tgt_test']

def search():
    def search():
        # create an Optuna object
        study = optuna.create_study(direction='maximize')  # or 'minimize'

        # Output file directory 
        output_file = "../../DualAlign/main/search_log/search_results.txt"

        # Open file
        with open(output_file, "w") as f:
            f.write("Optuna Search Log\n")
            f.write("====================\n")

        # Define callback function used to record results
        def log_trial(study, trial):
            with open(output_file, "a") as f:  
                f.write(f"Trial {trial.number}:\n")
                f.write(f"  Value: {trial.value}\n")
                f.write("  Params:\n")
                for key, value in trial.params.items():
                    f.write(f"    {key}: {value}\n")
                f.write("\n")

        # begin optimize，and add callback function
        study.optimize(objective, n_trials=50, callbacks=[log_trial])

        # Record the best results
        with open(output_file, "a") as f:
            f.write("====================\n")
            f.write(f"Number of finished trials: {len(study.trials)}\n")
            f.write("Best trial:\n")

            trial = study.best_trial
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

        print(f"Results have been saved to {output_file}")

def main():
    optimized_params = {
        'epochs': 300,
        'lr': 0.007,
        'K': 2,
        'alphamin_gnn': 0.5,
        'alphatimes_gnn': 1.5,
        'alphamin_mlp': 0.1,
        'alphatimes_mlp': 0.1,
        'opt_decay_step': 50,
        'opt_decay_rate': 0.8,
        'gcn_dim': 64,
        'hidden_dim': 128,
        'conv_dim': 64,
        'cls_dim': 16,
        'start_epoch': 200,
        'mlp_embed_dim': 128,
        'rw_freq': 5,
        'rw_lmda': 0.9,
        'mlp_layers': 1
    }
    # print(optimized_params)
    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    if 'rw' in args.method:
        print('args.reweight=True')
        args.reweight = True
        args.rw_freq = optimized_params['rw_freq']
        args.start_epoch = optimized_params['start_epoch']

    args.lr = optimized_params['lr']
    args.K = optimized_params['K']
    args.opt_decay_rate = optimized_params['opt_decay_rate']
    args.opt_decay_step = optimized_params['opt_decay_step']
    args.hidden_dim = optimized_params['hidden_dim']
    args.gcn_dim = optimized_params['gcn_dim']
    args.conv_dim = optimized_params['conv_dim']
    args.rw_lmda = optimized_params['rw_lmda']
    args.alphamin_gnn = optimized_params['alphamin_gnn']
    args.alphatimes_gnn = optimized_params['alphatimes_gnn']
    args.alphamin_mlp = optimized_params['alphamin_mlp']
    args.alphatimes_mlp = optimized_params['alphatimes_mlp']
    args.epochs = optimized_params['epochs']
    args.mlp_layers = optimized_params['mlp_layers']
    # args.dc_layers = optimized_params['dc_layers']
    # args.class_layers = optimized_params['class_layers']
    # args.dropout = optimized_params['dropout']

    print(args)
    num_seeds = args.seed
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    best_scores = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # prepare the dataset
        if args.dataset == "SBM":
            source_dataset = datasets.get_synthetic_source_data(args.num_nodes, args.sigma, args.ps, args.qs)
            target_dataset = datasets.get_synthetic_target_data(args.num_nodes, args.sigma, args.pt, args.qt,
                                                                rotation_angle=30)
        elif args.dataset == "dblp_acm":
            source_dataset = datasets.prepare_dblp_acm("../../DualAlign/dataset", args.src_name)
            target_dataset = datasets.prepare_dblp_acm("../../DualAlign/dataset", args.tgt_name)
        elif args.dataset == "Airport":
            source_dataset = datasets.prepare_airport("/../../DualAlign/dataset/Airport",
                                                      args.src_name)
            target_dataset = datasets.prepare_airport("/../../DualAlign/dataset/Airport",
                                                      args.tgt_name)
        elif args.dataset == "Arxiv":
            source_dataset = datasets.prepare_Arxiv("../../DualAlign/dataset/", [1950, 2007])
            target_dataset = datasets.prepare_Arxiv("../../DualAlign/dataset/", [2014, 2016])
            print("arxiv")
            print(source_dataset.num_edges)
        else:
            print("Invalid dataset name!")
            source_dataset = None
            target_dataset = None
        report_dict = train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)


if __name__ == "__main__":
    main()
