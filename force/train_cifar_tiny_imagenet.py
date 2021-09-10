'''
FORCE
Copyright (c) 2020-present NAVER Corp.
MIT license
'''

from metrics.flops import flops
from numpy.lib.shape_base import split
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from pruning.pruning_algos import iterative_pruning
from experiments.experiments import *
from pruning.mask_networks import apply_prune_mask

from pruning.mtx_io import *
from utils.ptflops import get_model_complexity_info
import os
import argparse
import random
import math



def parseArgs():

    parser = argparse.ArgumentParser(
        description="Training CIFAR / Tiny-Imagenet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pruning_factor", type=float, default=0.01, dest="pruning_factor",
                        help='Fraction of connections after pruning')

    parser.add_argument("--prune_method", type=int, default=3, dest="prune_method",
                        help="""Which pruning method to use:
                                1->Iter SNIP
                                2->GRASP-It
                                3->FORCE (default). """)

    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        dest="dataset_name", help='Dataset to train on')

    parser.add_argument("--network_name", type=str, default='resnet50', dest="network_name",
                        help='Model to train')

    parser.add_argument("--num_steps", type=int, default=10,
                        help='Number of steps to use with iterative pruning')

    parser.add_argument("--mode", type=str, default='exp',
                        help='Mode of creating the iterative pruning steps one of "linear" or "exp".')

    parser.add_argument("--num_batches", type=int, default=1,
                        help='''Number of batches to be used when computing the gradient.
                        If set to -1 they will be averaged over the whole dataset.''')

    parser.add_argument("--save_interval", type=int, default=50,
                        dest="save_interval", help="Number of epochs between model checkpoints.")

    parser.add_argument("--save_loc", type=str, default='saved_models/',
                        dest="save_loc", help='Path where to save the model')

    parser.add_argument("--opt", type=str, default='sgd',
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    parser.add_argument("--saved_model_name", type=str, default="cnn.model",
                        dest="saved_model_name", help="Filename of the pre-trained model")

    parser.add_argument("--frac-train-data", type=float, default=0.9, dest="frac_data_for_train",
                        help='Fraction of data used for training (only applied in CIFAR)')

    parser.add_argument("--init", type=str, default='normal_kaiming',
                        help='Which initialization method to use')

    parser.add_argument("--in_planes", type=int, default=64,
                        help='''Number of input planes in Resnet. Afterwards they duplicate after
                        each conv with stride 2 as usual.''')
    parser.add_argument("--mtx_file", type=str, default=None,
                        help='''Whether to load from mtx file.''')

    parser.add_argument("--target_acc", type=float,
                        default=None, help="The target accuracy")
    parser.add_argument("--seed", type=int,
                        default=None, help="Set seed for training")
    parser.add_argument("--epoch", type=int,
                        default=None, help="Number of epochs for training")
    parser.add_argument("--sparse", help="wether to use sparse", action="store_true")

    parser.add_argument("--profile", help="wether to profile the training", action="store_true")

    return parser.parse_args()


LOG_INTERVAL = 20
# Number of initialize-prune-train trials (minimum of 1)
REPEAT_WITH_DIFFERENT_SEED = 1
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New additions
args = parseArgs()


def check_global_pruning(mask):
    "Compute fraction of unpruned weights in a mask"
    flattened_mask = torch.cat([torch.flatten(x) for x in mask])
    return flattened_mask.mean()


def check_global_sparsity(net):
    "Compute fraction of unpruned weights in a mask"
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())
    all_masks = []
    for layer in prunable_layers:
        keep_mask = (layer.weight.data != 0.).float()
        all_masks.append(keep_mask)

    return check_global_pruning(all_masks)


def train(seed):

    # Set manual seed
    torch.manual_seed(seed)

    if 'resnet' in args.network_name:
        stable_resnet = False
        if 'stable' in args.network_name:
            stable_resnet = True
        if 'CIFAR' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_cifar_experiment(device, args.network_name,
                                                                  args.dataset_name, args.optimiser,
                                                                  args.frac_data_for_train,
                                                                  stable_resnet, args.in_planes)
        elif 'tiny_imagenet' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_tiny_imagenet_experiment(device, args.network_name,
                                                                          args.dataset_name, args.in_planes)

    elif 'vgg' in args.network_name or 'VGG' in args.network_name:
        if 'tiny_imagenet' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_tiny_imagenet_experiment(device, args.network_name,
                                                                       args.dataset_name)
        else:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_cifar_experiment(device, args.network_name,
                                                               args.dataset_name, args.frac_data_for_train)

    # Initialize network
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if args.init == 'normal_kaiming':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif args.init == 'normal_kaiming_fout':
                nn.init.kaiming_normal_(
                    layer.weight, nonlinearity='relu', mode='fan_out')
            elif args.init == 'normal_xavier':
                nn.init.xavier_normal_(layer.weight)
            elif args.init == 'orthogonal':
                nn.init.orthogonal_(layer.weight)
            else:
                raise ValueError(
                    f"Unrecognised initialisation parameter {args.init}")

    ############################################################################
    ####################        Pruning at init         ########################
    ############################################################################
    # Before pruning

    pruning_factor = args.pruning_factor
    keep_masks = []
    if(args.mtx_file != None):
        mtx_f = "{}_{}_{}".format(args.mtx_file, seed,pruning_factor)
    if(not os.path.exists(mtx_f)):
        if pruning_factor != 1:
            print(f'Pruning network iteratively for {args.num_steps} steps')
            torch.backends.cudnn.benchmark = False
            keep_masks = iterative_pruning(net, train_loader, device, pruning_factor,
                                            prune_method=args.prune_method,
                                            num_steps=args.num_steps,
                                            mode=args.mode, num_batches=args.num_batches)

            apply_prune_mask(net, keep_masks)
            if(args.mtx_file != None):
                mtx_dir = "{}_{}_{}".format(args.mtx_file, seed,pruning_factor)
                os.mkdir(mtx_dir)
                print("saving to file: " + mtx_dir)
                save_to_mtx(net, mtx_dir)

    elif (not args.sparse):
        print("load from " + mtx_f)
        _, keep_masks = load_from_mtx(net, mtx_f)
    else:
        # Sparse resnet
        net = sparse_resnet_cifar(args.network_name,args.dataset_name,stable_resnet, args.in_planes)
        print("load to sparse net")
        _, keep_masks = sparse_load_from_mtx(net, mtx_f,device)
        net.to(device)
        [optimiser, lr_scheduler, loss, EPOCHS] = sparse_resnet_cifar_experiment(net, args.optimiser)

    print("sparsity_global_mask:", check_global_pruning(keep_masks))
    # Load to sparse net
    # cpu_net = net.cpu()
    # if(not os.path.isdir("./mtx")):
    #     os.mkdir("./mtx")

    # Profile
    if(args.profile):
        # Get the first input
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device).cpu()
            labels = labels.to(device).cpu()
            macs, params = get_model_complexity_info(net.cpu(), data, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # reorder_dir("./mtx",input=False)
            # os.rename("./mtx","./{}_{}".format(args.network_name,round(1 - pruning_factor,2)))
            break
    net.to(device)
    
    ############################################################################
    ####################          Training              ########################
    ############################################################################
    evaluator = create_supervised_evaluator(net, {
        'accuracy': Accuracy(),
        'cross_entropy': Loss(loss)
    }, device)

    run_name = (args.network_name + '_' + args.dataset_name + '_spars' +
                str(1 - pruning_factor) + '_variant' + str(args.prune_method) +
                '_train-frac' + str(args.frac_data_for_train) +
                f'_steps{args.num_steps}_{args.mode}' + f'_{args.init}' +
                f'_batch{args.num_batches}' + f'_rseed_{seed}')

    writer_name = 'runs/' + run_name
    writer = SummaryWriter(writer_name)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # torch.backends.cudnn.benchmark = False
    
    if(args.target_acc != None):
        start.record()
        record_f = open(run_name+ 'target_acc.txt','w')
    if(args.epoch != None):
        EPOCHS = args.epoch
        print("number of epochs:", EPOCHS)
    iterations = 0
    for epoch in range(0, EPOCHS):
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
        train_loss = train_cross_entropy(epoch, net, train_loader, optimiser,lr_scheduler, device,
                                         writer, LOG_INTERVAL=20)
        epoch_end.record()
        torch.cuda.synchronize()
        epoch_time = epoch_start.elapsed_time(epoch_end)/1000
        # iterations += len(train_loader)
        iterations += 1
        # Evaluate
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        # Save history
        avg_accuracy = metrics['accuracy']
        avg_cross_entropy = metrics['cross_entropy']
        writer.add_scalar("train/loss", train_loss, iterations)
        writer.add_scalar("test/loss", avg_cross_entropy, iterations)
        writer.add_scalar("test/accuracy", avg_accuracy, iterations)
        writer.add_scalar("epoch_time", epoch_time, iterations)
        
        if(args.target_acc != None):
            print(avg_accuracy)
            print(args.target_acc)
            print(math.isclose(avg_accuracy,args.target_acc))
            if(math.isclose(avg_accuracy,args.target_acc) or avg_accuracy > args.target_acc):
                end.record()
                torch.cuda.synchronize()
                time = start.elapsed_time(end)/1000*60
                print("matching time",time)
                print("matching epoch", epoch)
                record_f.write("matching time: {} \n matching epoch: {}".format(time,epoch))
                record_f.close()
                return

        # Save model checkpoints
        if (epoch + 1) % args.save_interval == 0:
            print("sparsity:", check_global_sparsity(net))
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            save_name = args.save_loc + run_name + \
                '_cross_entropy_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)
        elif (epoch + 1) == EPOCHS:
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            save_name = args.save_loc + run_name + \
                '_cross_entropy_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)


if __name__ == '__main__':
    # Randomly pick a random seed for the experiment
    # Multiply the number of seeds to be sampled by 300 so there is wide range of seeds
    if(args.seed == None):
        seed = np.random.randint(300)
    else:
        seed = args.seed
    print("set seed", seed)
    train(seed)
