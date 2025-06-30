from os import makedirs
import os

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector

import argparse
from archs import load_architecture
from utilities import get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, make_base_directory
from data import load_dataset, DATASETS, take_first
import dataclasses
import NGD

def main(dataset: str, 
         arch_id: str, 
         loss: str,
         max_steps: int, 
         model_params: NGD.EmpiricalNGD.Params,
         neigs: int = 0,
         eig_freq: int = -1, 
         iterate_freq: int = -1, 
         save_freq: int = -1,
         save_model: bool = False,
         nproj: int = 0,
         loss_goal: float = None, 
         acc_goal: float = None, 
         seed: int = 0,
         abridged_size: int = 5000,
        ):    
    directory = f"{make_base_directory(dataset, arch_id, loss)}/seed_{seed}/true_ngd/{model_params.get_dirname()}"
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))


    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    
    cos_similarity = torch.zeros(max_steps)
    grad_norm = torch.zeros(max_steps)
    ng_norm = torch.zeros(max_steps)
    FIM_diag_norm = torch.zeros(max_steps)
    FIM_momentum_norm = torch.zeros(max_steps)
    
    
    # if model_params.gd_only:
    #     activation_fn = None
    # else:
    activation_fn = torch.nn.Softmax(dim=1)
        
    print("Using activation function: ", activation_fn)
    optimizer = NGD.TrueNGD(network, loss_fn, train_dataset, params=model_params, activation_fn=activation_fn)


    for step in range(max_steps):
        metrics = optimizer.step()
        
        cos_similarity[step] = metrics["cosine_similarity"]
        grad_norm[step] = metrics["grad_norm"]
        ng_norm[step] = metrics["ng_norm"]
        FIM_diag_norm[step] = metrics["FIM_diag"]
        FIM_momentum_norm[step] = metrics["FIM_momentum"]
        
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset, model_params.physical_batch_size, activation_fn)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, model_params.physical_batch_size, activation_fn)

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), 
                                    ("iterates", iterates[:step // iterate_freq]),
                                    ("train_loss", train_loss[:step]), 
                                    ("test_loss", test_loss[:step]),
                                    ("train_acc", train_acc[:step]), 
                                    ("test_acc", test_acc[:step]),
                                    ("cos_similarity", cos_similarity[:step]),
                                    ("grad_norm", grad_norm[:step]),
                                    ("ng_norm", ng_norm[:step]),
                                    ("FIM_diag_norm", FIM_diag_norm[:step]),
                                    ("FIM_momentum_norm", FIM_momentum_norm[:step]),
                                    ])

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=model_params.physical_batch_size)
            print("eigenvalues: ", eigs[step//eig_freq, :])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break

        
        
    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--epsilon", type=float, help="Adam epsilon parameter", default=1e-7)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--fisher_batch_size", type=int,
                        help="the number of examples used to compute the Fisher information matrix", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    # parser.add_argument("--momentum", type=float, default=None,
    #                     help="the momentum parameter for the natural gradient descent optimizer")
    # parser.add_argument("--gd_only", action="store_true",
    #                     help="if true, only use the gradient descent part of the optimizer")
    # parser.add_argument("--clip", type=float, default=None,
    #                     help="the maximum norm for the gradient clipping")
    parser.add_argument("--gradient_batch_size", type=int, default=None,
                        help="the number of examples used to compute the gradient")
    
    args = parser.parse_args()
    
    model_fields = {f.name for f in dataclasses.fields(NGD.TrueNGD.Params)}
    model_args = {k: v for k, v in vars(args).items() if k in model_fields}
    model_params = NGD.TrueNGD.Params(**model_args)
    
    main_args = {k: v for k, v in vars(args).items() if k not in model_fields}

    main(model_params=model_params, **main_args)
