from os import makedirs
import os

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector

import argparse
from archs import load_architecture
from utilities import get_loss_and_acc, compute_losses, \
    save_files, save_files_final
from data import load_dataset, take_first, DATASETS

import NGD


def main(dataset: str, arch_id: str, loss: str, opt: str,
         lr: float, beta1: float, beta2: float, epsilon: float,
         max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0):
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/ngd/lr_{lr}_beta1_{beta1}_beta2_{beta2}_eps_{epsilon}"
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    # abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    # torch.manual_seed(7)
    # projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))


    # train_loss, test_loss, train_acc, test_acc = \
    #     torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    # iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    # eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    
    loader = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=False)
    optimizer = NGD.EmpiricalNGD(lr=lr, fisher_batch_size=physical_batch_size)

    step = 0
    for X, y in loader:
        if step >= max_steps:
            break
        
        # train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
        #                                                    physical_batch_size)
        # test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        # if iterate_freq != -1 and step % iterate_freq == 0:
        #     iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        # if save_freq != -1 and step % save_freq == 0:
        #     save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
        #                            ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
        #                            ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        # print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        # if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
        #     break
        
        optimizer.step(network, loss_fn, X, y, train_dataset)
        
        
        
    # save_files_final(directory,
    #                  [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
    #                   ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
    #                   ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    # if save_model:
    #     torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta1", type=float, help="Adam beta1 parameter", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2 parameter", default=0.995)
    parser.add_argument("--epsilon", type=float, help="Adam epsilon parameter", default=1e-7)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta1=args.beta1,
         beta2=args.beta2, epsilon=args.epsilon, nproj=args.nproj, loss_goal=args.loss_goal,
         acc_goal=args.acc_goal, abridged_size=args.abridged_size, seed=args.seed)
