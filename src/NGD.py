import torch
from torch.optim.optimizer import Optimizer
from utilities import iterate_dataset
from torch.utils.data import DataLoader

class EmpiricalNGD():
    def __init__(self, lr: float = 0.01, fisher_batch_size: int = 1000):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fisher_batch_size = fisher_batch_size

        
    def step(self, model: torch.nn.Module, loss_fn: torch.nn.Module, X, y, dataset):
        loader_fim = DataLoader(dataset, batch_size=1, shuffle=False)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        
        model_probs = torch.nn.Softmax(dim=1)(model(X))
        loss = loss_fn(model_probs, y)
        print(f"loss: {loss}")
        model.zero_grad()
        l_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # calculate diagonal fisher:
        FIM_diag = [torch.zeros_like(param) for param in model.parameters()]
        
        llhs = []
        for i, (X_f, y_f) in enumerate(loader_fim):
            if i >= self.fisher_batch_size:
                break
            X_f = X_f.to(self.device)
            y_f = y_f.to(self.device)
            
            llh = torch.sum(torch.nn.Softmax(dim=1)(model(X_f)) * y_f, dim=1)
            llhs.append(llh)
                        
            true_label_llh = torch.log(llh + 1e-8)
            # print(f"llh: {llh}")
            
            llh_grad = torch.autograd.grad(true_label_llh, model.parameters(), create_graph=False)
            for j, param in enumerate(model.parameters()):
                FIM_diag[j] += llh_grad[j]**2
            
        llhs_ones = 0
        llhs_zeros = 0
        for llh in llhs:
            if llh > 1 - 1e-4:
                llhs_ones += 1
            elif llh < 1e-4:
                llhs_zeros += 1
                
                
        FIM_cat = torch.cat([FIM_diag[j].flatten() for j in range(len(FIM_diag))])
        ng = [l_grad[j] / (FIM_diag[j] + 1e-4) for j in range(len(l_grad))]
        
        fim_mean = torch.mean(FIM_cat)  
        
        grad_cat = torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])
        g_norm = torch.norm(grad_cat)
        
        ng_cat = torch.cat([ng[j].flatten() for j in range(len(ng))])   
        ng_norm = torch.norm(ng_cat) 
            
        print(f"llhs mean: \t{torch.mean(torch.stack(llhs)):.3f}")
        print(f"llhs ones: {llhs_ones}, zeros: {llhs_zeros}")
        
        print(f"FIM mean: \t{fim_mean:.3f}")
        print(f"FIM std: \t{torch.std(FIM_cat):.3f}")
        print(f"grad norm: \t{g_norm:.3f}")
        print(f"ng norm: \t{ng_norm:.3f}")
    
        print(f"disagre: \t{torch.norm(ng_cat / ng_norm - grad_cat / g_norm).item():.3f}")
        
        for j, param in enumerate(model.parameters()):
            param.data -= self.lr * ng[j] / max(1, ng_norm)