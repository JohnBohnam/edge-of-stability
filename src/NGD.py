import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import dataclasses

class EmpiricalNGD():
    
    @dataclass
    class Params:
        lr: float = 0.01, 
        fisher_batch_size: int = 1000,
        physical_batch_size: int = 1000,
        epsilon: float = 1e-8,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def get_filename(self) -> str:
            return f"lr_{self.lr}__fisher_bs_{self.fisher_batch_size}__phys_bs_{self.physical_batch_size}__epsilon_{self.epsilon}"
        
        
    def __init__(self, params: Params = Params(),):
        self.__dict__.update(dataclasses.asdict(params))
        
    def step(self, 
             model: torch.nn.Module, 
             loss_fn: torch.nn.Module, 
             X, 
             y, 
             dataset: torch.utils.data.Dataset,
             ):
        loader_fim = DataLoader(dataset, batch_size=1, shuffle=True)

        X = X.to(self.device)
        y = y.to(self.device)

        
        model_probs = torch.nn.Softmax(dim=1)(model(X))
        loss = loss_fn(model_probs, y)
        model.zero_grad()
        l_grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)

        # calculate diagonal fisher:
        FIM_diag = [torch.zeros_like(param) for param in model.parameters()]
    
        for i, (X_f, y_f) in enumerate(loader_fim):
            if i >= self.fisher_batch_size:
                break
            X_f = X_f.to(self.device)
            y_f = y_f.to(self.device)
            
            llh = torch.sum(torch.nn.Softmax(dim=1)(model(X_f)) * y_f, dim=1)                        
            true_label_llh = torch.log(llh + self.epsilon)
            
            llh_grad = torch.autograd.grad(true_label_llh, model.parameters(), create_graph=False)
            for j, param in enumerate(model.parameters()):
                FIM_diag[j] += llh_grad[j]**2

        # calculate the natural gradient:
        ng = [l_grad[j] / (FIM_diag[j] + self.epsilon) for j in range(len(l_grad))]
        ng_cat = torch.cat([ng[j].flatten() for j in range(len(ng))])   
        ng_norm = torch.norm(ng_cat) 
        
        for j, param in enumerate(model.parameters()):
            param.data -= self.lr * ng[j] / max(1.0, ng_norm.item())
