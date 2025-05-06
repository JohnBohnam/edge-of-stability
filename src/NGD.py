import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import dataclasses

class EmpiricalNGD():
    
    @dataclass
    class Params:
        lr: float = 0.01
        fisher_batch_size: int = 1000
        physical_batch_size: int = 1000
        epsilon: float = 1e-8
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        momentum: float = None
        clip: float = None
        
        def get_dirname(self) -> str:
            directory = f"lr_{self.lr}__fisher_bs_{self.fisher_batch_size}__phys_bs_{self.physical_batch_size}__epsilon_{self.epsilon}"
            if self.momentum is not None:
                directory += f"__momentum_{self.momentum}"
            if self.clip is not None:
                directory += f"__clip_{self.clip}"
            return directory
        
        
    def __init__(self, model: torch.nn.Module, params: Params = Params()):
        self.__dict__.update(dataclasses.asdict(params))
        self.model = model
        self.FIM_momentum = [torch.zeros_like(param) for param in self.model.parameters()]
        
        
    def step(self, 
             loss_fn: torch.nn.Module, 
             X, 
             y, 
             dataset: torch.utils.data.Dataset,
             ):
        loader_fim = DataLoader(dataset, batch_size=1, shuffle=True)

        X = X.to(self.device)
        y = y.to(self.device)

        
        model_probs = torch.nn.Softmax(dim=1)(self.model(X))
        loss = loss_fn(model_probs, y)
        self.model.zero_grad()
        l_grad = torch.autograd.grad(loss, self.model.parameters())
        # print(f"loss: {loss.item()}")

        # calculate diagonal fisher:
        FIM_diag = [torch.zeros_like(param) for param in self.model.parameters()]
    
        for i, (X_f, y_f) in enumerate(loader_fim):
            if i >= self.fisher_batch_size:
                break
            X_f = X_f.to(self.device)
            y_f = y_f.to(self.device)
            
            llh = torch.sum(torch.nn.Softmax(dim=1)(self.model(X_f)) * y_f, dim=1)            
            true_label_llh = torch.log(llh + self.epsilon)
            
            llh_grad = torch.autograd.grad(true_label_llh, self.model.parameters())
            for j, param in enumerate(self.model.parameters()):
                FIM_diag[j] += llh_grad[j]**2
                
        # update the FIM momentum:
        for j, param in enumerate(self.model.parameters()):
            if self.momentum is not None:
                self.FIM_momentum[j] = self.momentum * self.FIM_momentum[j] + (1 - self.momentum) * FIM_diag[j]
            else:
                self.FIM_momentum[j] = FIM_diag[j]
                

        # calculate the natural gradient:
        ng = [l_grad[j] / (self.FIM_momentum[j] + self.epsilon) for j in range(len(l_grad))]
        ng_cat = torch.cat([ng[j].flatten() for j in range(len(ng))])   
        ng_norm = torch.norm(ng_cat)
        
        for j, param in enumerate(self.model.parameters()):
            if self.clip is not None:
                ng[j] = ng[j] / max(self.clip, ng_norm.item())
            param.data -= self.lr * ng[j]
        
        cos_similarity = torch.nn.functional.cosine_similarity(
            torch.cat([l_grad_el.flatten() for l_grad_el in l_grad]),
            torch.cat([ng_el.flatten() for ng_el in ng]),
            dim=0,
        )
            
        return {
            "loss": loss.item(),
            "cosine_similarity": cos_similarity.item(),
            "ng_norm": ng_norm.item(),
            "grad_norm": torch.norm(torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])).item(),
            # "FIM_diag": torch.norm(torch.cat([FIM_diag_el.flatten() for FIM_diag_el in FIM_diag])).item(),
            # "FIM_momentum": torch.norm(torch.cat([FIM_momentum_el.flatten() for FIM_momentum_el in self.FIM_momentum])).item()
        }
