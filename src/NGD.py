import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import dataclasses

class TrueNGD():
    @dataclass
    class Params:
        lr: float = 0.01
        fisher_batch_size: int = 1000
        physical_batch_size: int = 1000
        gradient_batch_size: int = None
        epsilon: float = 1e-8
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes: int = 10
        momentum: float = 0
        clip: float = None
    
        def get_dirname(self) -> str:
            directory = f"lr_{self.lr}__fisher_bs_{self.fisher_batch_size}__phys_bs_{self.physical_batch_size}__epsilon_{self.epsilon}"
            if self.gradient_batch_size is not None:
                directory += f"__gradient_bs_{self.gradient_batch_size}"
            return directory
        
    def __init__(self,
                    model: torch.nn.Module, 
                    loss_fn: torch.nn.Module,
                    dataset: torch.utils.data.Dataset,
                    activation_fn: torch.nn.Module = None,
                    params: Params = Params()):
        print(f"True NGD with params: {params}")
        """
            param.gradient_batch_size has to be divisible by params.physical_batch_size
        """
        self.__dict__.update(dataclasses.asdict(params))
        self.model = model
        self.FIM_momentum = [torch.zeros_like(param) for param in self.model.parameters()]
        self.loader_train = DataLoader(dataset, batch_size=self.physical_batch_size, shuffle=True)
        self.loader_FIM = DataLoader(dataset, batch_size=1, shuffle=True)
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn if activation_fn is not None else torch.nn.Softmax(dim=1)
            
    def step(self):
        l_grad = [torch.zeros_like(param) for param in self.model.parameters()]
        FIM_diag = [torch.zeros_like(param) for param in self.model.parameters()]
        for i, (X, y) in enumerate(self.loader_train):
            compute_grad = self.gradient_batch_size is None or i * self.physical_batch_size < self.gradient_batch_size
            if not compute_grad:
                break
            
            X = X.to(self.device)
            y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float().to(self.device)
            model_logits = self.model(X)
                
            loss = self.loss_fn(model_logits, y)
            l_grad_batch = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            for j, param in enumerate(self.model.parameters()):
                l_grad[j] += l_grad_batch[j] / self.gradient_batch_size
                
        for i, (X, y) in enumerate(self.loader_FIM):
            compute_fim = self.fisher_batch_size is None or i < self.fisher_batch_size
            if not compute_fim:
                break
            X = X.to(self.device)
            y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float().to(self.device)
            model_logits = self.model(X)
            
            if self.activation_fn is not None:
                model_probs = self.activation_fn(model_logits)
            else:
                model_probs = model_logits
                
            for cl in range(self.num_classes):
                likeh = model_probs[0][cl]
                cl_loglikeh = torch.log(likeh + self.epsilon)
                
                grads = torch.autograd.grad(cl_loglikeh, self.model.parameters(), retain_graph=True)
                
                for j, param in enumerate(self.model.parameters()):
                    FIM_diag[j] += likeh * (grads[j]**2) / self.fisher_batch_size
        
        
        for j, param in enumerate(self.model.parameters()):
            if self.momentum is not None:
                self.FIM_momentum[j] = self.momentum * self.FIM_momentum[j] + (1 - self.momentum) * FIM_diag[j]
            else:
                self.FIM_momentum[j] = FIM_diag[j]
                    
        # calculate the natural gradient:
        ng = [l_grad[j] / (self.FIM_momentum[j] + self.epsilon) for j in range(len(l_grad))]
        ng_cat = torch.cat([ng[j].flatten() for j in range(len(ng))])   
        ng_norm = torch.norm(ng_cat)
        
        step_taken = [torch.zeros_like(param) for param in self.model.parameters()]
        
        for j, param in enumerate(self.model.parameters()):
            if self.clip is not None:
                ng[j] = ng[j] / max(1, ng_norm.item()) * self.clip
            step_taken[j] = ng[j] * self.lr
            param.data -= step_taken[j]
        
        
        cos_similarity = torch.nn.functional.cosine_similarity(
            torch.cat([l_grad_el.flatten() for l_grad_el in l_grad]),
            torch.cat([ng_el.flatten() for ng_el in ng]),
            dim=0,
        )
        
        print("cos_similarity: ", cos_similarity.item())
        print("ng_norm: ", ng_norm.item())
        print("grad_norm: ", torch.norm(torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])).item())
        print("FIM_diag: ", torch.norm(torch.cat([FIM_diag_el.flatten() for FIM_diag_el in FIM_diag])).item())
        print("FIM_momentum: ", torch.norm(torch.cat([FIM_momentum_el.flatten() for FIM_momentum_el in self.FIM_momentum])).item())
        print("step taken: ", torch.norm(torch.cat([step_taken[j].flatten() for j in range(len(step_taken))])).item())
            
        return {
            "loss": loss.item(),
            "cosine_similarity": cos_similarity.item(),
            "ng_norm": ng_norm.item(),
            "grad_norm": torch.norm(torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])).item(),
            "FIM_diag": torch.norm(torch.cat([FIM_diag_el.flatten() for FIM_diag_el in FIM_diag])).item(),
            "FIM_momentum": torch.norm(torch.cat([FIM_momentum_el.flatten() for FIM_momentum_el in self.FIM_momentum])).item()
        }

        

class EmpiricalNGD():
    
    @dataclass
    class Params:
        lr: float = 0.01
        fisher_batch_size: int = 1000
        physical_batch_size: int = 1000
        gradient_batch_size: int = None
        epsilon: float = 1e-8
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gd_only: bool = False
        momentum: float = 0
        clip: float = None
        grad_amount: float = 0
        
        def get_dirname(self) -> str:
            directory = f"lr_{self.lr}__fisher_bs_{self.fisher_batch_size}__phys_bs_{self.physical_batch_size}__epsilon_{self.epsilon}"
            if self.momentum != 0:
                directory += f"__momentum_{self.momentum}"
            if self.clip is not None:
                directory += f"__clip_{self.clip}"
            if self.gd_only:
                directory += f"__gd_only_{self.gd_only}"
            if self.gradient_batch_size is not None:
                directory += f"__gradient_bs_{self.gradient_batch_size}"
            if self.grad_amount > 0:
                directory += f"__grad_amount_{self.grad_amount}"
            return directory
        
        
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_fn: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 activation_fn: torch.nn.Module = None,
                 params: Params = Params()):
        """
            param.gradient_batch_size has to be divisible by params.physical_batch_size
        """
        self.__dict__.update(dataclasses.asdict(params))
        self.model = model
        self.FIM_momentum = [torch.zeros_like(param) for param in self.model.parameters()]
        self.loader_train = DataLoader(dataset, batch_size=self.physical_batch_size, shuffle=True)
        self.loader_FIM = DataLoader(dataset, batch_size=1, shuffle=True)
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn if activation_fn is not None else torch.nn.Softmax(dim=1)
        
        if self.fisher_batch_size is None or self.fisher_batch_size > len(self.loader_FIM):
            self.fisher_batch_size = len(self.loader_FIM)
        if self.gradient_batch_size is None:
            self.gradient_batch_size = len(self.loader_train) * self.physical_batch_size
        
        
    def step(self):
        # self.model.zero_grad()
        l_grad = [torch.zeros_like(param) for param in self.model.parameters()]
        FIM_diag = [torch.zeros_like(param) for param in self.model.parameters()]

        for i, (X, y) in enumerate(self.loader_train):
            if i >= self.gradient_batch_size // self.physical_batch_size:
                break
                        
            X = X.to(self.device)
            y = y.to(self.device)
            if y.dim() <= 1:
                y = torch.nn.functional.one_hot(y, num_classes=10).float()
            model_logits = self.model(X)
                
            loss = self.loss_fn(model_logits, y)
            l_grad_batch = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            for j, param in enumerate(self.model.parameters()):
                l_grad[j] += l_grad_batch[j] / self.gradient_batch_size
                
    
        for i, (X, y) in enumerate(self.loader_FIM):
            if i >= self.fisher_batch_size:
                break
            X = X.to(self.device)
            y = y.to(self.device)
            if y.dim() <= 1:
                y = torch.nn.functional.one_hot(y, num_classes=10).float()
            model_logits = self.model(X)
            
            if self.activation_fn is not None:
                model_probs = self.activation_fn(model_logits)
            else:
                model_probs = model_logits
                
            llh = torch.sum(model_probs * y, dim=1)
            true_label_llh = torch.log(llh + self.epsilon)
            grads = torch.autograd.grad(true_label_llh, self.model.parameters(), retain_graph=False)
            
            for j, param in enumerate(self.model.parameters()):
                FIM_diag[j] += (grads[j]**2) / self.fisher_batch_size
                

        for j, param in enumerate(self.model.parameters()):
            self.FIM_momentum[j] = self.momentum * self.FIM_momentum[j] + (1 - self.momentum) * FIM_diag[j]
            
        # calculate the natural gradient:
        ng = [l_grad[j] / (self.FIM_momentum[j] + self.epsilon) for j in range(len(l_grad))]
        ng_cat = torch.cat([ng[j].flatten() for j in range(len(ng))])   
        ng_norm = torch.norm(ng_cat)
        
        step_taken = [torch.zeros_like(param) for param in self.model.parameters()]
        
        
        for j, param in enumerate(self.model.parameters()):
            if self.clip is not None:
                ng[j] = ng[j] / max(1, ng_norm.item()) * self.clip
            step_taken[j] = (l_grad[j] * self.grad_amount + ng[j] * (1-self.grad_amount)) * self.lr
            param.data -= step_taken[j]
            
        
        cos_similarity = torch.nn.functional.cosine_similarity(
            torch.cat([l_grad_el.flatten() for l_grad_el in l_grad]),
            torch.cat([ng_el.flatten() for ng_el in ng]),
            dim=0,
        )
        
        print("cos_similarity: ", cos_similarity.item())
        print("ng_norm: ", ng_norm.item())
        print("grad_norm: ", torch.norm(torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])).item())
        print("FIM_diag: ", torch.norm(torch.cat([FIM_diag_el.flatten() for FIM_diag_el in FIM_diag])).item())
        print("FIM_momentum: ", torch.norm(torch.cat([FIM_momentum_el.flatten() for FIM_momentum_el in self.FIM_momentum])).item())
        print("step taken: ", torch.norm(torch.cat([step_taken[j].flatten() for j in range(len(step_taken))])).item())
            
        return {
            "loss": loss.item(),
            "cosine_similarity": cos_similarity.item(),
            "ng_norm": ng_norm.item(),
            "grad_norm": torch.norm(torch.cat([l_grad[j].flatten() for j in range(len(l_grad))])).item(),
            "FIM_diag": torch.norm(torch.cat([FIM_diag_el.flatten() for FIM_diag_el in FIM_diag])).item(),
            "FIM_momentum": torch.norm(torch.cat([FIM_momentum_el.flatten() for FIM_momentum_el in self.FIM_momentum])).item()
        }
