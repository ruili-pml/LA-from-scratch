import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad, jacrev 
from torch.distributions.multivariate_normal import _precision_to_scale_tril
import numpy as np

def get_per_sample_jacobian(model, inputs):
    """
    inputs: [B, D]
    """
    
    params = dict(model.named_parameters())  # ordered param dict

    def fmodel(params, x):
        return functional_call(model, params, (x,))  # returns logits of shape [C]

    
    jac_fn = jacrev(fmodel, argnums=0)  
    per_sample_jac = vmap(jac_fn, in_dims=(None, 0))(params, inputs)
    
    return per_sample_jac  # dict {'fc1.weight': [B, C, Dout, Din], ...}

def get_ce_loss_wrt_output_Hessian(logits):
    
    """
    logits : (B, C)  tensor
    returns: (B, C, C) Hessian blocks
    """
    p = logits.softmax(-1)  # [B, C]                   
    loss_wrt_output_Hessian = torch.diag_embed(p) - torch.einsum("bi, bj -> bij", p, p)
    return loss_wrt_output_Hessian


def get_GGN_Hessian(per_sample_jacobian, loss_wrt_output_Hessian):
    
    """
    per_sample_jacobian: {'fc1.weight': [B, C, Dout, Din], ...}
    loss_wrt_output_Hessian: [B, C, C]
    """
    
    full_GGN = []
    for layer_name_1 in ['fc1', 'fc2', 'fc3']:
        cur_layer_GGN = []
        for layer_name_2 in ['fc1', 'fc2', 'fc3']:
            cur_jacobian_1 = per_sample_jacobian[f"{layer_name_1}.weight"].flatten(start_dim = 2) # [B, C, Dout_1, Din_1] -> [B, C, Dout_1 * Din_1]
            cur_jacobian_2 = per_sample_jacobian[f"{layer_name_2}.weight"].flatten(start_dim = 2) # [B, C, Dout_2, Din_2] -> [B, C, Dout_2 * Din_2]
            cur_GGN_H = cur_jacobian_1.transpose(1, 2) @ loss_wrt_output_Hessian @ cur_jacobian_2 
            cur_layer_GGN.append(cur_GGN_H.sum(0).detach())
        full_GGN.append(torch.cat(cur_layer_GGN, dim = 1))
    
    return torch.cat(full_GGN, dim = 0)

def get_per_sample_gradient(model, loss_function, inputs, targets):
    """
    inputs: [B, D]
    """
    
    params = dict(model.named_parameters())  # ordered param dict

    def fmodel(params, x):
        return functional_call(model, params, (x,))  # returns logits of shape [C]

    def compute_loss(params, x, y):
        pred = fmodel(params, x)
        loss = loss_function(pred, y.squeeze(0))
        return loss
    
    grad_of_loss = grad(compute_loss, argnums=0)
    
    per_sample_gradient = vmap(grad_of_loss, in_dims=(None, 0, 0))(params, inputs, targets)
    
    return per_sample_gradient  # dict {'fc1.weight': [B, Dout, Din], }

def get_EF_Hessian(per_sample_gradient):
    """
    per_sample_gradient: {'fc1.weight': [B, D_out, D_in], ...}
    """
    
    full_EF = []
    for layer_name_1 in ['fc1', 'fc2', 'fc3']:
        cur_layer_EF = []
        for layer_name_2 in ['fc1', 'fc2', 'fc3']:
            cur_gradient_1 = per_sample_gradient[f"{layer_name_1}.weight"].flatten(start_dim = 1).unsqueeze(2) # [B, Dout_1, Din_1] -> [B, Dout_1 * Din_1] ->  [B, Dout_1 * Din_1, 1]
            cur_gradient_2 = per_sample_gradient[f"{layer_name_2}.weight"].flatten(start_dim = 1).unsqueeze(1) # [B, Dout_2, Din_2] -> [B, Dout_2 * Din_2] -> [B, 1, Dout_2 * Din_2]
            cur_EF_H = cur_gradient_1 @ cur_gradient_2
            cur_layer_EF.append(cur_EF_H.sum(0).detach())
        full_EF.append(torch.cat(cur_layer_EF, dim = 1))
    
    return torch.cat(full_EF, dim = 0)


class FullLA:
    
    def __init__(self, model, prior_prec, hessian_approx):
        
        self.model = model
        self.prior_prec = torch.nn.Parameter(torch.tensor([prior_prec], requires_grad=True)) 
        self.hessian_approx = hessian_approx
    
    def estimate_hessian(self, dataloader):
        
        """
        estimate on the whole data set, no mini-batching
        """
        X = dataloader.dataset.X
        Y = dataloader.dataset.Y
        
        if self.hessian_approx == 'GGN':
            per_sample_jacobian = get_per_sample_jacobian(self.model, X)
            logits = self.model(X)
            ce_hessian = get_ce_loss_wrt_output_Hessian(logits)
            
            GGN_H = get_GGN_Hessian(per_sample_jacobian, ce_hessian)
            self.H = GGN_H
        
        if self.hessian_approx == 'EF':
            loss_function = nn.CrossEntropyLoss(reduction='sum')
            per_sample_gradient = get_per_sample_gradient(self.model, loss_function, X, Y)
            
            EF_H = get_EF_Hessian(per_sample_gradient)
            self.H = EF_H

        
    def compute_log_marginal_likelihood(self, dataloader, prior_prec):
        
        X = dataloader.dataset.X
        Y = dataloader.dataset.Y
        
        # log likelihood
        logits = self.model(X)
        ce_loss = nn.CrossEntropyLoss(reduction='sum')
        log_likelihood = - ce_loss(logits, Y)
        
        # theta_map term
        theta_map = torch.cat([p.view(-1).detach() for p in self.model.parameters()])
        num_param = theta_map.shape[0]
        theta_map_term = torch.sum(theta_map **2) * prior_prec
        
        # log_det_prior_precision
        log_det_prior_precision = (torch.eye(num_param) * prior_prec).logdet()
        
        # log_det_pposterior_precision
        log_det_posterior_precision = (self.H + torch.eye(num_param) * prior_prec).logdet()
        
        log_marg_lik = log_likelihood - 0.5 * (theta_map_term + log_det_posterior_precision - log_det_prior_precision)
        
        return log_marg_lik


    def optimize_prior_precision(self, lr, num_epoch, dataloader, verbose = True):
        
        optimizer = torch.optim.Adam([self.prior_prec], lr, maximize=True)
        
        for epoch in range(num_epoch):
            
            optimizer.zero_grad()
            log_mag_lik = self.compute_log_marginal_likelihood(dataloader, self.prior_prec)
            log_mag_lik.backward()
            optimizer.step()
        
            if verbose:
                print(f"Epoch {epoch}: log marg lik {log_mag_lik.item(): .3f}")
    
    def get_posterior_precision(self):
                
        num_param = self.H.shape[0]
        posterior_precision = self.H + torch.diag_embed(self.prior_prec * torch.ones(num_param))
        
        return posterior_precision
    
    
    def predict(self, x, num_samples = 50):
        
        posterior_precision = self.get_posterior_precision()
        scale_tril = _precision_to_scale_tril(posterior_precision)
        
        num_param = posterior_precision.shape[0]
        # theta ~ N(theta_map, P^{-1})
        z = torch.randn(num_samples, num_param)
        theta_map = torch.cat([p.view(-1).detach() for p in self.model.parameters()]) # [num_param,]
        sampled_theta = theta_map.unsqueeze(0) + z @ scale_tril.T # [num_samples, num_params]
        
        
        # 1. Define the functional version of your model
        # This function takes a dictionary of parameters and a single input tensor 'x'
        def fmodel(params, x_input):
            return functional_call(self.model, params, (x_input,))
        
        
        # 2. Convert the [num_samples, num_params] matrix into a dictionary of batched parameters
        # Each value in the dictionary will have shape [num_samples, ...original_shape]
        
        network_info = [(name, tuple(param.shape), param.numel())
            for name, param in self.model.named_parameters()]
        
        dict_param = {}
        start = 0
        for name, shape, numel in network_info:
            dict_param[name] = sampled_theta[:, start : start + numel].reshape(num_samples, *shape)
            start += numel
        
        # 3. Use vmap to get predictions for all parameter sets
        # in_dims=(0, None) tells vmap to:
        # - Map over the first dimension (dim 0) of the first argument (batched_params).
        # - Do NOT map over the second argument (x). Instead, broadcast it for each call.
        
        predictions = vmap(fmodel, in_dims=(0, None))(dict_param, x)
        
        f_mean = predictions.mean(1)
        f_var = predictions.var(1)
        
        kappa = 1 / torch.sqrt(1.0 + np.pi / 8 * f_var)
        return torch.softmax(kappa * f_mean, dim=-1)

        

        
        
        

            
            
        
        