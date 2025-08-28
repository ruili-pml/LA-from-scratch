import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad, jacrev 
from torch.distributions.multivariate_normal import _precision_to_scale_tril
import numpy as np

def get_per_sample_jacobian(model, inputs):
    """
    Computes the Jacobian of the model's output with respect to its parameters
    for each individual sample in a batch.

    This function leverages PyTorch's functional programming tools (`torch.func`)
    for efficient batching of Jacobian calculations.

    Args:
        model (nn.Module): The neural network model.
        inputs (torch.Tensor): A batch of inputs with shape [B, D], where B is the batch size and D is the input dimension.

    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are parameter names (e.g., 'fc1.weight')
                                 and values are the corresponding Jacobians for each sample.
                                 The shape of a Jacobian tensor for a weight matrix will be
                                 [B, C, D_out, D_in], where C is the number of output classes.
    """

    # Create a dictionary of the model's named parameters.
    params = dict(model.named_parameters())  # ordered param dict

    # Define a stateless version of the model.
    def fmodel(params, x):
        return functional_call(model, params, (x,))  # returns logits of shape [C]

    # Create a function that computes the Jacobian of `fmodel`'s output with respect to its first argument (the parameters).
    jac_fn = jacrev(fmodel, argnums=0)  
    
    # Vectorize the Jacobian computation over the batch of inputs.
    # `vmap` maps `jac_fn` over the `inputs` tensor's batch dimension (the `0` in `in_dims`).
    # The `params` argument is not vectorized (the `None` in `in_dims`), so the same parameters are used for each sample.
    per_sample_jac = vmap(jac_fn, in_dims=(None, 0))(params, inputs)
    
    return per_sample_jac  # dict {'fc1.weight': [B, C, Dout, Din], ...}

def get_ce_loss_wrt_output_Hessian(logits):
    """
    Computes the Hessian of the cross-entropy loss with respect to the model's
    output logits for each sample in a batch.

    For a single sample, the Hessian of the categorical cross-entropy loss is given by:
    H = diag(p) - p * p.T
    where p = softmax(logits).

    Args:
        logits (torch.Tensor): The model's output logits of shape [B, C],
                               where B is batch size and C is the number of classes.

    Returns:
        torch.Tensor: A tensor of shape [B, C, C] containing the Hessian matrix
                      for each sample in the batch.
    """
    
    # Apply softmax to convert logits to probabilities.
    p = logits.softmax(-1)  # [B, C]        
    
    # Compute the Hessian for each sample in the batch.
    # `torch.diag_embed(p)` creates the diagonal matrix part, diag(p).
    # `torch.einsum("bi, bj -> bij", p, p)` computes the outer product p * p.T for each sample.           
    loss_wrt_output_Hessian = torch.diag_embed(p) - torch.einsum("bi, bj -> bij", p, p)
    
    return loss_wrt_output_Hessian # [B, C, C]


def get_GGN_Hessian(per_sample_jacobian, loss_wrt_output_Hessian, layer_name_list):
    """
    Constructs the full Generalized Gauss-Newton (GGN) Hessian approximation
    by combining per-sample Jacobians and the loss Hessian w.r.t. logits.

    The GGN is approximated as H_ggn = sum_n(J_n.T @ H_loss_n @ J_n), where J_n is
    the Jacobian and H_loss_n is the loss Hessian for the n-th sample. 

    Args:
        per_sample_jacobian (dict[str, torch.Tensor]): A dictionary of per-sample Jacobians, {'fc1.weight': [B, C, Dout, Din], ...}
                
        loss_wrt_output_Hessian (torch.Tensor): The Hessian of the loss w.r.t. logits of shape [B, C, C] 
        
        layer_name_list (list[str]): List with each layers name

    Returns:
        torch.Tensor: The full GGN Hessian matrix, assembled from all blocks,
                      with shape [P, P], where P is the total number of model parameters.
    """
    
    # The nested loops below construct the full GGN Hessian block by block.
    # The (i, j) block is (∂f/∂θ^(i))^T @ H_loss @ ∂f/∂θ^(j)
    # where ∂f/∂θ^(i) is the Jacobians of the model output f w.r.t. parameters of the i-th layer.
    
    full_GGN_rows = []
    for layer_name_1 in layer_name_list:
        current_row_blocks = []
        for layer_name_2 in layer_name_list:
            # Compute current GGN block
            jac1 = per_sample_jacobian[f"{layer_name_1}.weight"].flatten(start_dim = 2) # [B, C, Dout_1, Din_1] -> [B, C, Dout_1 * Din_1]
            jac2 = per_sample_jacobian[f"{layer_name_2}.weight"].flatten(start_dim = 2) # [B, C, Dout_2, Din_2] -> [B, C, Dout_2 * Din_2]
            cur_GGN_block = jac1.transpose(1, 2) @ loss_wrt_output_Hessian @ jac2
            
            # Sum over the batch dimension to get the final GGN block.
            current_row_blocks.append(cur_GGN_block.sum(0).detach())
            
        # Concatenate all blocks in the current row horizontally.
        full_GGN_rows.append(torch.cat(current_row_blocks, dim = 1))
    
    # Concatenate all block rows vertically to form the final GGN matrix.
    return torch.cat(full_GGN_rows, dim = 0)

def get_per_sample_gradient(model, loss_function, inputs, targets):
    """
    Computes the gradient of the loss with respect to the model's parameters
    for each individual sample in a batch.

    Args:
        model (nn.Module): The neural network model.
        loss_function (callable): The loss function (e.g., nn.CrossEntropyLoss).
        inputs (torch.Tensor): A batch of inputs with shape [B, ...].
        targets (torch.Tensor): A batch of targets with shape [B, ...].

    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are parameter names (e.g., 'fc1.weight')
                                 and values are the corresponding gradients for each sample.
                                 The shape of a gradient tensor for a weight matrix will be
                                 [B, D_out, D_in].
    """
    
    # Create a dictionary of the model's named parameters.
    params = dict(model.named_parameters())  # ordered param dict

    # Define a stateless version of the model.
    def fmodel(params, x):
        return functional_call(model, params, (x,))  # returns logits of shape [C]

    # Define a function that computes the loss for a single data point.
    def compute_loss(params, x, y):
        pred = fmodel(params, x)
        loss = loss_function(pred, y.squeeze(0))
        return loss
    
    # Create a function that computes the gradient of the loss w.r.t. its first argument (the parameters).
    grad_of_loss = grad(compute_loss, argnums=0)

    # Vectorize the gradient computation over the batch of inputs and targets.
    # `vmap` maps `grad_of_loss` over the batch dimension (0) of both `inputs` and `targets`.  
    per_sample_gradient = vmap(grad_of_loss, in_dims=(None, 0, 0))(params, inputs, targets)
    
    return per_sample_gradient  # dict {'fc1.weight': [B, Dout, Din], }


def get_EF_Hessian(per_sample_gradient, layer_name_list):
    """
    Constructs the full Empirical Fisher (EF) matrix, which approximates the Hessian.

    The EF is the sum of the outer products of per-sample gradients:
    H_ef = sum_n( grad_n @ grad_n.T )
    where grad_n is the gradient of the loss for the n-th sample. 

    Args:
        per_sample_gradient (dict[str, torch.Tensor]): A dictionary of per-sample gradients. {'fc1.weight': [B, D_out, D_in], ...}
        layer_name_list (list[str]): List with each layers name

    Returns:
        torch.Tensor: The full Empirical Fisher matrix, assembled from all blocks,
                      with shape [P, P], where P is the total number of model parameters.
    """
    
    # The nested loops below construct the full EF matrix block by block.
    # The (i, j) block is (∂ℓ_n/∂θ^(i))^T @ ∂ℓ_n/∂θ^(j) where ∂ℓ_n/∂θ^(i) is the gradient of the loss for sample n with respect to the parameters of layer i.
    
    full_EF_rows = []
    
    for layer_name_1 in ['fc1', 'fc2', 'fc3']:
        current_row_blocks = []
        for layer_name_2 in ['fc1', 'fc2', 'fc3']:
            # Compute current EF block
            grad1 = per_sample_gradient[f"{layer_name_1}.weight"].flatten(start_dim = 1).unsqueeze(2) # [B, Dout_1, Din_1] -> [B, Dout_1 * Din_1] ->  [B, Dout_1 * Din_1, 1]
            grad2 = per_sample_gradient[f"{layer_name_2}.weight"].flatten(start_dim = 1).unsqueeze(1) # [B, Dout_2, Din_2] -> [B, Dout_2 * Din_2] -> [B, 1, Dout_2 * Din_2]
            cur_EF_H = grad1 @ grad2
            
            # Sum over the batch dimension to get the final EF block.
            current_row_blocks.append(cur_EF_H.sum(0).detach())
        
        # Concatenate all blocks in the current row horizontally.
        full_EF_rows.append(torch.cat(current_row_blocks, dim = 1))
        
    # Concatenate all block rows vertically to form the final EF matrix.
    return torch.cat(full_EF_rows, dim = 0)


class FullLA:
    
    def __init__(self, model, prior_prec, hessian_approx):
        
        self.model = model
        self.layer_name_list = list(dict.fromkeys([name.split('.')[0] for name, _ in model.named_parameters() if 'weight' in name]))
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
            
            GGN_H = get_GGN_Hessian(per_sample_jacobian, ce_hessian, self.layer_name_list)
            self.H = GGN_H
        
        if self.hessian_approx == 'EF':
            loss_function = nn.CrossEntropyLoss(reduction='sum')
            per_sample_gradient = get_per_sample_gradient(self.model, loss_function, X, Y)
            
            EF_H = get_EF_Hessian(per_sample_gradient, self.layer_name_list)
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

        

        
        
        

            
            
        
        