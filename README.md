## LA-from-Scratch

This repo aims to implement **Laplace Approximation (LA)** from scratch for a tiny MLP with Pytorch.

Each posterior structure is implemented in a self-contained, single Python file, making it easy to follow. 

> If you are looking for simple-to-use LA library, the excellent [`laplace-redux`](https://github.com/AlexImmer/laplace-redux) is where you should go.

## What's implemented

* **Posterior Covariance Structures**:
    * Full Covariance
    * KFAC Covariance 
    * Diagonal Covariance

      
* **Hessian Approximations**:
    * Generalized Gauss-Newton (GGN)
    * Empirical Fisher (EF)



* **Prior optimising**:
    * Marginal Likelihood
    
* **Prediction Methods**:
    * Sampling

Hessian approximation is tested with [`curvlinops`](https://github.com/f-dangel/curvlinops) to ensure correctness. 
Marginal likelihood computation is tested with with [`laplace-redux`](https://github.com/AlexImmer/laplace-redux) to ensure correctness.


## File Structure

### LA with full covariance

`full_la.py` implements LA with a full covariance matrix. 
The Hessian can be estimated using either GGN or EF. 
This method is computationally expensive but captures the full parameter correlations.

### LA with KFAC covariance
`kfac_la.ipynb` implements LA with a Kronecker-factored (KFAC) covariance. 
The Hessian can be estimated using either GGN or EF. 
This method offers a compromise between the full and diagonal structures by dropping the cross-layer correlation.

### LA with diagonal covariance

`diag_la.py` implements LA with a diagonal covariance matrix. 
The Hessian can be estimated using either GGN or EF. 
This is a highly efficient and scalable version of Laplace that assumes no correlation between parameters.
