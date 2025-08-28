## LA-from-Scratch

This repo aims to implement **Laplace Approximation (LA)** from scratch for a tiny MLP with Pytorch.

Each posterior structure is implemented in a self-contained, single Python file, making it easy to follow. 

> If you are looking for simple-to-use LA library, the excellent [`laplace-redux`](https://github.com/AlexImmer/laplace-redux) is where you should go.

## What's implemented

* **Posterior Covariance Structures**:
    * Full Covariance
    * Diagonal Covariance
    * KFAC Covariance (TODO)
      
* **Hessian Approximations**:
    * Generalized Gauss-Newton (GGN)
    * Empirical Fisher (EF)

* **Prior optimising**:
    * Marginal Likelihood

* **Prediction Methods**:
    * Sampling
    * GLM (Generalized Linear Model)(TODO) 

## File Structure

The implementation is broken down by the covariance structure.

### LA with full covariance

`full_la.py` implements LA with a full covariance matrix. The Hessian can be estimated using either GGN or EF. This method is computationally expensive but captures the full parameter correlations.

### LA with diagonal covariance

`diag_la.py` implements LA with a diagonal covariance matrix. The Hessian can be estimated using either GGN or EF. This is a highly efficient and scalable version of Laplace that assumes no correlation between parameters.
