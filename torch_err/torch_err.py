import torch

def error(x:torch.Tensor, y:torch.Tensor, x_err:torch.Tensor):
    """
    Compute the error of outputs y with a given error x_err on the inputs x,
    explicitly without knowing the function f.
    x and x_err must be of the same shape [batch_size, n_inputs].
    y must be of shape [batch_size].
    returns error.
    """
    inputs = [x]
    outputs = list(y)
    gradient = torch.autograd.grad(outputs=outputs, inputs=inputs)
    return (gradient[0] * x_err.abs()).pow(2).sum(dim=-1).sqrt()

def ferror(f, x:torch.Tensor, x_err:torch.Tensor, *args, **kwargs):
    """
    Compute the error of a function f with a given error x_err on the inputs x.
    x and x_err must be of the same shape [batch_size, n_inputs].
    returns outputs, error.
    """
    outputs = f(x, *args, **kwargs)
    err = error(x, outputs, x_err)
    return outputs, err

