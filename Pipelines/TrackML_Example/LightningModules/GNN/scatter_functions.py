import torch


def scatter_add(src,
    index,
    dim=-1,
    dim_size=None,
    out=None,
):
    """
    Scatter add function that mimics PyTorch's scatter_add, but exportable to ONNX.
    """

    index_unique, inverse = index.unique(return_inverse=True)

    if dim_size is None:
        dim_size = index_unique.numel()

    if out is None:
        out = torch.zeros(
            (dim_size, *src.shape[1:]),
            dtype=src.dtype,
            device=src.device,
        )
    else:
        out = out.to(src.device)
        out.zero_()

    out.index_add_(0, index_unique, src)
    out = out.index_select(0, inverse)
    out = out.index_add(dim, index, src)
    return out


def scatter_max(src,
    index,
    dim=-1,
    dim_size=None,
    out=None,
):
    """
    Scatter max function that mimics PyTorch's scatter_max, but exportable to ONNX.
    """

    index_unique, inverse = index.unique(return_inverse=True)

    if dim_size is None:
        dim_size = index_unique.numel()

    if out is None:
        out = torch.full(
            (dim_size, *src.shape[1:]),
            -float("inf"),
            dtype=src.dtype,
            device=src.device,
        )
        out_mask = torch.zeros_like(out, dtype=torch.bool)
    else:
        out = out.to(src.device)
        out.zero_()
        out_mask = torch.zeros_like(out, dtype=torch.bool)

    out.index_copy_(0, index_unique, src)
    out_mask.index_fill_(0, index_unique, True)
    out = out.index_select(0, inverse)
    out_mask = out_mask.index_select(0, inverse)

    return out.index_add(dim, index, src), out_mask


def scatter_mean(src,
    index,
    dim=-1,
    dim_size=None,
    out=None,
):
    """
    Scatter mean function that mimics PyTorch's scatter_mean, but exportable to ONNX.
    """

    index_unique, inverse = index.unique(return_inverse=True)

    if dim_size is None:
        dim_size = index_unique.numel()

    if out is None:
        out = torch.zeros(
            (dim_size, *src.shape[1:]),
            dtype=src.dtype,
            device=src.device,
        )
        count = torch.zeros_like(out, dtype=torch.float)
    else:
        out = out.to(src.device)
        out.zero_()
        count = torch.zeros_like(out, dtype=torch.float)

    count.index_add_(0, index_unique, torch.ones_like(src))
    out.index_add_(0, index_unique, src)
    out = out.index_select(0, inverse)
    count = count.index_select(0, inverse)

    return out.index_add(dim, index, src) / count