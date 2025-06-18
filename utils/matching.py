import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def corr_to_matches(
    corr4d, 
    k_size=1, 
    do_softmax=False, 
    scale='positive', 
    return_indices=False,
    invert_matching_direction=False, 
    get_maximum=True, 
    device="cuda:0"
):

    device = corr4d.device
    batch_size, _, fs1, fs2, fs3, fs4 = corr4d.size()

    XA, YA = torch.meshgrid(
        torch.linspace(0, fs2 - 1, fs2 * k_size, device=device),
        torch.linspace(0, fs1 - 1, fs1 * k_size, device=device),
        indexing="ij"
    )
    XB, YB = torch.meshgrid(
        torch.linspace(0, fs4 - 1, fs4 * k_size, device=device),
        torch.linspace(0, fs3 - 1, fs3 * k_size, device=device),
        indexing="ij"
    )

    JA, IA = torch.meshgrid(torch.arange(fs2, device=device), torch.arange(fs1, device=device), indexing="ij")
    JB, IB = torch.meshgrid(torch.arange(fs4, device=device), torch.arange(fs3, device=device), indexing="ij")

    JA, IA = JA.reshape(1, -1), IA.reshape(1, -1)
    JB, IB = JB.reshape(1, -1), IB.reshape(1, -1)

    if invert_matching_direction:
        nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4).to(device)
        if do_softmax:
            nc_A_Bvec = F.softmax(nc_A_Bvec, dim=3)
        match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3) if get_maximum else torch.min(nc_A_Bvec, dim=3)
        score = match_A_vals.view(batch_size, -1)
        iB, jB = IB.view(-1)[idx_A_Bvec.view(-1)], JB.view(-1)[idx_A_Bvec.view(-1)]
        iA, jA = IA.expand_as(iB), JA.expand_as(jB)
    else:
        nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4).to(device)
        if do_softmax:
            nc_B_Avec = F.softmax(nc_B_Avec, dim=1)
        match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1) if get_maximum else torch.min(nc_B_Avec, dim=1)
        score = match_B_vals.view(batch_size, -1)
        import pdb; pdb.set_trace()
        iA, jA = IA.view(-1)[idx_B_Avec.view(-1)], JA.view(-1)[idx_B_Avec.view(-1)]
        iB, jB = IB.repeat(batch_size, 1), JB.repeat(batch_size, 1)
    

    xA = XA[iA.long(), jA.long()].view(batch_size, -1)
    yA = YA[iA.long(), jA.long()].view(batch_size, -1)
    xB = XB[iB.long(), jB.long()].view(batch_size, -1)
    yB = YB[iB.long(), jB.long()].view(batch_size, -1)

    if return_indices:
        return xA, yA, xB, yB, score, iA, jA, iB, jB
    else:
        return xA, yA, xB, yB, score
    
def corr_to_matches(corr4d, delta4d=None, k_size=1, do_softmax=False, scale='positive', return_indices=False,
                    invert_matching_direction=False, get_maximum=True, device='cuda:0'):
    """
    Modified from NC-Net. Perform argmax over the correlation.
    Args:
        corr4d: correlation, shape is b, 1, H_s, W_s, H_t, W_t
        delta4d:
        k_size:
        do_softmax:
        scale:
        return_indices:
        invert_matching_direction:
        get_maximum:

    Returns:

    """
    to_cuda = lambda x: x.to(device) if corr4d.is_cuda else x
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, fs2 * k_size), np.linspace(-1, 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(-1, 1, fs4 * k_size), np.linspace(-1, 1, fs3 * k_size))
    elif scale == 'positive':
        # keep normal range of coordinate
        XA, YA = np.meshgrid(np.linspace(0, fs2 - 1, fs2 * k_size), np.linspace(0, fs1 - 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(0, fs4 - 1, fs4 * k_size), np.linspace(0, fs3 - 1, fs3 * k_size))

    JA, IA = np.meshgrid(range(fs2), range(fs1))
    JB, IB = np.meshgrid(range(fs4), range(fs3))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).view(1, -1)))
    JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    if invert_matching_direction:
        nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        if get_maximum:
            match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3)
        else:
            match_A_vals, idx_A_Bvec = torch.min(nc_A_Bvec, dim=3)
        score = match_A_vals.view(batch_size, -1)

        iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        iA = IA.expand_as(iB)
        jA = JA.expand_as(jB)

    else:
        nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        if get_maximum:
            match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1)
        else:
            match_B_vals, idx_B_Avec = torch.min(nc_B_Avec, dim=1)
        score = match_B_vals.view(batch_size, -1)

        iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        iB = IB.expand_as(iA)
        jB = JB.expand_as(jA)

    if delta4d is not None:  # relocalization
        delta_iA, delta_jA, delta_iB, delta_jB = delta4d

        diA = delta_iA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        djA = delta_jA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        diB = delta_iB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        djB = delta_jB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]

        iA = iA * k_size + diA.expand_as(iA)
        jA = jA * k_size + djA.expand_as(jA)
        iB = iB * k_size + diB.expand_as(iB)
        jB = jB * k_size + djB.expand_as(jB)

    xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    xB = XB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)
    yB = YB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)

    # XA is index in channel dimension (source)
    if return_indices:
        return xA, yA, xB, yB, score, iA, jA, iB, jB
    else:
        return xA, yA, xB, yB, score