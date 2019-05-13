import torch


def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
    nx, ny = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny
