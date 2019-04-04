import torch


def create_grids(self, img_size, nG, device='cpu'):
    self.img_size = img_size
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)
    self.nG = torch.FloatTensor([nG]).to(device)
