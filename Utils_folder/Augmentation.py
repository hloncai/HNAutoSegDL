import kornia.augmentation as K
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from kornia.geometry.transform import warp_affine3d

class AugmentationPipeline_dist(nn.Module):
   def __init__(self) -> None:
      super(AugmentationPipeline_dist, self).__init__()
      self.aff = K.RandomAffine3D(degrees = 5, translate=(0,0.08,0.08),scale=(0.9,1.1),p=1)

   def forward(self, input, mask, dist):
      aff_params = self.aff.generate_parameters(input.shape)
      input = self.aff(input, aff_params)
      mask = self.aff(mask, aff_params)
      dist = self.aff(dist, aff_params)
      return input, mask, dist
      
class AugmentationPipeline(nn.Module):
   def __init__(self) -> None:
      super(AugmentationPipeline, self).__init__()
      self.aff = K.RandomAffine3D(degrees = 5, translate=(0,0.08,0.08),scale=(0.9,1.1),p=1)

   def forward(self, input, mask):
      aff_params = self.aff.generate_parameters(input.shape)
      input = self.aff(input, aff_params)
      mask = self.aff(mask, aff_params)
      return input, mask