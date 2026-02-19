import torch
import torch.nn as nn
import kornia as K
import sys
import time

class TOVO(nn.Module):
    def __init__(self, color_space='hsv', default_iteration=5):
        super(TOVO, self).__init__()
        if color_space not in ['hsv', 'rgb']:
            raise ValueError("color_space must be either 'hsv' or 'rgb'")
        self.color_space = color_space
        self.default_iteration = default_iteration

    def enhance(self, x, iteration):
        E = ((1-torch.tanh(x))/2)
        for _ in range(iteration):
            x = x + (1-x)*torch.sinh(E*x)
        return x

    def forward(self, x, iteration=None):
        x = x / x.max().clamp_min(1e-6) #prevents division by zero if input is all zeros
        if iteration is None:
            iteration = self.default_iteration
        if self.color_space == 'hsv':
            h, s, v = torch.split(K.color.rgb_to_hsv(x), 1, dim=1)
            v_enhanced = self.enhance(v, iteration)
            enhanced_image = K.color.hsv_to_rgb(torch.cat([h, s, v_enhanced], dim=1))
        else:  # RGB direct enhancement
            enhanced_image = self.enhance(x, iteration)

        return enhanced_image

