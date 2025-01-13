import torch
import torch.nn as nn


class AdaptivePolyphaseSampling(nn.Module):
    """
    Adaptive Polyphase Sampling (APS) Layer.
    
    :param stride: The stride for downsampling.
    :param p: The norm to use for selecting the polyphase component.
    """
    
    def __init__(self, 
                 stride: int = 2, 
                 p: int = 2
                 ):
        
        super(AdaptivePolyphaseSampling, self).__init__()
        self.stride = stride
        self.p = p

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the APS layer. Given an input tensor ``x``, this method essentially applies a strided 
        reduction of the input shape by selecting the polyphase component with the maximum norm.
        
        :param x: Input tensor of shape (batch_size, channels, height, width)
        
        :returns: torch.Tensor of the shape (batch_size, channels, height // stride, width // stride)
        """
        b, c, h, w = x.shape
        s = self.stride
        # Generate polyphase components
        # polyphase: (stride^2, batch_size, channels, height, width)
        polyphase = [
            x[:, :, i::s, j::s] for i in range(s) for j in range(s)
        ]
        
        # Calculate the norm of each component
        # norms: (stride^2, batch_size)
        norms = [torch.norm(component, p=self.p, dim=(1, 2, 3)) for component in polyphase]
        norms = torch.stack(norms, dim=0) 
        
        # Select the component with the maximum norm
        max_indices = torch.argmax(norms, dim=0)  # Shape: (batch_size,)
        # Apply stride to the selected components
        selected = torch.stack([polyphase[i][batch_idx] for batch_idx, i in enumerate(max_indices)], dim=0)
        
        return selected
    



