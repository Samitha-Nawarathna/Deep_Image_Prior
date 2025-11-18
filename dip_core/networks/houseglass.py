import torch
import torch.nn as nn
import torch.nn.functional as F

class HourglassNetwork(nn.Module):
    """
    A U-Net/Hourglass-style network built based on user-defined specifications
    for encoder, decoder, and skip connection blocks.

    The architecture is dynamic based on the lists provided for channels and kernels.
    """
    def __init__(self, n_u, n_d, k_d, k_u, n_s, k_s, upsampling, use_bn, use_sigmoid, in_channel, out_channel, activation):
        """
        Initializes the Hourglass Network.

        Parameters:
        n_u (list): List of output channels for each encoder block.
        n_d (list): List of output channels for each decoder block.
        k_d (list): List of kernel sizes for each decoder block.
        k_u (list): List of kernel sizes for each encoder block.
        n_s (list): List of output channels for each skip connection. (0 disables skip)
        k_s (list): List of kernel sizes for each skip connection.
        upsampling (str): Upsampling mode ('nearest', 'bilinear', etc.).
        use_bn (bool): Whether to use BatchNorm2d in activation blocks.
        use_sigmoid (bool): Whether to apply a sigmoid to the final output.
        in_channel (int): Number of input channels to the network.
        out_channel (int): Number of output channels from the network.
        activation (str): Activation function name ('leakyReLU', 'ReLU', etc.).
        """
        super(HourglassNetwork, self).__init__()

        # --- Store configuration ---
        self.n_u = n_u
        self.n_d = n_d
        self.n_s = n_s
        self.use_bn = use_bn
        self.upsampling = upsampling
        
        # --- Define activation layer ---
        if activation == 'leakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        # Add other activations as needed
        else:
            raise ValueError(f"Activation '{activation}' not recognized.")

        # --- Module Lists ---
        self.encoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.decoders = nn.ModuleList()

        n_layers = len(n_u)
        if n_layers == 0:
            raise ValueError("n_u (encoder channels) list cannot be empty.")
            
        if not all(len(lst) == n_layers for lst in [n_d, k_d, k_u, n_s, k_s]):
            raise ValueError("All configuration lists (n_u, n_d, k_d, k_u, n_s, k_s) must have the same length.")

        # ==================================
        # 1. Build Encoder and Skip Paths
        # ==================================
        current_channels = in_channel
        for i in range(n_layers):
            # --- Encoder Block ---
            out_c = n_u[i]
            k = k_u[i]
            encoder = self.encoder_block(current_channels, out_c, k)
            self.encoders.append(encoder)
            current_channels = out_c

            # --- Skip Connection Block ---
            skip_out_c = n_s[i]
            skip_k = k_s[i]
            if skip_out_c > 0:
                # Only build skip_connection module if n_s[i] > 0
                skip = self.skip_connection(current_channels, skip_out_c, skip_k)
                self.skips.append(skip)
            else:
                # Use Identity as a placeholder if skip is disabled
                # This layer won't be used in forward pass if n_s[i] is 0
                self.skips.append(nn.Identity())

        # ==================================
        # 2. Build Bottleneck
        # ==================================
        # A simple conv block at the deepest part of the network
        self.bottleneck = self._conv_block(current_channels, current_channels, kernel_size=3, stride=1, padding=1)

        # ==================================
        # 3. Build Decoder Path
        # ==================================
        # Loop backwards from the last layer to the first
        for i in range(n_layers - 1, -1, -1):
            skip_channels = n_s[i]
            # Input to decoder is (previous_decoder_output + skip_connection_output)
            decoder_in_channels = current_channels + skip_channels
            decoder_out_channels = n_d[i]
            k = k_d[i]
            
            decoder = self.decoder_block(decoder_in_channels, decoder_out_channels, k)
            self.decoders.append(decoder)
            current_channels = decoder_out_channels # Output of this decoder is input to next

        # ==================================
        # 4. Final Output Layer
        # ==================================
        self.final_conv = nn.Conv2d(current_channels, out_channel, kernel_size=1, stride=1, padding=0)
        
        if use_sigmoid:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity() # No-op

    def _activation_block(self, n_ch):
        """
        Returns an activation block (BatchNorm + Activation or just Activation).
        """
        if self.use_bn:
            return nn.Sequential(
                nn.BatchNorm2d(n_ch),
                self.activation
            )
        else:
            return self.activation

    def _conv_block(self, in_ch, out_ch, kernel_size, stride, padding):
        """
        Returns a (Conv2d -> ActivationBlock) sequence.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            self._activation_block(out_ch)
        )

    def encoder_block(self, in_ch, out_ch, kernel_size):
        """
        Returns one encoder block as specified:
        1. Conv2d (maps channels, no size change)
        2. AvgPool2d (downsample)
        3. ActivationBlock
        4. _conv_block (no size change)
        5. ActivationBlock
        
        Note: The sequence (ConvBlock -> Activation) is redundant but
        implemented as requested.
        """
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            # 1. Conv2d
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding),
            # 2. Downsample
            nn.AvgPool2d(2, 2),
            # 3. ActivationBlock
            self._activation_block(out_ch),
            # 4. _conv_block (Conv -> Act)
            self._conv_block(out_ch, out_ch, kernel_size, stride=1, padding=padding),
            # 5. ActivationBlock (Note: This is the redundant activation)
            self._activation_block(out_ch)
        )

    def skip_connection(self, in_ch, out_ch, kernel_size):
        """
        Returns a skip connection block:
        1. _conv_block (maps channels, no size change)
        """
        padding = (kernel_size - 1) // 2
        return self._conv_block(in_ch, out_ch, kernel_size, stride=1, padding=padding)

    def decoder_block(self, in_ch, out_ch, kernel_size):
        """
        Returns one decoder block as specified:
        1. BatchNorm2d
        2. _conv_block (maps channels)
        3. _conv_block (no size change)
        4. Upsample
        """
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            # 1. BatchNorm2d
            nn.BatchNorm2d(in_ch),
            # 2. _conv_block (maps channels)
            self._conv_block(in_ch, out_ch, kernel_size, stride=1, padding=padding),
            # 3. _conv_block (no size change)
            self._conv_block(out_ch, out_ch, kernel_size, stride=1, padding=padding),
            # 4. Upsample
            nn.Upsample(scale_factor=2, mode=self.upsampling)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        skip_outputs = []
        x_current = x

        # --- Encoder Path ---
        # 
        for i in range(len(self.encoders)):
            x_current = self.encoders[i](x_current)
            # Store skip connection output
            if self.n_s[i] > 0:
                skip_val = self.skips[i](x_current)
                skip_outputs.append(skip_val)
            else:
                skip_outputs.append(None) # Append None if skip is disabled

        # --- Bottleneck ---
        x_current = self.bottleneck(x_current)

        # --- Decoder Path ---
        # 
        for i in range(len(self.decoders)):
            # Get skip connection in reverse order
            skip_val = skip_outputs.pop() 
            
            # Concatenate if skip connection exists
            if skip_val is not None:
                x_current = torch.cat([x_current, skip_val], dim=1)
            
            x_current = self.decoders[i](x_current)

        # --- Final Output ---
        x_out = self.final_conv(x_current)
        x_out = self.final_act(x_out)

        return x_out

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters provided by the user
    params = {
        'n_u': [16, 32, 64, 128, 128, 128],
        'n_d': [16, 32, 64, 128, 128, 128],
        'k_d': [3, 3, 3, 3, 3, 3],
        'k_u': [5, 5, 5, 5, 5, 5],
        'n_s': [0, 0, 0, 0, 0, 0], # Skips disabled
        'k_s': [0, 0, 0, 0, 0, 0],
        'upsampling': 'nearest',
        'use_bn': True,
        'use_sigmoid': True,
        'in_channel': 2,
        'out_channel': 3,
        'activation': 'leakyReLU'
    }

    # --- Test with skips disabled (as per user params) ---
    print("--- Testing with user parameters (skips disabled) ---")
    model_no_skip = HourglassNetwork(**params)
    
    # Create a dummy input tensor
    # Batch size 1, 2 channels, 256x256 image
    # Note: Encoder downsamples 6 times (2^6 = 64). Input must be divisible by 64.
    dummy_input = torch.randn(1, 2, 256, 256) 
    
    try:
        output = model_no_skip(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        # The output size should be (1, 3, 256, 256)
        # Decoder upsamples 6 times.
        assert output.shape == (1, params['out_channel'], dummy_input.shape[2], dummy_input.shape[3])
        print("Model (no skip) forward pass successful!")
    except Exception as e:
        print(f"Model (no skip) forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Test with skips enabled (modified params) ---
    print("\n--- Testing with modified parameters (skips enabled) ---")
    params_with_skip = params.copy()
    # Enable skip connections with matching channels
    params_with_skip['n_s'] = [16, 32, 64, 128, 128, 128] 
    params_with_skip['k_s'] = [1, 1, 1, 1, 1, 1] # Use 1x1 conv for skips

    model_with_skip = HourglassNetwork(**params_with_skip)
    
    try:
        output_skip = model_with_skip(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output_skip.shape}")
        assert output_skip.shape == (1, params['out_channel'], dummy_input.shape[2], dummy_input.shape[3])
        print("Model (with skip) forward pass successful!")
    except Exception as e:
        print(f"Model (with skip) forward pass failed: {e}")
        import traceback
        traceback.print_exc()