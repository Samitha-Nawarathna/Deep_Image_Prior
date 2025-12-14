import torch
import torch.nn as nn

class HourglassNetwork(nn.Module):
    def __init__(self, n_u, n_d, k_d, k_u, n_s, k_s, upsampling, use_bn, use_sigmoid, in_channel, out_channel, activation):
        super(HourglassNetwork, self).__init__()
        
        self.n_u = n_u
        self.n_d = n_d
        self.n_s = n_s
        self.use_bn = use_bn
        self.upsampling = upsampling
        self.activation_type = activation  # Store the string, not the object

        self.encoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # ... (Validation checks remain the same) ...

        # 1. Build Encoder and Skip Paths
        current_channels = in_channel
        for i in range(len(n_u)):
            out_c = n_u[i]
            k = k_u[i]
            # Pass activation type string, not object
            self.encoders.append(self.encoder_block(current_channels, out_c, k))
            current_channels = out_c

            # Skip Connection
            skip_out_c = n_s[i]
            skip_k = k_s[i]
            if skip_out_c > 0:
                self.skips.append(self.skip_connection(current_channels, skip_out_c, skip_k))
            else:
                self.skips.append(nn.Identity())

        # 2. Bottleneck
        self.bottleneck = self._conv_block(current_channels, current_channels, 3, 1, 1)

        # 3. Decoder Path
        for i in range(len(n_u) - 1, -1, -1):
            skip_channels = n_s[i]
            decoder_in_channels = current_channels + skip_channels
            decoder_out_channels = n_d[i]
            k = k_d[i]
            self.decoders.append(self.decoder_block(decoder_in_channels, decoder_out_channels, k))
            current_channels = decoder_out_channels

        # 4. Final Output
        self.final_conv = nn.Conv2d(current_channels, out_channel, kernel_size=1)
        self.final_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def _get_activation(self):
        """Helper to create a NEW activation instance every time."""
        if self.activation_type == 'leakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif self.activation_type == 'ReLU':
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Activation {self.activation_type} not recognized.")

    def _activation_block(self, n_ch):
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm2d(n_ch))
        layers.append(self._get_activation()) # Create NEW instance
        return nn.Sequential(*layers)

    def _conv_block(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            self._activation_block(out_ch)
        )

    def encoder_block(self, in_ch, out_ch, kernel_size):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            # 1. Conv + Downsample
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=2, padding=padding),
            # 2. Activation
            self._activation_block(out_ch),
            # 3. Conv Block (includes Act)
            self._conv_block(out_ch, out_ch, kernel_size, stride=1, padding=padding)
            # REMOVED the redundant 5th step (self._activation_block)
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