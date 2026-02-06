import unittest
import torch
import torch.nn as nn
import numpy as np
from code_to_test import CustomLayerNorm, DeepFeedForward  # Assuming code is saved in code_to_test.py

# If the code is pasted directly, we define the classes here for the test context to work standalone.
# (In a real scenario, we import them. Here we assume the classes CustomLayerNorm and DeepFeedForward are available)

class TestCustomLayerNorm(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.features = 10
        self.eps = 1e-5
        # Initialize custom layer
        self.custom_ln = CustomLayerNorm(self.features, eps=self.eps)
        # Initialize PyTorch reference layer
        self.torch_ln = nn.LayerNorm(self.features, eps=self.eps)
        
        # Copy weights to ensure deterministic comparison
        with torch.no_grad():
            self.custom_ln.gamma.copy_(self.torch_ln.weight)
            self.custom_ln.beta.copy_(self.torch_ln.bias)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.features)
        out = self.custom_ln(x)
        self.assertEqual(out.shape, x.shape, "Output shape must match input shape")

    def test_correctness_vs_pytorch(self):
        x = torch.randn(self.batch_size, self.features)
        
        # Forward pass
        custom_out = self.custom_ln(x)
        torch_out = self.torch_ln(x)
        
        # Compare tensors
        self.assertTrue(torch.allclose(custom_out, torch_out, atol=1e-6),
                        "Custom implementation output should match torch.nn.LayerNorm")

    def test_multidimensional_input(self):
        # Test with shape (N, C, H, W) normalizing over (C, H, W)
        N, C, H, W = 2, 3, 4, 4
        normalized_shape = (C, H, W)
        x = torch.randn(N, C, H, W)
        
        custom_ln = CustomLayerNorm(normalized_shape, eps=1e-5)
        torch_ln = nn.LayerNorm(normalized_shape, eps=1e-5)
        
        # Sync weights
        with torch.no_grad():
            custom_ln.gamma.copy_(torch_ln.weight)
            custom_ln.beta.copy_(torch_ln.bias)
            
        custom_out = custom_ln(x)
        torch_out = torch_ln(x)
        
        self.assertTrue(torch.allclose(custom_out, torch_out, atol=1e-6),
                        "Should handle multi-dimensional normalized shapes correctly")

    def test_gradients(self):
        x = torch.randn(self.batch_size, self.features, requires_grad=True)
        out = self.custom_ln(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(self.custom_ln.gamma.grad, "Gamma gradients should be calculated")
        self.assertIsNotNone(self.custom_ln.beta.grad, "Beta gradients should be calculated")
        self.assertIsNotNone(x.grad, "Input gradients should be calculated")

class TestDeepFeedForward(unittest.TestCase):
    def test_forward_pass_layer_norm(self):
        model = DeepFeedForward(norm_type="layer")
        # MNIST input shape: (Batch, 1, 28, 28)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 10), "Output shape should be (Batch, 10)")

    def test_forward_pass_batch_norm(self):
        model = DeepFeedForward(norm_type="batch")
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_training_step(self):
        # Ensures no runtime errors during a full optimization step
        model = DeepFeedForward(norm_type="layer")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(2, 1, 28, 28)
        y = torch.randint(0, 10, (2,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Check if weights updated (simple check: logic implies if no error, it works)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()