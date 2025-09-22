import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel for blurring."""
    coords = torch.arange(size, dtype=torch.float32)
    coords = coords - size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D kernel
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.view(1, 1, size, size)


def gradient_descent(input, model, loss, iterations=256, l2_reg=0.001, octave_scale=1.5, num_octaves=3):
    """
    Enhanced gradient descent to maximize the class logit with improvements from research papers.
    
    Based on insights from:
    - Simonyan et al. (2013): L2 regularization, class score optimization
    - Google Inceptionism: Multi-octave processing, feature visualization
    
    Args:
        input: Input image tensor (1, 3, H, W) with requires_grad=True
        model: Pre-trained VGG model
        loss: Loss function that takes model output and returns scalar
        iterations: Number of optimization iterations per octave
        l2_reg: L2 regularization strength (from Simonyan et al.)
        octave_scale: Scale factor between octaves (from Inceptionism)
        num_octaves: Number of octaves to process (from Inceptionism)
    
    Returns:
        Optimized input image tensor
    """
    # Store original input size
    original_size = input.shape[-2:]
    
    # Multi-octave processing (from Inceptionism paper)
    for octave in range(num_octaves):
        # Calculate current octave size
        octave_size = (int(original_size[0] / (octave_scale ** (num_octaves - 1 - octave))),
                      int(original_size[1] / (octave_scale ** (num_octaves - 1 - octave))))
        
        if octave == 0:
            # Start with small image
            current_input = torch.nn.functional.interpolate(input, size=octave_size, mode='bilinear', align_corners=False)
        else:
            # Upscale from previous octave
            current_input = torch.nn.functional.interpolate(previous_input, size=octave_size, mode='bilinear', align_corners=False)
        
        # Detach and create a new leaf tensor to avoid optimization issues
        current_input = current_input.detach().clone().requires_grad_(True)
        
        # Set up optimizer with adaptive learning rate
        learning_rate = 0.01 * (0.8 ** octave)  # Decrease learning rate for higher octaves
        optimizer = torch.optim.Adam([current_input], lr=learning_rate, weight_decay=l2_reg)
        
        print(f"Processing octave {octave + 1}/{num_octaves} (size: {octave_size})")
        
        # Optimization loop for current octave
        for i in tqdm(range(iterations), desc=f"Octave {octave + 1}"):
            # Zero gradients
            optimizer.zero_grad()
            
            # Normalize and jitter the input
            normalized_input = normalize_and_jitter(current_input)
            
            # Forward pass through model
            output = model(normalized_input)
            
            # Compute class-score loss (Simonyan et al.)
            class_loss = -loss(output)  # Negative because we want to maximize
            current_loss = class_loss
            
            # Backward pass
            current_loss.backward()

            # Gradient normalization (RMS) and light gradient blur
            if current_input.grad is not None:
                rms = current_input.grad.pow(2).mean().sqrt().clamp_min(1e-8)
                current_input.grad.div_(rms)
                # 3x3 Gaussian to suppress high-frequency noise in updates
                grad_kernel = create_gaussian_kernel(3, 0.8).to(current_input.device)
                grad_kernel = grad_kernel.expand(3, 1, 3, 3)
                current_input.grad = torch.nn.functional.conv2d(
                    current_input.grad, grad_kernel, padding=1, groups=3
                )
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([current_input], max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Clamp pixel values to [0, 1] for valid image range
            with torch.no_grad():
                current_input.clamp_(0, 1)
            
            # Enhanced regularization techniques
            if i % 25 == 0 and i > 0:  # More frequent regularization
                with torch.no_grad():
                    # Gaussian blur for noise reduction
                    kernel_size = 3 + (i // 100)  # Increase blur over time
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    # Create Gaussian kernel
                    sigma = kernel_size / 6.0
                    kernel = create_gaussian_kernel(kernel_size, sigma).to(current_input.device)
                    kernel = kernel.expand(3, 1, kernel_size, kernel_size)
                    
                    # Apply blur with padding
                    padding = kernel_size // 2
                    blurred = torch.nn.functional.conv2d(current_input, kernel, padding=padding, groups=3)
                    current_input.data = 0.85 * current_input.data + 0.15 * blurred.data
        
        # Store result for next octave
        previous_input = current_input.detach()
    
    # Final upscale to original size
    final_input = torch.nn.functional.interpolate(previous_input, size=original_size, mode='bilinear', align_corners=False)
    
    return final_input


def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()