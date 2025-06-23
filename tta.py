import math
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class RandomImageTransformer:
    def __init__(self, degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10),
                 padding_mode='border'):
        """
        Initialize the transformer with parameter ranges for augmentation.
        
        Parameters:
          degrees: tuple (min_angle, max_angle) in degrees for random rotation.
          translate: tuple (max_dx, max_dy) as a fraction of image dimensions.
          scale: tuple (min_scale, max_scale) for random scaling.
          shear: tuple (min_shear, max_shear) in degrees (shear along the x-axis).
          padding_mode: Specifies how to fill areas outside the input. Options:
                        'zeros' (default), 'border', or 'reflection'.
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.padding_mode = padding_mode

    def _get_forward_affine_matrix(self, center, angle, translate, scale, shear):
        """
        Build a 3x3 affine transformation matrix in pixel coordinates.
        
        The transformation is defined as:
           M = T(translation) · T(center) · (R · S · Sc) · T(-center)
        
        Parameters:
          center: tuple (cx, cy) center of the image.
          angle: rotation angle in degrees.
          translate: tuple (tx, ty) translation in pixels.
          scale: scale factor (a float).
          shear: shear angle in degrees (applied along the x-axis).
          
        Returns:
          A 3x3 torch.Tensor representing the forward affine transformation.
        """
        cx, cy = center
        tx, ty = translate
        angle_rad = math.radians(angle)
        shear_rad = math.radians(shear)

        # Rotation matrix.
        R = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0],
            [0,                   0,                  1]
        ])

        # Shear matrix (shear along x axis).
        S = torch.tensor([
            [1, math.tan(shear_rad), 0],
            [0, 1,                   0],
            [0, 0,                   1]
        ])

        # Scale matrix.
        Sc = torch.tensor([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ])

        # Combine scale, shear, and rotation.
        A = R @ S @ Sc

        # Translation matrices to shift the center.
        T_neg = torch.tensor([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])
        T_pos = torch.tensor([
            [1, 0, cx + tx],
            [0, 1, cy + ty],
            [0, 0, 1]
        ])

        # Full forward affine matrix.
        M = T_pos @ A @ T_neg
        return M

    @staticmethod
    def _convert_affine_matrix_to_theta(M, width, height):
        """
        Convert a 3x3 affine matrix in pixel coordinates into a 2x3 matrix in normalized coordinates.
        
        PyTorch's grid sampling expects a 2x3 matrix in normalized coordinates (in [-1, 1]).
        
        Parameters:
          M: 3x3 affine matrix (torch.Tensor)
          width: image width in pixels.
          height: image height in pixels.
          
        Returns:
          A 2x3 torch.Tensor usable by F.affine_grid.
        """
        # Matrix that converts normalized coordinates to pixel coordinates.
        T_denorm = torch.tensor([
            [width / 2.0, 0,           width / 2.0],
            [0,           height / 2.0, height / 2.0],
            [0,           0,           1]
        ])
        # Matrix that converts pixel coordinates to normalized coordinates.
        T_norm = torch.tensor([
            [2.0 / width, 0,          -1],
            [0,          2.0 / height, -1],
            [0,          0,           1]
        ])

        theta = T_norm @ M @ T_denorm
        return theta[:2, :]

    def transform(self, image):
        """
        Randomly transforms an image and returns the transformed image along with the inverse transformation matrix.
        
        Parameters:
          image: a torch.Tensor of shape (C, H, W).
          
        Returns:
          transformed: the transformed image (torch.Tensor of shape (C, H, W)).
          M_inv: a 3x3 torch.Tensor representing the inverse transformation (in pixel coordinates).
        """
        # Image dimensions.
        C, H, W = image.shape
        center = (W / 2.0, H / 2.0)

        # Sample random parameters.
        angle = random.uniform(self.degrees[0], self.degrees[1])
        max_dx = self.translate[0] * W
        max_dy = self.translate[1] * H
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear_angle = random.uniform(self.shear[0], self.shear[1])

        # Compute the forward affine transformation matrix (pixel coordinates).
        M = self._get_forward_affine_matrix(center, angle, (tx, ty), scale_factor, shear_angle)
        # Compute its inverse (to later restore the image).
        M_inv = torch.inverse(M)

        # Convert the forward matrix to a 2x3 matrix in normalized coordinates.
        theta = self._convert_affine_matrix_to_theta(M, W, H)

        # Prepare for grid sampling.
        image_batch = image.unsqueeze(0)  # Add batch dimension.
        grid = F.affine_grid(theta.unsqueeze(0), image_batch.size(), align_corners=False)
        transformed = F.grid_sample(image_batch, grid, align_corners=False, padding_mode=self.padding_mode)

        # Remove batch dimension.
        transformed = transformed.squeeze(0)
        return transformed, M_inv

# ------------------ Example Usage ------------------
if __name__ == '__main__':
    import PIL.Image as Image
    import matplotlib.pyplot as plt

    # Load an image and convert to a tensor.
    image_pil = Image.open("car.png").convert("RGB")
    image = T.ToTensor()(image_pil)  # Shape: (C, H, W)
    
    # Create an instance of the transformer using a padding mode that fills empty areas
    # Here, 'border' will fill with the border values of the image.
    padding_modes = ['zeros', 'border', 'reflection']
    transformer = RandomImageTransformer(degrees=(-30, 30),
                                         translate=(0.1, 0.1),
                                         scale=(0.9, 1.1),
                                         shear=(-10, 10),
                                         padding_mode=padding_modes[1])
    
    # Apply a random transformation.
    transformed_img, inverse_matrix = transformer.transform(image)
    
    print("Transformed image shape:", transformed_img.shape)
    print("Inverse transformation matrix:\n", inverse_matrix)

    # Plot the original, transformed, and restored images.
    plt.figure(figsize=(15, 5))
    
    # Original image.
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')

    # Transformed image.
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_img.permute(1, 2, 0))
    plt.title("Transformed Image")
    plt.axis('off')

    # Restore the original image by applying the inverse transformation.
    C, H, W = transformed_img.shape
    theta_inv = RandomImageTransformer._convert_affine_matrix_to_theta(inverse_matrix, W, H)
    
    transformed_img_batch = transformed_img.unsqueeze(0)
    grid_inv = F.affine_grid(theta_inv.unsqueeze(0), transformed_img_batch.size(), align_corners=False)
    restored_img = F.grid_sample(transformed_img_batch, grid_inv, align_corners=False, padding_mode=transformer.padding_mode)
    restored_img = restored_img.squeeze(0)
    
    plt.subplot(1, 3, 3)
    plt.imshow(restored_img.permute(1, 2, 0))
    plt.title("Restored Image")
    plt.axis('off')

    plt.show()
