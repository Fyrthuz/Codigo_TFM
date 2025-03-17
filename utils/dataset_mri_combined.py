import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class LGGSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        """
        Args:
            root_dir (str): Root directory containing case subfolders.
            image_transform (callable, optional): Optional transform for images.
            mask_transform (callable, optional): Optional transform for masks.
        """
        self.root_dir = root_dir
        self.image_mask_pairs = []
        
        # Iterate over case folders
        for case in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case)
            if os.path.isdir(case_path):
                # Process each file in the case folder
                for file in os.listdir(case_path):
                    # Look for .tif files that are not masks (i.e. do not contain '_mask')
                    if file.lower().endswith('.tif') and '_mask' not in file:
                        image_path = os.path.join(case_path, file)
                        # Construct the corresponding mask filename
                        base, ext = os.path.splitext(file)
                        mask_file = base + '_mask' + ext
                        mask_path = os.path.join(case_path, mask_file)
                        if os.path.exists(mask_path):
                            self.image_mask_pairs.append((image_path, mask_path))
                        else:
                            print(f'Warning: Mask not found for image {image_path}')
        
        # Sort the pairs for consistency
        self.image_mask_pairs.sort(key=lambda x: x[0])
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Use the same read/transform approach as the previous simple dataset.
        if self.image_transform is None:
            image = transforms.ToTensor()(image)
        else:
            image = self.image_transform(image)
        
        if self.mask_transform is None:
            mask = transforms.ToTensor()(mask)
        else:
            mask = self.mask_transform(mask)
        
        return image, mask

    
    
# Example usage:
if __name__ == '__main__':
    # Replace with the path to your LGG Segmentation Dataset root directory
    root_directory = '/mnt/netapp2/Store_uni/home/usc/ci/fgs/Codigo/MRI/kaggle_3m'
    dataset = LGGSegmentationDataset(root_directory)
    
    # Create a DataLoader (adjust batch_size and num_workers as needed)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Iterate over the DataLoader
    for images, masks in dataloader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            images = images.to('cuda')
            masks = masks.to('cuda')
            # model = model.to('cuda')  # Ensure your model is moved to GPU
        
        # For example, perform inference with your model:
        # with torch.no_grad():
        #     outputs = model(images)
        # print(torch.round(outputs[0]))
        
        print(f'Batch - Images shape: {images.shape}, Masks shape: {masks.shape}')
        break