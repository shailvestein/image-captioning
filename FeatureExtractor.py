# Advanced Feature Extractor module utilizing pretrained EfficientNet-B4 architecture
class FeatureExtractor(nn.Module):
    def __init__(self, fine_tune=False):
        super(FeatureExtractor, self).__init__()
        
        # Step 1: Load EfficientNet-B4 backbone model pretrained on ImageNet
        efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        
        # Step 2: Separate spatial feature extraction layers from the classification head
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Step 3: Freeze or unfreeze parameters based on the fine_tune flag
        for param in self.parameters():
            param.requires_grad = fine_tune

    def forward(self, img_tensor):
        # Step 4: Pass image tensor through feature layers and global average pooling
        x = self.features(img_tensor)
        x = self.avgpool(x)
        
        # Step 5: Flatten pooled features into a 1792-dimensional vector format
        features = torch.flatten(x, 1)
        return features

# Define image preprocessing transformation pipeline optimized for EfficientNet-B4
def get_image_transform():
    return transforms.Compose([
        # Step 1: Resize images to 380x380 resolution required by EfficientNet-B4
        transforms.Resize((380, 380)),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        # Step 2: Normalize tensor color channels using ImageNet dataset mean and standard deviation
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

