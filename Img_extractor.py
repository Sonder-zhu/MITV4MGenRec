import torch
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import vit_base_patch16_clip_224

class ImgDataset(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class ImgExtractTool:
    # './pretrained/vit_base_patch16_clip_224.pth'
    def __init__(self,
                 device, 
                 img_path,
                 img_dim_n,
                 model_path):
        self.device = device
        self.model_path = model_path
        self.basic_path = img_path
        self.img_dim_n = img_dim_n
        self.model = self.load_weight()
        self.data_transform = {
            'val': transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    def load_weight(self):
        model = vit_base_patch16_clip_224()
        weights_dict = torch.load(self.model_path)
        print(model.load_state_dict(weights_dict, strict=False))
        return model.to(self.device)

    @staticmethod
    def replace_RGB(file_path):
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img_rgb = img.convert("RGB")
            os.remove(file_path)
            img_rgb.save(file_path)

    def extract_one_instance(self, instance):
        target_path = self.basic_path
        image_name = instance + '.jpg'
        image_path = os.path.join(target_path, image_name)
        try:
            self.replace_RGB(image_path)
        except:
            return torch.zeros((1, self.img_dim_n), dtype=torch.float)

        dataset = ImgDataset([image_path], [0], transform=self.data_transform['val'])
        assert len(dataset) <= 1
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)
        i = 0
        for _, data in enumerate(dataloader): 
            img, _ = data
            output = self.model.forward_features(img.to(self.device)).cpu()
            feature = output[:, 0].view(1, self.img_dim_n)
            return feature

        return torch.zeros((1, self.img_dim_n), dtype=torch.float)
    
def img_extractor(device, img_path, item_id_list, img_dim, model_path, padding_idx=0):
    img_dim_n = int(img_dim)
    with torch.no_grad():
        tool = ImgExtractTool(device, img_path, img_dim_n, model_path)
        result = []
        for _, t in tqdm(enumerate(item_id_list), desc='Image Extracting', total=len(item_id_list)):
            result.append(tool.extract_one_instance(t))
        img_emb = torch.cat(result, dim=0)
    return img_emb

