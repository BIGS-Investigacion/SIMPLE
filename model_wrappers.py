import torch

class Biomed(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        from open_clip import create_model_from_pretrained, get_tokenizer
        self.network, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)[0]

class RetCCL(torch.nn.Module):
    def __init__(self, checkpoint_path):
        from patho_networks import retccl_rn50
        super(RetCCL, self).__init__()

        self.network = retccl_rn50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        
        del self.network.fc
        self.network.fc = torch.nn.Identity()
        self.network.load_state_dict(torch.load(checkpoint_path), strict=True)

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Hiobu(torch.nn.Module):
    def __init__(self, arch):
        from transformers import AutoImageProcessor, AutoModel
        super(Hiobu, self).__init__()

        # histai/hibou-b histai/hibou-L
        self.network = AutoModel.from_pretrained(arch, trust_remote_code=True)

        image_mean = (0.7068, 0.5755, 0.722)
        image_std = (0.195, 0.2316, 0.1816)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x).pooler_output

class Uni(torch.nn.Module):
    def __init__(self, checkpoint_path):
        import timm
        super(Uni, self).__init__()

        self.network = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

        image_mean = (0.485, 0.456, 0.406) 
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Uni2(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(Uni2, self).__init__()
        import timm
        timm_kwargs = {
                'model_name': 'vit_giant_patch14_224', 'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
                'num_classes': 0, 'no_embed_class': True, 'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
                }
        self.network = timm.create_model(
            pretrained=False, **timm_kwargs
        )

        self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

        image_mean = (0.485, 0.456, 0.406) 
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Phikonv2(torch.nn.Module):
    def __init__(self, checkpoint_path):
        from transformers import AutoImageProcessor, AutoModel
        super(Phikonv2, self).__init__()

        self.network = AutoModel.from_pretrained("owkin/phikon-v2")

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x).last_hidden_state[:, 0, :]
    
class Virchow2(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super(Virchow2, self).__init__()
        import timm
        from timm.layers import SwiGLUPacked

        self.network = timm.create_model("hf_hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        image_mean = (0.4850, 0.4560, 0.4060)
        image_std = (0.2290, 0.2240, 0.2250)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        output = self.network(x)
        
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]

        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    
class Quilt(torch.nn.Module):

    def __init__(self, arch):
        super(Quilt, self).__init__()
        from open_clip import create_model_and_transforms, get_tokenizer

        # QuiltNet-B-16-PMB, QuiltNet-B-32, QuiltNet-B-16

        self.network, _, _ = create_model_and_transforms('hf-hub:wisdomik/' + arch)

        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)[0]
    
class Conch(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(Conch, self).__init__()
        from conch.open_clip_custom import create_model_from_pretrained

        self.network, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        self.network = self.network.visual.to(torch.device("cuda"))
        self.network.use_attentional_pool_contrast=False

        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network.forward_no_head(x, normalize=True)

class Pathoduet(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(Pathoduet, self).__init__()
        from patho_networks import pathoduet

        self.network = pathoduet(pretext_token=True, global_pool='avg')
        self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    def forward(self, x):
        x = self.network.forward_features(x)
        return x[:, self.network.num_prefix_tokens:].mean(dim=1)

class GPFM(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super(GPFM, self).__init__()
        from patho_networks import GPFM_get_model
        self.network = GPFM_get_model('GPFM', 0, 1)

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class mSTAR(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(mSTAR, self).__init__()
        import timm
        self.network = timm.create_model('hf-hub:Wangyh/mSTAR', pretrained=True, init_values=1e-5, dynamic_img_size=True)

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Hoptimus0(torch.nn.Module):

    def __init__(self, checkpoint_path):
        import timm
        super(Hoptimus0, self).__init__()
        self.network = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

        image_mean=(0.707223, 0.578729, 0.703617)
        image_std=(0.211883, 0.230117, 0.177517)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Hoptimus1(torch.nn.Module):

    def __init__(self, checkpoint_path):
        import timm
        super(Hoptimus1, self).__init__()
        self.network = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)

        image_mean=(0.707223, 0.578729, 0.703617)
        image_std=(0.211883, 0.230117, 0.177517)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class GigaPath(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(GigaPath, self).__init__()
        import timm
        self.network = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

        image_mean=(0.485, 0.456, 0.406)
        image_std=(0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Plip(torch.nn.Module):

    def __init__(self, checkpoint_path):
        from transformers import CLIPModel

        super(Plip, self).__init__()
        self.network = CLIPModel.from_pretrained("vinid/plip")

        image_mean=(0.48145466, 0.4578275, 0.40821073)
        image_std=(0.26862954, 0.26130258, 0.27577711)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network.get_image_features(pixel_values=x)

class CTransPath(torch.nn.Module):

    def __init__(self, checkpoint_path):
        from patho_networks import ctranspath
        super(CTransPath, self).__init__()
        self.network = ctranspath()
        self.network.head = torch.nn.Identity()
        self.network.load_state_dict(torch.load(checkpoint_path)['model'], strict=True)

        image_mean=(0.485, 0.456, 0.406)
        image_std=(0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)

class Musk(torch.nn.Module):

    def __init__(self, checkpoint_path):
        from patho_networks import musk_utils, musk_modeling
        import timm
        super(Musk, self).__init__()

        from trident.patch_encoder_models import encoder_factory
        self.network = encoder_factory(model_name='musk')

        image_mean = timm.data.constants.IMAGENET_INCEPTION_MEAN
        image_std = timm.data.constants.IMAGENET_INCEPTION_STD

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(384,384), mode='bicubic', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.network(x)

class Phikon(torch.nn.Module):
    def __init__(self, checkpoint_path):
        from transformers import AutoImageProcessor, AutoModel
        super(Phikon, self).__init__()

        self.network = AutoModel.from_pretrained("owkin/phikon")

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x).last_hidden_state[:, 0, :]

class Conchv1_5(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(Conchv1_5, self).__init__()
        from trident.patch_encoder_models import encoder_factory
        self.network = encoder_factory(model_name='conch_v15')

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        self.register_buffer('mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(image_std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.network(x)