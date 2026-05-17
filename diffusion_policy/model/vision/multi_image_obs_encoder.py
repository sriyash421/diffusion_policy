from typing import Union
import copy
import torch
import torch.nn as nn
try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
import matplotlib.pyplot as plt


# Custom GaussianNoise transform
class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return img
        noise = torch.randn_like(img) * self.std
        return img + noise


class TrainingOnlyWrapper(nn.Module):
    def __init__(self, transforms: list[nn.Module], debug: bool = False):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.debug = debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or len(self.transforms) == 0:
            return x
        out = x
        for i, t in enumerate(self.transforms):
            out = t(out)
            if self.debug:
                # Only plot for the first image in the batch
                img = out[0]
                # If CHW, move to HWC
                if img.dim() == 3 and img.shape[0] in [1, 3]:
                    img_np = img.detach().cpu().numpy()
                    img_np = img_np.transpose(1, 2, 0)
                else:
                    img_np = img.detach().cpu().numpy()
                # Clamp and convert to uint8 for display
                img_np = img_np.clip(0, 1)
                if img_np.shape[2] == 1:
                    img_np = img_np.squeeze(-1)
                plt.figure()
                plt.title(f"After transform {i}: {t.__class__.__name__}")
                plt.imshow(img_np)
                plt.axis('off')
                plt.show()
        return out


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, dict[str,nn.Module]],
            resize_shape: Union[tuple[int,int], dict[str,tuple], None]=None,
            crop_shape: Union[tuple[int,int], dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            extra_randomizations=None,
            feature_dim: int=None,
            depth_model: nn.Module = None
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        depth_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure extra randomizations
                randomization_list = []
                if extra_randomizations is not None:
                    for t in extra_randomizations:
                        name = t['name']
                        params = t['params']
                        if name == 'ColorJitter':
                            randomization_list.append(T.ColorJitter(**params))
                        elif name == 'GaussianBlur':
                            randomization_list.append(T.GaussianBlur(**params))
                        elif name == 'RandomGrayscale':
                            randomization_list.append(T.RandomGrayscale(**params))
                        elif name == 'GaussianNoise':
                            randomization_list.append(GaussianNoise(**params))
                        else:
                            raise ValueError(f"Unknown transform: {name}")
                randomization_module = TrainingOnlyWrapper(randomization_list, debug=False) if randomization_list else nn.Identity()
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, randomization_module, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'depth':
                depth_keys.append(key)
                key_model_map[key] = copy.deepcopy(depth_model)
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_dim = feature_dim
        self.projector = None

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        if self.depth_keys:
            # process depth input
            for key in self.depth_keys:
                data = obs_dict[key]
                if data.ndim == 5:
                    data = data.squeeze(2)  # B,C,1,H,W -> B,C,H,W
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                
                assert data.shape[1:] == self.key_shape_map[key]
                feature = self.key_model_map[key](data)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        if self.feature_dim is not None:
            if self.projector is None:
                self.projector = nn.Sequential(
                    nn.Linear(result.shape[-1], self.feature_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.feature_dim * 2, self.feature_dim)
                )
            result = self.projector(result)

        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

class FlattenObsEncoder(nn.Module):
    """Flatten each tensor in the input dict and concatenate them.

    Skips rgb/depth keys by default — those are typically routed to a separate
    encoder (or directly to a verifier), and flattening a (3,H,W) image into an
    MLP input is rarely meaningful. Set `flatten_image_types=True` to restore
    the legacy "flatten everything" behavior.
    """

    def __init__(self, shape_meta, flatten_image_types: bool = False):
        super().__init__()
        self.shape_meta = shape_meta
        self.flatten_image_types = bool(flatten_image_types)
        if shape_meta is None:
            self.keys = None
        else:
            self.keys = [
                k for k, attr in shape_meta['obs'].items()
                if self.flatten_image_types
                or (attr or {}).get('type', 'low_dim') not in ('rgb', 'depth')
            ]

    def forward(self, obs_dict: dict) -> torch.Tensor:
        keys = self.keys or sorted(obs_dict.keys())
        batch_size = None
        parts = []
        for k in keys:
            t = obs_dict[k]
            if batch_size is None:
                batch_size = t.shape[0]
            else:
                assert t.shape[0] == batch_size, f"batch size mismatch for key {k}"
            parts.append(t.reshape(batch_size, -1))
        if not parts:
            return torch.zeros((0, 0))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def output_shape(self) -> tuple:
        """Flattened output shape (features,), counting only the keys actually
        consumed by `forward` (i.e. skipping rgb/depth unless opted in)."""
        included = set(self.keys) if self.keys is not None else None
        total = 0
        for k, attr in self.shape_meta['obs'].items():
            if included is not None and k not in included:
                continue
            shape = tuple(attr['shape'])
            total += int(torch.tensor(shape).prod().item())
        return (total,)