from .convnext_base import convnext_base
from .shared_convnext import shared_convnext
from .slice_param_convnext import sliceparamconvnext
from .template_mixing_convnext import templatemixingconvnext
from .hypernet_convnext import hyperconvnext
from .depthwise_convnext import depthwiseconvnext

from .channel_vit_adapt import channelvit_adapt
from .dichavit import dichavit
from .vit_adapt import vit_adapt
#avantika added this dont want to make a whole new copy shouldnt affect anything else
from .vit import vit_tiny, vit_small, vit_base
##
from .depthwise_vit import depthwisevit_adapt
from .hyper_vit import hypervit_adapt
from .template_mixing_vit import templatemixingvit

## for new model, add the new model in _forward_model(), trainer.py
