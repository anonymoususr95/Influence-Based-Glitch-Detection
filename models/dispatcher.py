from .logitstic_regression_torch import LogisticRegressionModelTorch
from .pretrained.convnext import ConvNextModel
from .pretrained.inception import InceptionModel
from .pretrained.resnet20 import Resnet20Model
from .pretrained.vit import VisionTransformerModel

dispatcher = {
    "logistic": LogisticRegressionModelTorch,
    "resnet20": Resnet20Model,
    "inception": InceptionModel,
    "vit": VisionTransformerModel,
    "convnext": ConvNextModel,
}
