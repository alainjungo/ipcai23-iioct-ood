import torchvision.models as mdls
from torch import nn


def get_effnet_features(net: mdls.EfficientNet, x):
    features = []
    for block in net.features:
        x = block(x)
        mean_feat = x.mean(dim=(2,3))
        features.append(mean_feat)
    return features


def get_adapted_effnet(out_classes, weights=None):
    model = mdls.efficientnet_b0(weights=weights)

    model.classifier = nn.Sequential(
        nn.Dropout(p=model.classifier[0].p, inplace=True),
        nn.Linear(model.classifier[1].in_features, out_classes),
    )
    return model
