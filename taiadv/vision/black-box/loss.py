import torch.nn.functional as F

def cos(x, y, model, noiseModule):
    feat_adv = noiseModule.forward(model, x).flatten(1)
    feat_clean = model.forward_features(x).flatten(1)
    return -F.cosine_similarity(feat_adv, feat_clean).mean()

def ce(x, y, model, noiseModule):
    yp = noiseModule(model,x)    
    loss = F.cross_entropy(yp, y)
    return loss