import timm

for v in ["convnext_tiny", "convnext_small", "convnext_base"]:
    print("Prefetching", v)
    timm.create_model(v, pretrained=True)
    
print("All Weights Cached!")