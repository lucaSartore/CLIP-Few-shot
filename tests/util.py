def load_clip_to_cpu():
    model_url = clip.clip._MODELS["RN50"]

    model_path = clip.clip._download(model_url, "./models")
    model_pt = pt.jit.load(model_path, map_location="cpu").eval()
    model = clip.clip.build_model(model_pt.state_dict())
    return model
