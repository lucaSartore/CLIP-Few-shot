from PIL import Image
from PIL.ImageFile import ImageFile
import torch
from torch._prims_common import clone_preserve_strides
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch as pt
from torchvision.transforms import transforms

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    model_url = clip.clip._MODELS["RN50"]

    model_path = clip.clip._download(model_url, "./models")
    model_pt = pt.jit.load(model_path, map_location="cpu").eval()
    model = clip.clip.build_model(model_pt.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classname: str, clip_model: CLIP):
        super().__init__()
        n_ctx = 4
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # print("Initializing a generic context")
        ctx_vectors = torch.empty(1,n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classname = classname.replace("_", " ")
        name_len = len(_tokenizer.encode(classname))
        prompt = prompt_prefix + " " + classname + "."

        tokenized_prompts = clip.tokenize(prompt)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("embeddings", embedding)  # SOS
        self.register_buffer("token_prefix", embedding[:,:1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:,1 + n_ctx :, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_len = name_len

    def forward(self):

        ctx = self.ctx

        prefix = self.token_prefix
        suffix = self.token_suffix


        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim) #type: ignore
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts

    @staticmethod
    def build(path: str, class_name: str):
        to_return = PromptLearner()
        pt.load()


    def to_tokens(self, clip: CLIP) -> str:
        """
        convert the matrix of learned embeddings into tokens_vectors
        (not useful for training itself, but interesting for explainability reasons)
        note: the result does not look that good, as the projection loses a ton of infos
        """
        with torch.no_grad():
            indexes = self()[0] @ clip.token_embedding.weight.T.type(clip.dtype)
            tokens_index = np.argmax(indexes.cpu().numpy(), axis=1)
            return _tokenizer.decode(tokens_index)


class CustomCLIP(nn.Module):
    def __init__(self, classname: str, clip_model: CLIP):
        super().__init__()
        self.prompt_learner = PromptLearner(classname, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return prompts, logits


def load_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image = transforms.Resize((224, 224))(image)
    tensor = transforms.ToTensor()(image)
    tensor = tensor.unsqueeze(0)
    return tensor


class ImageWeights(nn.Module):
    """
    class that represent a set of weight that each of the images has.
    """

    def __init__(self, num_images: int):
        super().__init__()
        weights = torch.empty(num_images)
        nn.init.normal_(weights, std=0.02)
        self.weights = nn.Parameter(weights) 

    def forward(self):
        return pt.nn.functional.softmax(self.weights, dim=0)

class PromptDistance(nn.Module):
    """
    return the distance form a prompt to the closest one (to maximize dissimilarity between prompts)
    """

    def __init__(self, prompt_learned_so_far: pt.Tensor):
        super().__init__()
        # size: [num_prompt, num_tokens/prompt, embedding_size]
        self.prompt_learned_so_far = prompt_learned_so_far.detach()

    def forward(self, learned_prompt: pt.Tensor):
        """
        learned_prompt's size: [num_tokens/prompt, embeddings_size]
        """

        # calculating euclidean distance
        x = self.prompt_learned_so_far - learned_prompt
        x = x ** 2
        x = pt.sum(x, dim=(1,2))
        x = x ** 0.5

        # picking the minimum value
        x = pt.min(x)

        return x


class Util():

    def __init__(self):
        self.device = "cuda"


        images = []
        for i in range(24,80):
            try:
                image = load_image(
                    f"C:\\Users\\lucas\\Desktop\\ml_project\\CoOp\\data\\oxford_pets\\images\\Abyssinian_{i}.jpg"
                ).to(self.device)
                if image.shape != torch.Size([1,3,224,224]):
                    continue
                images.append(image)
            except:
                pass

        self.image = pt.vstack(images)

    def build_model(self, prompt_learned_so_far: pt.Tensor):

        self.raw_clip = load_clip_to_cpu()
        
        classname = "cat"
        self.custom_clip_model = CustomCLIP( classname, self.raw_clip)

        for name, param in self.custom_clip_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)


        self.custom_clip_model.to(self.device)
        self.optim = pt.optim.SGD(self.custom_clip_model.prompt_learner.parameters(), maximize=True)
        self.raw_clip.to(self.device)
        self.images_weight = ImageWeights(self.image.shape[0]).to(self.device)
        self.prompt_distance = PromptDistance(prompt_learned_so_far).to(self.device)


    def forward_backward(self):
        
        self.optim.zero_grad()
        prompts: pt.Tensor
        images_similarity: pt.Tensor
        prompts, images_similarity = self.custom_clip_model(self.image)

        # the similarity weighted by the images, weighted by the selected images
        weights: pt.Tensor = self.images_weight()


        similarity_factor = torch.dot(images_similarity.squeeze().type(pt.float), weights)
        generalization_factor = -torch.sum(weights * torch.log(weights + 1e-12))
        diversity_factor: pt.Tensor = self.prompt_distance(prompts)

        # print("loss factors: ")
        # print(similarity_factor.squeeze())
        # print(generalization_factor.squeeze())
        # print(diversity_factor.squeeze())

        loss = (
            1.0 * similarity_factor +
            30 * generalization_factor +
            50 * diversity_factor
        )

        # print(f"final loss: {loss.squeeze()}")
        loss.backward()
        self.optim.step()


    def get_learned_prompt(self) -> pt.Tensor:
        return self.custom_clip_model.prompt_learner()


    def prompt_to_embedding(self, prompt: str):
        token = clip.tokenize(prompt).to(self.device)
        embedding: pt.Tensor = self.raw_clip.token_embedding(token)
        return embedding

    def save_prompt(self, path: str):
        pt.save(
            self.custom_clip_model.prompt_learner.state_dict(),
            path
        )

def main():


    N_PROMPTS = 10
    N_STEPS = 300

    u = Util()
    u.build_model(pt.Tensor())

    prompts_learned_so_far = u.prompt_to_embedding("a picture of a cat")

    for i in range(N_PROMPTS):


        u.build_model(prompts_learned_so_far)
        for _ in range(N_STEPS):
            u.forward_backward()

        print(f"prompt learned: {u.custom_clip_model.prompt_learner.to_tokens(u.raw_clip)}")
        prompts_learned_so_far = pt.vstack([
            prompts_learned_so_far,
            u.get_learned_prompt()
        ])
        u.save_prompt(f"./learned_prompts/prompt_{i}")


if __name__ == "__main__":
    main()



