from typing import Annotated, cast
from PIL import Image
from PIL.ImageFile import ImageFile
import torch
from torch._C import device
from torch._prims_common import clone_preserve_strides
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch as pt
from torchvision.transforms import transforms
from functools import lru_cache
from classes_names import CLASS_NAMES
from dataset_loader import get_data_splitted


MAIN_CLASS = "flower"
N_PROMPTS = 30
N_STEPS = 300
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

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor):
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, class_name: str, clip_model: CLIP, device: str, n_ctx: int = 4):
        super().__init__()
        self.device = device
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # randomly initialize the contextual token
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        prompt = prompt_prefix + " " + class_name + "."

        tokenized_prompt = clip.tokenize(prompt).to(self.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompt).type(dtype)[0]

        self.register_buffer("embeddings", embedding)  # SOS
        self.register_buffer("token_prefix", embedding[:1, :])  # SOS
        self.register_buffer("token_suffix", embedding[1 + n_ctx :, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompt.to(device)

    def forward(self):
        prefix = cast(pt.Tensor, self.token_prefix)
        suffix = cast(pt.Tensor, self.token_suffix)

        prompts = torch.cat(
            [
                prefix,
                self.ctx,
                suffix,
            ],
            dim=0,
        )
        return prompts

    @staticmethod
    def build(path: str, class_name: str, clip: CLIP, device: str, n_ctx: int = 4):
        to_return = PromptLearner(class_name, clip, device, n_ctx)
        to_return.load_state_dict(
            pt.load(path, weights_only=True)
        )
        return to_return.to(device)


    def to_tokens(self, clip: CLIP) -> str:
        """
        convert the matrix of learned embeddings into tokens_vectors
        (not useful for training itself, but interesting for explainability reasons)
        note: the result does not look that good, as the projection loses a ton of infos
        """
        with torch.no_grad():
            indexes = self() @ clip.token_embedding.weight.T.type(clip.dtype)
            tokens_index = np.argmax(indexes.cpu().numpy(), axis=1)
            return _tokenizer.decode(tokens_index)


class CustomCLIP(nn.Module):
    def __init__(self, class_name: str, clip_model: CLIP, device: str):
        super().__init__()
        self.prompt_learner = PromptLearner(class_name, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image: pt.Tensor) -> tuple[
        Annotated[pt.Tensor, "Prompt used"],
        Annotated[pt.Tensor, "1D tensor with score"]
    ]:
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

    def __init__(self, prompt_learned_so_far: pt.Tensor, prompts_learned_so_far_tokens: pt.Tensor, text_encoder: TextEncoder):
        super().__init__()
        # size: [num_prompt, num_tokens/prompt, embedding_size]
        
        self.text_encoder = text_encoder

        self.embeddings_learned_so_far = self.calculate_learned_so_far_embeddings(
            prompt_learned_so_far,
            prompts_learned_so_far_tokens
        )

    def forward(self, learned_prompt: pt.Tensor, learned_prompt_tokens: pt.Tensor):
        """
        learned_prompt's size: [num_tokens/prompt, embeddings_size]
        """

       
        new_embeddings = self.text_encoder(learned_prompt, learned_prompt_tokens).squeeze()
      
        if self.embeddings_learned_so_far is None == 0:
            # max distance
            return pt.Tensor(1).to(learned_prompt.device)

        new_embeddings = F.normalize(new_embeddings, dim=0)

        cos_sim = self.embeddings_learned_so_far @ new_embeddings

        min_cos_dist = 1 - torch.max(cos_sim)

        return min_cos_dist


    def calculate_learned_so_far_embeddings(self, prompt_learned_so_far: pt.Tensor, prompts_learned_so_far_tokens: pt.Tensor):
        embeddings: list[pt.Tensor] = []

        with torch.no_grad():
            for (prompt, token) in zip(prompt_learned_so_far, prompts_learned_so_far_tokens):
                embedding = self.text_encoder(prompt, token)
                embeddings.append(embedding)

        if len(embeddings) == 0:
            return None

        embeddings_stacked = pt.stack(embeddings)

        embeddings_stacked = F.normalize(embeddings_stacked, dim=1)

        return embeddings_stacked.detach()



class Util():

    def __init__(self):
        self.device = "cuda"

        train_dataset = get_data_splitted("RN50", self.device)[0]


        self.images = pt.vstack([x[0].unsqueeze(dim=0) for x in train_dataset]).to(self.device)
        """
        shape: n_images x 3 x 244 x 244
        """
        # reduce for quick test
        # self.images = self.images[3:40]

    def build_model(self, prompt_learned_so_far: pt.Tensor, prompts_learned_so_far_tokens: pt.Tensor):

        self.raw_clip = load_clip_to_cpu().to(self.device)
        
        self.custom_clip_model = CustomCLIP( MAIN_CLASS, self.raw_clip, self.device)

        for name, param in self.custom_clip_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)


        self.custom_clip_model.to(self.device)
        self.optim = pt.optim.SGD(self.custom_clip_model.prompt_learner.parameters(), maximize=True)
        self.raw_clip.to(self.device)
        self.images_weight = ImageWeights(self.images.shape[0]).to(self.device)
        self.prompt_distance = PromptDistance(
            prompt_learned_so_far.type(self.custom_clip_model.dtype),
            prompts_learned_so_far_tokens.type(self.custom_clip_model.dtype),
            self.custom_clip_model.text_encoder
        ).to(self.device)


    def forward_backward(self):
        
        self.optim.zero_grad()
        prompt: pt.Tensor
        images_similarity: pt.Tensor
        prompt, images_similarity = self.custom_clip_model(self.images)

        # the similarity weighted by the images, weighted by the selected images
        weights: pt.Tensor = self.images_weight()


        similarity_factor = torch.dot(images_similarity.squeeze().type(pt.float), weights)
        generalization_factor = -torch.sum(weights * torch.log(weights + 1e-12))

        prompt_tokens = self.custom_clip_model.tokenized_prompts
        diversity_factor: pt.Tensor = self.prompt_distance(prompt, prompt_tokens)


        loss = (
            1.0 * similarity_factor +
            30 * generalization_factor +
            50 * diversity_factor
        )

        # print(f"final loss: {loss.squeeze()}")
        loss.backward()
        self.optim.step()

    def inference(self, image: pt.Tensor, options: list[PromptLearner]) -> int:
        best_so_far: int = -1
        best_score: float = 0
        print("="*50)
        with torch.no_grad():
            for i, option in enumerate(options):

                self.custom_clip_model.prompt_learner = option
                self.custom_clip_model.tokenized_prompts = option.tokenized_prompts

                _, score = self.custom_clip_model.forward(image)

                print(f"label {i} has probability {score[0]}")
                score_float = float(score[0])

                if score_float > best_score:
                    best_score = score_float
                    best_so_far = i

        return best_so_far


    def get_learned_prompt(self) -> pt.Tensor:
        return self.custom_clip_model.prompt_learner()

    def get_tokenized_prompt(self) -> pt.Tensor:
        return self.custom_clip_model.prompt_learner.tokenized_prompts


    def prompt_to_embedding(self, prompt: str):
        token = clip.tokenize(prompt).to(self.device)
        embedding: pt.Tensor = self.raw_clip.token_embedding(token)
        return embedding, token.unsqueeze(dim=0)

    def save_prompt(self, path: str):
        pt.save(
            self.custom_clip_model.prompt_learner.state_dict(),
            path
        )


    def load_first_stage_prompts(self):
        to_return = list[PromptLearner]()
        for i in range(N_PROMPTS):
            x = PromptLearner.build(
                f"./learned_prompts/prompt_{i}",
                MAIN_CLASS,
                self.raw_clip,
                self.device
            )

            to_return.append(x)
        return to_return

    @lru_cache(maxsize=100)
    def load_second_stage_prompts(self, prompt_index: int):
        to_return = list[PromptLearner]()
        for item in CLASS_NAMES:
            x = PromptLearner.build(
                f"./learned_prompts/prompt_{prompt_index}",
                item,
                self.raw_clip,
                self.device
            )

            to_return.append(x)

        return to_return


def main():



    u = Util()
    u.build_model(pt.Tensor(), pt.Tensor())

    prompts_learned_so_far, prompts_learned_so_far_tokens  = u.prompt_to_embedding(f"a picture of a {MAIN_CLASS}")

    for i in range(N_PROMPTS):


        u.build_model(prompts_learned_so_far, prompts_learned_so_far_tokens)
        for _ in range(N_STEPS):
            u.forward_backward()

        print(f"prompt learned: {u.custom_clip_model.prompt_learner.to_tokens(u.raw_clip)}")
        prompts_learned_so_far = pt.vstack([
            prompts_learned_so_far,
            u.get_learned_prompt().unsqueeze(dim=0)
        ])

        prompts_learned_so_far_tokens = pt.vstack([
            prompts_learned_so_far_tokens,
            u.get_tokenized_prompt().unsqueeze(dim=0)
        ])

        u.save_prompt(f"./learned_prompts/prompt_{i}")


if __name__ == "__main__":
    main()



