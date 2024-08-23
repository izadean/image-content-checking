import torch
import torchvision.transforms as transforms
from ts.torch_handler.vision_handler import VisionHandler
from sentence_transformers import SentenceTransformer


class SentenceImageSimilarityHandler(VisionHandler):
    def __init__(self):
        super().__init__()
        self.image_transform = None
        self.initialized = False
        self.sentence_transformer = None

    def initialize(self, context):
        super().initialize(context)

        self.sentence_transformer = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0", revision=None)
        self.sentence_transformer.to(self.device)
        self.sentence_transformer.eval()

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.initialized = True

    def preprocess(self, data):
        images = super().preprocess(data)
        sentences = []

        for row in data:
            sentence = row.get("sentence")
            if sentence is not None:
                sentences.append(sentence)
        sentence_embeddings = self.sentence_transformer.embed(sentences)
        return images, sentence_embeddings

    def inference(self, data, **kwargs):
        images, sentence_embeddings = data
        with torch.no_grad():
            return self.model(images, sentence_embeddings)

    def postprocess(self, inference_output):
        return inference_output.cpu().tolist()

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)