import json
import os

import torch
from torch.utils.data import Dataset

from my_utils.data_preprocessing import (
    preprocess_image_from_file,
    preprocess_transcript,
)

################################################################################################ Single-source:


class CTCDataset(Dataset):
    def __init__(
        self,
        name,
        img_folder_path,
        transcripts_file,
        train=True,
        da_train=False,
        width_reduction=2,
    ):
        self.name = name
        self.train = train
        self.da_train = da_train
        self.width_reduction = width_reduction

        # Get image paths and transcripts
        self.X, self.Y = self.get_images_and_transcripts(
            img_folder_path, transcripts_file
        )

        # Check and retrieve vocabulary
        vocab_name = "w2i.json"
        vocab_folder = os.path.join(os.path.join("data", self.name.lower()), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(transcripts_file)

        # Preprocess transcripts after retrieving vocabulary
        self.Y = self.preprocess_all_transcripts(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.da_train:
            # Domain Adaptation setting
            x = preprocess_image_from_file(self.X[idx])
            return x

        else:
            # CTC Training setting
            x = preprocess_image_from_file(self.X[idx])
            y = torch.tensor(self.Y[idx])

            if self.train:
                # x.shape = [channels, height, width]
                return x, x.shape[2] // self.width_reduction, y, len(y)

            return x, y

    def get_images_and_transcripts(self, img_folder_path, transcripts_file):
        images = []
        transcripts = []
        # TODO:
        # Lo que al final devolvía era una lista con los paths de las imágenes
        # y otra con las transcripciones de cada imagen porque tenía un fichero
        # que era word_transcriptions.txt donde cada fila era:
        # {nombre de la imagen}\t{transcripción}
        # Todo esto hay que adaptarlo al formato de los datasets que usemos

        return images, transcripts

    def check_and_retrieve_vocabulary(self, transcripts):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary(transcripts)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self, transcripts):
        vocab = []

        # TODO:
        # Retrive vocab!
        vocab = sorted(vocab)

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def preprocess_all_transcripts(self, transcripts):
        pre_transcripts = []
        for t in transcripts:
            pre_transcripts.append(preprocess_transcript(t, self.w2i))
        return pre_transcripts
