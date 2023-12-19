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
        transcripts_folder,
        img_folder,
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
            img_folder_path, img_folder, transcripts_folder
        )

        # Check and retrieve vocabulary
        vocab_name = "w2i.json"
        vocab_folder = os.path.join(os.path.join("data", self.name.lower()), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(transcripts_folder)

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

    def get_images_and_transcripts(self, img_dat_file_path, img_directory, transcripts_directory):
        images = []
        transcripts = []
        
        # En el caso de OMR, las imágenes están en un directorio y las transcripciones
        # en otro. Las transcripciones están en formato agnóstico.
        # Cada archivo del directorio es la transcripción de la imagen con el mismo
        # nombre pero en el directorio de imágenes.
        # El nombre de las transcripciones es:
        # {nombre de la imagen con su extensión}.txt
        # Y el formato de cada archivo es:
        # {transcripción}
        
        # Leer el archivo .dat para obtener los nombres de los archivos de imagen
        with open(img_dat_file_path, 'r') as file:
            img_files = file.read().splitlines()

        for img_file in img_files:
            img_path = os.path.join(img_directory, img_file)

            # El nombre del archivo de transcripción incluye la extensión completa de la imagen
            transcript_file = img_file + '.txt'
            transcript_path = os.path.join(transcripts_directory, transcript_file)

            if os.path.exists(img_path) and os.path.exists(transcript_path):
                images.append(img_path)
                with open(transcript_path, 'r') as file:
                    transcripts.append(file.read().split())

        return images, transcripts

    def check_and_retrieve_vocabulary(self, transcripts_dir):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            transcripts = self.read_transcripts(transcripts_dir)
            w2i, i2w = self.make_vocabulary(transcripts)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w
    
    def read_transcripts(self, transcripts_dir):
        transcripts = []
        for transcript_file in os.listdir(transcripts_dir):
            with open(os.path.join(transcripts_dir, transcript_file), 'r') as file:
                transcripts.append(file.read())
        return transcripts

    def make_vocabulary(self, transcripts):
        vocab = set()  # Usamos un conjunto para evitar duplicados

        for transcript in transcripts:
            # Dividir cada transcripción en palabras/tokens
            words = transcript.split()  # Esto divide el texto por espacios
            vocab.update(words)  # Añade las palabras al conjunto de vocabulario
        
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