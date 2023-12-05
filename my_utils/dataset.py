import json
import os

import torch
from torch.utils.data import Dataset

from data.config import DS_CONFIG, check_iam_word_images
from data.cvl.writers_config import writers as cvl_writers
from data.iam.writers_config import writers as iam_writers
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
        preprocess=True,
        lowercase=False,
        train=True,
        da_train=False,
        width_reduction=2,
    ):
        self.name = name
        self.lowercase = lowercase
        self.train = train
        self.da_train = da_train
        self.width_reduction = width_reduction
        self.resize = False if "synthetic_words" in self.name else True

        # Get image paths and transcripts
        if self.name in iam_writers.keys() or self.name in cvl_writers.keys():
            # IAM or CVL writer setting
            self.X, self.Y = self.get_writer_images_and_transcripts(
                img_folder_path, transcripts_file
            )
        else:
            self.X, self.Y = self.get_images_and_transcripts(
                img_folder_path, transcripts_file
            )

        # Check and retrieve vocabulary
        vocab_name = "w2i"
        vocab_name += "_lowercase" if self.lowercase else ""
        vocab_name += ".json"
        if self.name in iam_writers.keys() or self.name in cvl_writers.keys():
            # IAM or CVL writer setting
            vocab_folder = os.path.join(
                os.path.join("data", self.name.split("_")[0].lower()), "vocab"
            )
        else:
            vocab_folder = os.path.join(
                os.path.join("data", self.name.lower()), "vocab"
            )
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(transcripts_file)

        # Preprocess transcripts after retrieving vocabulary
        if preprocess:
            self.Y = self.preprocess_all_transcripts(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.da_train:
            # Domain Adaptation setting
            x = preprocess_image_from_file(self.X[idx], resize=self.resize)
            return x

        else:
            # CTC Training setting
            x = preprocess_image_from_file(self.X[idx], resize=self.resize)
            y = torch.tensor(self.Y[idx])

            if self.train:
                # x.shape = [channels, height, width]
                return x, x.shape[2] // self.width_reduction, y, len(y)

            return x, y

    def get_images_and_transcripts(self, img_folder_path, transcripts_file):
        images = [
            filename
            for filename in os.listdir(img_folder_path)
            if filename.endswith(".png") and not filename.startswith(".")
        ]
        # Check IAM word images
        if "iam" in self.name:
            images = check_iam_word_images(images)

        final_images = []
        transcripts = []
        with open(transcripts_file, "r") as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            image_name, transcript = line.split(" ", 1)
            image_name = image_name + ".png"
            if image_name in images:
                transcripts.append(transcript)
                final_images.append(os.path.join(img_folder_path, image_name))

        return final_images, transcripts

    def get_writer_images_and_transcripts(self, img_folder_path, transcripts_file):
        # The img_folder_path in a writer is the actual data split
        # A .txt file containing the word ids and transcriptions
        # The transcripts_file is None in this case

        assert transcripts_file is None, "Transcripts file should be None in this case"

        images = []
        transcripts = []

        with open(img_folder_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                # Line example: train 0609-4-4-8-a a
                img_folder, img_name, transcript = line.split(" ", 2)
                # Check IAM word images
                if "iam" in self.name:
                    if len(check_iam_word_images([img_name + ".png"])) == 0:
                        continue
                img_name = os.path.join(
                    "data",
                    self.name.split("_")[0].lower(),
                    "words",
                    img_folder,
                    img_name + ".png",
                )
                images.append(img_name)
                transcripts.append(transcript)
        return images, transcripts

    def check_and_retrieve_vocabulary(self, transcripts_file):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary(transcripts_file)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self, transcripts_file):
        vocab = []

        if not isinstance(transcripts_file, list):
            transcripts_file = [transcripts_file]

        for f in transcripts_file:
            with open(f, "r") as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                _, transcript = line.split(" ", 1)
                if self.lowercase:
                    transcript = transcript.lower()
                for char in transcript:
                    if char not in vocab:
                        vocab.append(char)
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
            if self.lowercase:
                t = t.lower()
            pre_transcripts.append(preprocess_transcript(t, self.w2i))
        return pre_transcripts


################################################################################################ Multi-source:


class TestMultiSourceCTCDataset(CTCDataset):
    def __init__(
        self,
        name,
        img_folder_paths,
        transcripts_files,
        w2i,
        i2w,
        lowercase=False,
        width_reduction=2,
    ):
        self.name = name
        self.lowercase = lowercase
        self.train = False
        self.da_train = False
        self.width_reduction = width_reduction

        # Get image paths and transcripts
        self.X, self.Y = self.merge_images_and_transcripts(
            img_folder_paths, transcripts_files
        )

        # Set vocabulary
        self.w2i = w2i
        self.i2w = i2w

        # Preprocess transcripts after retrieving vocabulary
        self.Y = self.preprocess_all_transcripts(self.Y)

    def merge_images_and_transcripts(self, img_folder_paths, transcripts_files):
        X, Y = [], []
        for img_folder_path, transcripts_file in zip(
            img_folder_paths, transcripts_files
        ):
            xx, yy = self.get_images_and_transcripts(img_folder_path, transcripts_file)
            X.extend(xx)
            Y.extend(yy)
        return X, Y


class TrainMultiSourceCTCDataset(Dataset):
    def __init__(
        self,
        name,
        img_folder_paths,
        transcripts_files,
        lowercase=False,
        da_train=False,
        width_reduction=2,
    ):
        self.name = name
        self.lowercase = lowercase
        self.da_train = da_train
        self.width_reduction = width_reduction
        self.prepare(img_folder_paths, transcripts_files)

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        items = [dataset[idx % len(dataset)] for dataset in self.datasets]
        return items

    def prepare(self, img_folder_paths, transcripts_files):
        # 1) Create datasets
        self.datasets = self.prepare_datasets(img_folder_paths, transcripts_files)

        # 2) Check and retrieve vocabulary
        vocab_name = "w2i"
        vocab_name += "_lowercase" if self.lowercase else ""
        vocab_name += ".json"
        vocab_folder = os.path.join(os.path.join("data", self.name.lower()), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(self.datasets)
        self.set_vocabularies(self.datasets)

        # 3) Preprocess transcripts after retrieving vocabulary
        self.preprocess_all_transcripts()

    def prepare_datasets(self, img_folder_paths, transcripts_files):
        datasets = []
        ds_names = self.name.split("_")  # Ex.: self.name = "iam_washington"
        for ds_name, img_folder, t_file in zip(
            ds_names, img_folder_paths, transcripts_files
        ):
            datasets.append(
                CTCDataset(
                    name=ds_name,
                    img_folder_path=img_folder,
                    transcripts_file=t_file,
                    preprocess=False,
                    lowercase=self.lowercase,
                    train=not (self.da_train),
                    da_train=self.da_train,
                    width_reduction=self.width_reduction,
                )
            )
        return datasets

    def check_and_retrieve_vocabulary(self, datasets):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary(datasets)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self, datasets):
        # Get common vocabulary
        vocab = set()
        for ds in datasets:
            vocab.update(ds.w2i.keys())
        vocab = sorted(vocab)

        # Create dictionaries
        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def set_vocabularies(self, datasets):
        for ds in datasets:
            ds.w2i = self.w2i
            ds.i2w = self.i2w

    def preprocess_all_transcripts(self):
        for ds in self.datasets:
            ds.Y = ds.preprocess_all_transcripts(ds.Y)


if __name__ == "__main__":
    import argparse
    import random

    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image

    from my_utils.augmentations import AugmentStage
    from my_utils.data_preprocessing import (
        ctc_batch_preparation,
        multisource_ctc_batch_preparation,
        multisource_pad_batch_images,
        pad_batch_images,
    )

    CHECK_DIR = "check"
    if not os.path.isdir(CHECK_DIR):
        os.mkdir(CHECK_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_name",
        type=str,
        default="iam",
        choices=["iam", "cvl", "washington", "esposalles"],
        help="Dataset name",
    )
    args = parser.parse_args()

    ################################################# CTC Setting:

    train_ds = CTCDataset(
        name=args.ds_name,
        img_folder_path=DS_CONFIG[args.ds_name]["train"],
        transcripts_file=DS_CONFIG[args.ds_name]["transcripts"],
    )
    val_ds = CTCDataset(
        name=args.ds_name,
        img_folder_path=DS_CONFIG[args.ds_name]["val"],
        transcripts_file=DS_CONFIG[args.ds_name]["transcripts"],
        train=False,
    )
    test_ds = CTCDataset(
        name=args.ds_name,
        img_folder_path=DS_CONFIG[args.ds_name]["test"],
        transcripts_file=DS_CONFIG[args.ds_name]["transcripts"],
        train=False,
    )

    print("Vocabulary size:", len(train_ds.w2i))
    assert train_ds.w2i == test_ds.w2i, "Vocabulary mismatch"

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=ctc_batch_preparation
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Train dataset size: {len(train_ds)}")
    for x, x_len, y, y_len in train_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("\tx_len:", x_len.dtype)
        print("\ty:", y.dtype)
        print("\ty_len:", y_len.dtype)
        print("Shapes:")
        print("\tx:", x.shape)
        print("\tx_len:", x_len.shape)
        print("\ty:", y.shape)
        print("\ty_len:", y_len.shape)

        # Save batch images
        save_image(
            make_grid(x, nrow=4), f"{CHECK_DIR}/{args.ds_name}_x_train_batch.jpg"
        )

        # See first sample
        print(f"Shape with padding: {y[0].shape}; Original shape: {y_len[0].numpy()}")
        print("Transcription:", [train_ds.i2w[i.item()] for i in y[0]])
        save_image(x[0], f"{CHECK_DIR}/{args.ds_name}_x0_train_batch.jpg")

        break

    print(f"Val dataset size: {len(val_ds)}")
    for x, y in val_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("\ty:", y.dtype)
        print("Shapes:")
        print("\tx:", x.shape)
        print("\ty:", y.shape)

        # See first sample
        print("Transcription:", [val_ds.i2w[i.item()] for i in y[0]])
        save_image(x[0], f"{CHECK_DIR}/{args.ds_name}_x0_val_batch.jpg")

        break

    print(f"Test dataset size: {len(test_ds)}")
    for x, y in test_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("\ty:", y.dtype)
        print("Shapes:")
        print("\tx:", x.shape)
        print("\ty:", y.shape)

        # See first sample
        print("Transcription:", [test_ds.i2w[i.item()] for i in y[0]])
        save_image(x[0], f"{CHECK_DIR}/{args.ds_name}_x0_test_batch.jpg")

        break

    ################################################# DA Setting:

    augment = AugmentStage()

    da_train_ds = CTCDataset(
        name=args.ds_name,
        img_folder_path=DS_CONFIG[args.ds_name]["train"],
        transcripts_file=DS_CONFIG[args.ds_name]["transcripts"],
        train=False,
        da_train=True,
    )
    da_train_loader = DataLoader(
        da_train_ds, batch_size=16, shuffle=True, collate_fn=pad_batch_images
    )

    print(f"Train dataset size: {len(da_train_ds)}")
    for x in da_train_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("Shapes:")
        print("\tx:", x.shape)

        x_aug = augment(x)

        # Save batch images
        save_image(
            make_grid(x, nrow=4), f"{CHECK_DIR}/{args.ds_name}_x_da_train_batch.jpg"
        )

        # See first sample
        save_image(x[0], f"{CHECK_DIR}/{args.ds_name}_x0_da_train_batch.jpg")

        # Save batch images
        save_image(
            make_grid(x_aug, nrow=4),
            f"{CHECK_DIR}/{args.ds_name}_x_da_train_batch_aug.jpg",
        )

        # See first sample
        save_image(x_aug[0], f"{CHECK_DIR}/{args.ds_name}_x0_da_train_batch_aug.jpg")

        break

    #################################################################################################
    #################################################################################################

    ################################################# Multiple sources CTC Setting:

    ds_names = [args.ds_name] + random.choices(
        [
            ds_name
            for ds_name in ["iam", "cvl", "washington", "esposalles"]
            if ds_name != args.ds_name
        ],
        k=2,
    )

    print(f"Using datasets: {ds_names}")

    train_ds_multi = TrainMultiSourceCTCDataset(
        name="_".join(ds_names),
        img_folder_paths=[DS_CONFIG[ds]["train"] for ds in ds_names],
        transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_names],
    )
    test_ds_multi = TestMultiSourceCTCDataset(
        name="_".join(ds_names),
        img_folder_paths=[DS_CONFIG[ds]["test"] for ds in ds_names],
        transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_names],
        i2w=train_ds_multi.i2w,
        w2i=train_ds_multi.w2i,
    )

    print("Vocabulary size:", len(train_ds_multi.w2i))
    assert train_ds_multi.w2i == test_ds_multi.w2i, "Vocabulary mismatch"
    for ds in train_ds_multi.datasets:
        assert ds.w2i == train_ds_multi.w2i, "Vocabulary mismatch"

    train_multi_loader = DataLoader(
        train_ds_multi,
        batch_size=16 // len(ds_names),
        shuffle=True,
        collate_fn=multisource_ctc_batch_preparation,
    )
    test_multi_loader = DataLoader(test_ds_multi, batch_size=1, shuffle=False)

    print(f"Train dataset size: {len(train_ds_multi)}")
    for x, x_len, y, y_len in train_multi_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("\tx_len:", x_len.dtype)
        print("\ty:", y.dtype)
        print("\ty_len:", y_len.dtype)
        print("Shapes:")
        print("\tx:", x.shape)
        print("\tx_len:", x_len.shape)
        print("\ty:", y.shape)
        print("\ty_len:", y_len.shape)

        # Save batch images
        save_image(
            make_grid(x, nrow=4), f"{CHECK_DIR}/{'-'.join(ds_names)}_x_train_batch.jpg"
        )

        # See first sample
        print(f"Shape with padding: {y[0].shape}; Original shape: {y_len[0].numpy()}")
        print("Transcription:", [train_ds_multi.i2w[i.item()] for i in y[0]])
        save_image(x[0], f"{CHECK_DIR}/{'-'.join(ds_names)}_x0_train_batch.jpg")

        break

    print(f"Test dataset size: {len(test_ds_multi)}")
    for x, y in test_multi_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("\ty:", y.dtype)
        print("Shapes:")
        print("\tx:", x.shape)
        print("\ty:", y.shape)

        # See first sample
        print("Transcription:", [test_ds_multi.i2w[i.item()] for i in y[0]])
        save_image(x[0], f"{CHECK_DIR}/{'-'.join(ds_names)}_x0_test_batch.jpg")

        break

    ################################################# Multiple sources DA Setting:

    da_train_ds_multi = TrainMultiSourceCTCDataset(
        name="_".join(ds_names),
        img_folder_paths=[DS_CONFIG[ds]["train"] for ds in ds_names],
        transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_names],
        da_train=True,
    )
    da_train_multi_loader = DataLoader(
        da_train_ds_multi,
        batch_size=16 // len(ds_names),
        shuffle=True,
        collate_fn=multisource_pad_batch_images,
    )

    print(f"Train dataset size: {len(da_train_ds_multi)}")
    for x in da_train_multi_loader:
        print("Types:")
        print("\tx:", x.dtype)
        print("Shapes:")
        print("\tx:", x.shape)

        # Save batch images
        save_image(
            make_grid(x, nrow=4),
            f"{CHECK_DIR}/{'-'.join(ds_names)}_x_da_train_batch.jpg",
        )

        # See first sample
        save_image(x[0], f"{CHECK_DIR}/{'-'.join(ds_names)}_x0_da_train_batch.jpg")

        break
