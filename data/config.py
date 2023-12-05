from data.cvl.writers_config import writers as cvl_writers
from data.iam.writers_config import writers as iam_writers

DS_CONFIG = {
    "iam": {
        "train": "data/iam/words/trainset",
        # NOTE: There are 2 validation sets
        # We keep just one (val1)
        # "val1": "data/iam/words/validationset1",
        # "val2": "data/iam/words/validationset2",
        "val": "data/iam/words/validationset1",
        "test": "data/iam/words/testset",
        "transcripts": "data/iam/word_transcriptions.txt",
    },
    "washington": {
        # NOTE: There are 4 cross-validation splits
        # We keep just one (cv1)
        # "cv1_train": "data/washington/words/cv1/train",
        # "cv1_valid": "data/washington/words/cv1/valid",
        # "cv1_test": "data/washington/words/cv1/test",
        # "cv2_train": "data/washington/words/cv2/train",
        # "cv2_valid": "data/washington/words/cv2/valid",
        # "cv2_test": "data/washington/words/cv2/test",
        # "cv3_train": "data/washington/words/cv3/train",
        # "cv3_valid": "data/washington/words/cv3/valid",
        # "cv3_test": "data/washington/words/cv3/test",
        # "cv4_train": "data/washington/words/cv4/train",
        # "cv4_valid": "data/washington/words/cv4/valid",
        # "cv4_test": "data/washington/words/cv4/test",
        "train": "data/washington/words/cv1/train",
        "val": "data/washington/words/cv1/valid",
        "test": "data/washington/words/cv1/test",
        "transcripts": "data/washington/word_transcriptions.txt",
    },
    "esposalles": {
        # NOTE: The val split was randomly created using a 10% of the training set
        "train": "data/esposalles/words/train",
        "val": "data/esposalles/words/val",
        "test": "data/esposalles/words/test",
        "transcripts": "data/esposalles/word_transcriptions.txt",
    },
    "cvl": {
        # NOTE: The val split was randomly created using a 10% of the training set
        "train": "data/cvl/words/train",
        "val": "data/cvl/words/val",
        "test": "data/cvl/words/test",
        "transcripts": "data/cvl/word_transcriptions.txt",
    },
    "synthetic_words_real": {
        # NOTE: The actual test are the real datasets
        "val": "data/synthetic_words_real/words/val",
        "train_transcripts": "data/synthetic_words_real/train_word_transcriptions.txt",
        "transcripts": "data/synthetic_words_real/val_word_transcriptions.txt",  # Val transcripts
    },
    "synthetic_words_random": {
        # NOTE: The actual test are the real datasets
        "train": 30000,  # 30000 samples per epoch
        "val": "data/synthetic_words_random/words/val",  # Fix the validation set to 3000 samples
        "train_transcripts": "data/synthetic_words_random/train_word_transcriptions.json",
        "transcripts": "data/synthetic_words_random/val_word_transcriptions.txt",  # Val transcripts
    },
    "synthetic_words_wiki": {
        # NOTE: The actual test are the real datasets
        "val": "data/synthetic_words_wiki/words/val",
        "train_transcripts": "data/synthetic_words_wiki/splits/train.txt",
        "transcripts": "data/synthetic_words_wiki/splits/valid.txt",  # Val transcripts
    },
}
# Merge the writers config from IAM and CVL with the DS_CONFIG
DS_CONFIG = {**DS_CONFIG, **iam_writers, **cvl_writers}


def check_iam_word_images(images):
    # Problems with the following images:
    # - data/iam/words/trainset/r06-022-03-05.png
    # - data/iam/words/trainset/a01-117-05-02.png
    # They are not properly read by PIL.Image.open()
    # We believe that the images are corrupted, so we remove them
    if "r06-022-03-05.png" in images:
        images.remove("r06-022-03-05.png")
    if "a01-117-05-02.png" in images:
        images.remove("a01-117-05-02.png")
    # Remove also images with w \\ 2 < len(label)
    with open("data/iam/exceptions.txt", "r") as f:
        exceptions = f.read().splitlines()
    for exception in exceptions:
        if exception in images:
            images.remove(exception)
    return images
