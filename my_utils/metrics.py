import torch
import fastwer
import edit_distance

# -------------------------------------------- METRICS:


def compute_all_metrics(y_true, y_pred, forbidden_chars=[]):
    metrics = compute_metrics(y_true, y_pred)
    if len(forbidden_chars) > 0:
        metrics.update(
            compute_metrics_without_forbidden_chars(
                y_true, y_pred, forbidden_chars=forbidden_chars
            )
        )
    else:
        metrics.update({"cer_wout_fc": metrics["cer"], "wer_wout_fc": metrics["wer"]})
    return metrics


def compute_metrics(y_true, y_pred):
    y_true = ["".join(s) for s in y_true]
    y_pred = ["".join(s) for s in y_pred]
    return {
        "cer": fastwer.score(y_pred, y_true, char_level=True),
        "wer": fastwer.score(y_pred, y_true),
    }


# Since the vocabularies between the source and target domains are different,
# we need metrics that tell us the performance of the source model in the target domain
# when predicting characters that are both in the source and target vocabularies


def edit_distance_without_forbidden_chars(ref, hyp, forbidden_chars=[]):
    # See: https://github.com/belambert/edit-distance

    def get_distance(opcodes, forbidden_chars=[]):
        d = 0
        for c in opcodes:
            op, ref, hyp = c
            if op == "equal":
                continue
            if any(c in forbidden_chars for c in ref):
                continue
            if any(c in forbidden_chars for c in hyp):
                continue
            d += 1
        return d

    def translate_opcode(ref, hyp, opcode):
        # Ex: ["replace", 0, 1, 0, 1]
        operation, start_ref, end_ref, start_hyp, end_hyp = opcode
        return [operation, ref[start_ref:end_ref], hyp[start_hyp:end_hyp]]

    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)

    if len(forbidden_chars) == 0:
        return sm.distance()
    else:
        opcodes = sm.get_opcodes()
        translated_opcodes = [translate_opcode(ref, hyp, code) for code in opcodes]
        return get_distance(translated_opcodes, forbidden_chars)


def compute_metrics_without_forbidden_chars(y_true, y_pred, forbidden_chars=[]):
    ed_acc = 0
    length_acc = 0
    label_acc = 0

    for t, h in zip(y_true, y_pred):
        ed = edit_distance_without_forbidden_chars(
            ref=t, hyp=h, forbidden_chars=forbidden_chars
        )
        ed_acc += ed
        length_acc += len([c for c in t if c not in forbidden_chars])
        if ed > 0:
            label_acc += 1

    return {
        "cer_wout_fc": 100.0 * ed_acc / length_acc,
        "wer_wout_fc": 100.0 * label_acc / len(y_pred),
    }


# -------------------------------------------- CTC DECODERS:


def ctc_greedy_decoder(y_pred, i2w):
    # y_pred = [seq_len, num_classes]
    # Best path
    y_pred_decoded = torch.argmax(y_pred, dim=1)
    # Merge repeated elements
    y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()
    # Convert to string; len(i2w) -> CTC-blank
    y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
    return y_pred_decoded
