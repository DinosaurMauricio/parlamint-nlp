import torch


def collate_fn(batch, pad_token_id):
    input_tokens = [e["tokenized_text"]["input_ids"] for e in batch]
    input_tokens_maxlen = max([len(t.squeeze()) for t in input_tokens])

    input_ids_list = []
    attention_mask_list = []
    labels = []

    for sample in batch:
        pad_len = input_tokens_maxlen - len(
            sample["tokenized_text"]["input_ids"].squeeze()
        )
        input_ids_list.append(
            sample["tokenized_text"]["input_ids"].squeeze().tolist()
            + pad_len * [pad_token_id]
        )  # Pad token is 1
        attention_mask_list.append(
            sample["tokenized_text"]["attention_mask"].squeeze().tolist()
            + pad_len * [0]
        )
        labels.append(sample["label"])

    inputs = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }
    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, labels
