import torch.nn as nn


class ClassificationParlamint(nn.Module):

    def __init__(self, encoder, num_classes, unfreeze=False, dropout=0.1):
        super().__init__()

        self.encoder = encoder

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last two layers
        if unfreeze:  # TODO: just to speed up in collab... delete after
            for param in self.encoder.encoder.layer[-2:].parameters():
                param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
