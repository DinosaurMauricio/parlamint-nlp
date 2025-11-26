import torch.nn as nn


class CustomClassifier(nn.Module):

    def __init__(self, encoder, num_classes, config):
        super().__init__()

        self.encoder = encoder

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last two layers
        if config.training.unfreeze:  # TODO: just to speed up in collab... delete after
            for param in self.encoder.encoder.layer[-2:].parameters():
                param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size  # 768
        # hidden_size = 768
        dropout = config.training.classifier.dropout
        classifier_hidden_dim = config.training.classifier.hidden_dim

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS]

        pooled_output = self.norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
