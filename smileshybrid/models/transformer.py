from torch import nn
from transformers import AutoModel

class Transformer(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", 
                                               is_decoder=True, 
                                               add_cross_attention=True)
        self.dropout = nn.Dropout(0.5)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        transformer_outputs  = self.model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          encoder_hidden_states=encoder_hidden_states)
        
        hidden_states = transformer_outputs[0]
        logits = self.regressor(hidden_states)
        #pooled_logits = logits[range(batch_size), sequence_length]
        pooled_logits = logits.squeeze(-1)[:,-1]
        
        return pooled_logits
