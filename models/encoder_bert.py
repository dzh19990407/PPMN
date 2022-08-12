import torch
import torch.nn as nn

from models.tokenization import BertTokenizer
from models.modeling import VISUAL_CONFIG, BertPreTrainedModel
from models.modeling import BertEmbeddings, CrossEncoder, BertPooler

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip()) #list of words
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2: #max_seq_length=230
            tokens_a = tokens_a[:(max_seq_length - 2)] 
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))

    return features


def set_visual_config():
    VISUAL_CONFIG.l_layers = 12 # language
    VISUAL_CONFIG.x_layers = 5 # cross
    VISUAL_CONFIG.r_layers = 3 # vision


class BertModel(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CrossEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #extended_attention_mask.shape=[60,1,1,230]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_visual_attention_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids) #[60,230,768]
        lang_feats = self.encoder(
            embedding_output,
            extended_attention_mask)
        # lang_feats, visn_feats = self.encoder(
        #     embedding_output,
        #     extended_attention_mask,
        #     visn_attention_mask=extended_visual_attention_mask,
        #     noun_phrases=noun_phrases)
        pooled_output = self.pooler(lang_feats) #[60,768]

        # return (lang_feats, visn_feats), pooled_output
        return lang_feats, pooled_output

class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_length = cfg.max_sequence_length
        set_visual_config()

        self.model = VisBert.from_pretrained(cfg.pretrained_bert)
        # self.model.cuda()
        self.tokenizer = BertTokenizer.from_pretrained(
            cfg.bert_tokenize, do_lower_case=True)
        # self.tokenizer.cuda()

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        #shape=[60,230]

        output_lang, output_cross = self.model(input_ids, segment_ids, input_mask)

        return output_lang, output_cross

class VisBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return feat_seq, pooled_output