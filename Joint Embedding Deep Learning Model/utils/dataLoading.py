import os

class InputExample(object):

    def __init__(self, unique_id, text, label):
        self.unique_id = unique_id
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, input_labels):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.input_labels = input_labels

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = [token.lower() for token in example.text]
        if len(tokens) > seq_length - 2:
            tokens = tokens[0:(seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_type_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_labels = ['O'] + example.label + ['O']

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
            input_labels.append('O')

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        assert len(input_labels) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                input_labels=input_labels,
                ))
    return features

def read_examples(input_path):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    for file in os.listdir(input_path):
        unique_id = int(file.replace('.txt', ''))
        text = []
        label = []
        with open('%s/%s' % (input_path, file), "r", encoding='utf-8') as reader:
            while True:
                line = reader.readline().strip()
                if len(line)<1:
                    break
                line = line.split('\t')
                text.append(line[0])
                label.append(line[1])
        assert len(text) == len(label)
        examples.append(
            InputExample(unique_id=unique_id, text=text, label=label))
    return examples

def generate_instance(input_file, max_seq_length, tokenizer):
    examples = read_examples(input_file)
    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)
    return features

def labelSummary(trainFeatures):
    labelIndex={'O':1}
    index = 2
    for f in trainFeatures:
        for label in f.input_labels:
            if not label in labelIndex.keys():
                labelIndex[label]=index
                index+=1
    return labelIndex

def dataLabelIndexed(features,labelIndex):
    for f in features:
        input_labelsIndex=[labelIndex[w] for w in f.input_labels]
        f.input_labels=input_labelsIndex
    return features
