from turkish_tokenizer import TurkishTokenizer, HFTurkishTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = TurkishTokenizer()

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
class HFTokenizer:
    def __init__(self):
        self.tokenizer = HFTurkishTokenizer()

    def tokenize(self, text, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        return self.tokenizer(
            text, 
            add_special_tokens=add_special_tokens, 
            max_length=max_length,
            padding=padding, 
            truncation=truncation,
            return_tensors=return_tensors
        )