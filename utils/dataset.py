import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids: list, context_length: int, stride: int):
        super(TextDataset, self).__init__()
        
        self.inputs = []
        self.outputs = []
        self.padding_id = 0
        
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # context_length = 4, stride = 2
        # self.inputs = [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11]
        # self.outputs = [1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10], [9, 10, 11, 12]
        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i+context_length]
            output_chunk = token_ids[i+1:i+context_length+1]
            
            # If the input_chunk is longer than context_length, truncate it
            if len(input_chunk) > context_length:
                input_chunk = input_chunk[:context_length]
            if len(output_chunk) > context_length:
                output_chunk = output_chunk[:context_length]
            
            # If the input_chunk is shorter than context_length, pad it
            if len(input_chunk) < context_length:
                input_chunk = input_chunk + [self.padding_id] * (context_length - len(input_chunk))
            if len(output_chunk) < context_length:
                output_chunk = output_chunk + [self.padding_id] * (context_length - len(output_chunk))
            
            self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
            self.outputs.append(torch.tensor(output_chunk, dtype=torch.long))
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
def create_dataloader(dataset: TextDataset, batch_size: int, shuffle: bool = True, device: str = "cpu"):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        generator=torch.Generator(device=device) # For reproducibility
    )
    
if __name__ == "__main__":
    from tokenizer import Tokenizer
    from settings.config import app_config
    
    text = app_config.get_test_texts()["english_long"]
    
    tokenizer = Tokenizer()
    token_ids = tokenizer.tokenize(text)
    
    dataset = TextDataset(
        token_ids,
        app_config.get_test_parameters()["context_length"],
        app_config.get_test_parameters()["stride"]
    )
    
    dataloader = create_dataloader(
        dataset, 
        app_config.get_test_parameters()["batch_size"], 
        True
    )
    for x, y in dataloader:
        print("Input tensor:", x)
        print("Output tensor:", y)
        
        # Her batch'teki her sample'ı ayrı ayrı detokenize et
        for i in range(x.shape[0]):
            input_tokens = x[i].tolist()
            output_tokens = y[i].tolist()
            
            print(f"Sample {i}:")
            print(f"  Input: {input_tokens}")
            print(f"  Input text: {tokenizer.detokenize(input_tokens)}")
            print(f"  Output: {output_tokens}")
            print(f"  Output text: {tokenizer.detokenize(output_tokens)}")
            print("-" * 50)
        break