"""!pip install transformers[torch]
!pip install accelerate -U"""

import torch
from torch.utils.data import Dataset
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, TextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline


# class PairDataset(Dataset):

#     def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        
#         with open(file_path, encoding="utf-8") as f:
#             lines = [line.strip() for line in f.read().split("\n\n") if (len(line) > 0 and not line.isspace())]

#         batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
#         self.examples = batch_encoding["input_ids"]
#         self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i) -> Dict[str, torch.tensor]:
#         return self.examples[i]


torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

# Freeze some first layers not to change all the pre-trained model's weights or if the GPU memory is insufficient. 
# for layer in model.transformer.h[:12].parameters():
#   layer.requires_grad = False

# Create a dataset from the paragraphs
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path= r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\custom_data.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    fp16=True,
    optim="adafactor",
    save_steps=10000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# Generate text using Trainer's generate method
generator = pipeline('text-generation', model=trainer.model, tokenizer=tokenizer)

prompt = "Does Furkan like to eat pizza? <QUESTION>"
output = generator(prompt, max_length=64,
                   #temperature=1.,
                   #top_p=0.1,
                   #top_k=10,
                   #repetition_penalty=1.,
                   num_return_sequences=1,
                   do_sample=True,
                   truncation=True,
                   )

print(output[0]['generated_text'])
