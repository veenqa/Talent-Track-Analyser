
//DATASET PREPROCESSING WITH BERT


from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch import cuda

# Check for GPU
device = "cuda" if cuda.is_available() else "cpu"

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_)).to(device)

# Reduce dataset size for quick testing
train_dataset = train_dataset.select(range(100))  # Use only the first 100 samples
test_dataset = test_dataset.select(range(50))     # Use only the first 50 samples

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=1,              # Reduce epochs for faster training
    per_device_train_batch_size=4,   # Reduce batch size
    per_device_eval_batch_size=8,    # Reduce eval batch size
    warmup_steps=100,                # Fewer warmup steps
    weight_decay=0.01,               # Weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,                # Log every 10 steps
    evaluation_strategy="steps",     # Evaluate periodically
    save_strategy="steps",           # Save periodically
    fp16=True,                       # Enable mixed precision training
    max_steps=50,                    # Train for 50 steps for quick debugging
    load_best_model_at_end=True      # Load best model at the end of training
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=test_dataset,           # Evaluation dataset
    tokenizer=tokenizer                  # Tokenizer for text processing
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the trained model and tokenizer
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")




OUTPUT..

model.safetensors: 100%
 440M/440M [00:06<00:00, 63.3MB/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
<ipython-input-5-f81d06353507>:32: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
max_steps is given, it will override any value given in num_train_epochs
wandb: WARNING The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: ··········
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
Tracking run with wandb version 0.18.6
Run data is saved locally in /content/wandb/run-20241117_170622-ht0axzen
Syncing run ./results to Weights & Biases (docs)
View project at https://wandb.ai/panchami3095-sns-college-of-technology/huggingface
View run at https://wandb.ai/panchami3095-sns-college-of-technology/huggingface/runs/ht0axzen
 [50/50 30:52, Epoch 2/2]
Step	Training Loss	Validation Loss
10	3.279100	3.251335
20	3.237300	3.191369
30	3.203300	3.210199
40	3.182700	3.236214
50	3.052000	3.159226
 [7/7 01:16]
('./final_model/tokenizer_config.json',
 './final_model/special_tokens_map.json',
 './final_model/vocab.txt',
 './final_model/added_tokens.json')
