"""
Fine-tune BART or T5 for summarization on CNN/DailyMail.
This is a template using HuggingFace datasets + Trainer.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

model_name = "facebook/bart-large-cnn"  # or t5-base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("cnn_dailymail", "3.0.0")
def preprocess(batch):
    inputs = batch["article"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

args = Seq2SeqTrainingArguments(
    output_dir="./summ-tuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    logging_steps=200,
    save_total_limit=2,
    num_train_epochs=1,
    fp16=True if True else False
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], tokenizer=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model("./summ-tuned")
