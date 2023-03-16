from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)
from datasets import load_metric
import numpy as np
import wandb
from nlp_utils import read_ds

model_type = "t5-base"
batch_size = 4
epochs = 100
max_length = 200
prefix = "translate German to English: "

run_name = 'baseline'
wandb.login(key='7573cbc6e943326835b588046bf1ee71f3f43408')
wandb.init(project=run_name, name=f'attempt 1')


# def path_to_dict(path):
#     data_dict = {'translation': []}
#     sample = {}
#     with open(path) as f:
#         for line in f:
#             if line == "German:\n":
#                 status = "de"
#                 sample[status] = ""
#
#             elif line == "English:\n":
#                 status = "en"
#                 sample[status] = ""
#
#             elif line != '\n':
#                 sample[status] += line
#
#             else:
#                 data_dict['translation'].append(sample)
#                 sample = {}
#
#     return data_dict


def train(datasets, source_lang, target_lang):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    metric = load_metric("sacrebleu")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)

    args = Seq2SeqTrainingArguments(
        f"models/{run_name}_{model_type}",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=epochs,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="wandb")

    def preprocess_function(examples):
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, max_length=max_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


def main():
    train_dataset = Dataset.from_dict(read_ds("data/train.labeled"))
    validation_dataset = Dataset.from_dict(read_ds("data/val.labeled"))
    datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    train(datasets=datasets, source_lang="de", target_lang="en")


if __name__ == '__main__':
    main()
