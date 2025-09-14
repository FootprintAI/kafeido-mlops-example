def trainOp(
    data_name: str = 'my_audio_dataset',
    model_relative_path: str = 'whisper_model',
    model_name: str = 'whisper-small-finetuned',
    model_dir: str = 'openai/whisper-small',
    num_train_epochs: float = 3,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    max_length: int = 225,
    precision: str = 'fp16',
    use_quantization: bool = False,
    full_finetune: bool = True,
    lora_rank: int = None,
    lora_alpha: int = None,
    quantization_type: str = None
):
    """
    Fine-tunes an OpenAI Whisper ASR model on a custom audio dataset.
    
    This function handles the entire fine-tuning process for a Whisper model.
    It loads a dataset (e.g., from Hugging Face Hub), preprocesses the audio
    and text data, sets up the model and a specialized data collator, and
    executes the training loop using the Hugging Face Trainer API.
    
    Args:
        data_name (str): The name of the dataset to load from the Hugging Face
            Hub or a local directory. Defaults to 'my_audio_dataset'.
        model_relative_path (str): The relative path within the home directory
            to save the fine-tuned model. Defaults to 'whisper_model'.
        model_name (str): The name of the fine-tuned model directory.
            Defaults to 'whisper-small-finetuned'.
        model_dir (str): The Hugging Face Hub model ID or local path for the
            pre-trained Whisper model. Defaults to 'openai/whisper-small'.
        num_train_epochs (float): The number of training epochs. Defaults to 3.
        per_device_train_batch_size (int): The number of audio examples per
            device in a training batch. Defaults to 16.
        gradient_accumulation_steps (int): The number of steps to accumulate
            gradients before performing a backward pass. Defaults to 1.
        learning_rate (float): The learning rate for the optimizer.
            Defaults to 1e-5.
        max_length (int): The maximum number of tokens for the model's
            generated output. Defaults to 225.
        precision (str): The training precision, either 'fp16' or 'bf16'.
            'bf16' is recommended for newer GPUs. Defaults to 'fp16'.
        use_quantization (bool): Whether to use quantization (not typically used
            for standard Whisper fine-tuning). Defaults to False.
        full_finetune (bool): Whether to perform a full fine-tuning. For Whisper,
            this is the standard approach. Defaults to True.
        lora_rank (int): LoRA attention dimension. This is not used in this
            script as it's set up for full fine-tuning. Defaults to None.
        lora_alpha (int): The alpha parameter for LoRA scaling. Not used.
            Defaults to None.
        quantization_type (str): The type of quantization to use. Not used.
            Defaults to None.

    Returns:
        None: The function saves the trained model to the specified directory
            and does not return a value.
    """
    import os
    import logging
    import torch

    from datasets import load_dataset, Audio, DatasetDict
    from transformers import (
        WhisperProcessor, 
        WhisperForConditionalGeneration, 
        Seq2SeqTrainingArguments, 
        Seq2SeqTrainer,
    )
    import evaluate
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    home = '/home/jovyan'
    input_dir = os.path.join(home, 'data')
    output_dir = os.path.join(home, model_relative_path, model_name)

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # --- Data Loading and Preparation (Whisper-specific) ---
    logger.info("Loading and preparing dataset...")

    try:
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", trust_remote_code=True)
        common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", trust_remote_code=True)
        
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
        common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # --- Model and Processor Loading (Whisper-specific) ---
    logger.info("Loading Whisper model and processor...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    
    # --- Prepare Data for Training (Whisper-specific) ---
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.map(prepare_dataset, num_proc=4)

    # --- Data Collator and Metrics (Whisper-specific) ---
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    # --- Training Setup ---
    logger.info("Setting up training...")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        evaluation_strategy="steps",
        per_device_eval_batch_size=per_device_train_batch_size,
        predict_with_generate=True,
        generation_max_length=max_length,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # --- Start Training ---
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        trainer.save_model()
        logger.info(f"Model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    return None
