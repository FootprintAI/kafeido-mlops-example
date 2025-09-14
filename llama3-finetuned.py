def trainOp(data_name: str = 'financial_sentiment_data-20250715.jsonl', 
            model_relative_path: str = 'model', 
            model_name: str = 'llama31-financial-sentiment',
            model_dir: str = '/mnt/pretrained-models/',
            num_train_epochs: float = 1,
            per_device_train_batch_size: int = 4, 
            gradient_accumulation_steps: int = 4, 
            learning_rate: float = 2e-4,
            max_length: int = 1024,
            lora_rank: int = 16,
            lora_alpha: int = 32,
            precision: str = 'bf16',
            use_quantization: bool = False,
            full_finetune: bool = False,
            quantization_type: str = 'nf4'):
    """
    Main entry point for fine-tuning and saving a LLM model.
    
    Args:
        data_name (str): Relative name for input data, default 'financial_sentiment_data-20250715.jsonl'
        model_relative_path (str): Relative path for model storage, default 'model'
        model_name (str): Name of the model, default 'llama31-financial-sentiment'
        model_dir (str): Relative path for model path, default '/mnt/pretrained-models/'
        num_train_epochs (int): Number of training epochs, default 1
        per_device_train_batch_size (int): Number of train batch size, default 4
        gradient_accumulation_steps (int): Number of gradient accumulation steps, default 4
        learning_rate (float): Number of lr, default 2e-4
        max_length (int): The maximum length the generated tokens can have, default 1024
        lora_rank (int): Lora attention dimension (the “rank”), default 16
        lora_alpha (int): The alpha parameter for Lora scaling, default 32
        precision (str): Training precision (fp32, bf16, or fp16)
        use_quantization (bool): Use 4-bit quantization (useful for smaller GPUs)
        full_finetune (bool): Do full fine-tuning instead of LoRA (requires more VRAM)
        quantization_type (str): quantization type, default nf4
        
        
        
    
    Returns:
        None
    """
    #!/usr/bin/env python3
    """
    fine_tune_llama_minimal.py - Minimal working fine-tuning script for Llama-3.1-8B
    Enhanced with FP32 support and no quantization for A100 40GB
    """
    
    import os
    import json
    import logging
    import numpy as np
    import torch
    import zipfile
    from typing import Dict, List, Any

    
    
    
    home = '/home/jovyan'
    input_dir = os.path.join(home, 'data')
    output_dir = os.path.join(home, model_relative_path, model_name)
    # Create input directory
    os.makedirs(input_dir, exist_ok=True)
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    logger.info(f"Local model cache path {model_dir}")
    
    
    
    def download_data():
        """Download fine-tuning data from MinIO storage."""
        from tintin.file.minio import FileManager as FileManager
        print('Downloading data')
        print(input_dir)
        debug = False
        recursive = False
        mgr = FileManager('', debug)
        mgr.download(input_dir, [data_name], recursive)
        print('Download completed')
    
    def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
        """Load training examples from a jsonl file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    
    def format_instruction(example: Dict[str, str]) -> str:
        """Format the instruction, input, and output for Llama-3.1 chat format."""
        system_prompt = "You are a financial analyst who specializes in detecting emotional subtext in earnings calls."
        
        # Get the text fields and clean them
        instruction = str(example.get('instruction', '')).strip()
        input_text = str(example.get('input', '')).strip()
        output_text = str(example.get('output', '')).strip()
        
        # Create the formatted text - DO NOT add <|begin_of_text|> here, tokenizer will add it
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        return formatted
    
    def train():
        
        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Quantization: {'Yes' if use_quantization else 'No'}")
        logger.info(f"  Training type: {'Full fine-tune' if full_finetune else 'LoRA'}")
        logger.info(f"  Batch size: {per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        download_data()
        input_file = os.path.join(input_dir, data_name)
        # Load data
        logger.info("Loading data...")
        examples = load_jsonl_data(input_file)
        
        # Filter examples with output
        valid_examples = [ex for ex in examples if ex.get('output')]
        logger.info(f"Found {len(valid_examples)} valid examples with output")
        
        # Take a subset for testing
        if len(valid_examples) > 10000:
            valid_examples = valid_examples[:10000]
            logger.info(f"Using subset of {len(valid_examples)} examples for testing")
        
        # Format examples
        logger.info("Formatting examples...")
        formatted_texts = []
        for i, example in enumerate(valid_examples):
            try:
                formatted = format_instruction(example)
                formatted_texts.append(formatted)
            except Exception as e:
                logger.warning(f"Failed to format example {i}: {e}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_texts)} examples")
        
        # Show first example
        if formatted_texts:
            logger.info(f"First example (first 300 chars):\n{formatted_texts[0][:300]}...")
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig
        import torch  # Ensure torch is available in this scope
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Determine torch dtype based on precision
        if precision == "fp32":
            torch_dtype = torch.float32
            logger.info("Using FP32 precision")
        elif precision == "fp16":
            torch_dtype = torch.float16
            logger.info("Using FP16 precision")
        else:  # bf16
            torch_dtype = torch.bfloat16
            logger.info("Using BF16 precision")
        
        # Setup model loading arguments
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        # Add quantization config if requested
        if use_quantization and (quantization_type == 'int2' or quantization_type == 'int4'):
            logger.info("Using int2 or int4 quantization...")
            quantization_config = QuantoConfig(weights=quantization_type)
            model_kwargs["quantization_config"] = quantization_config
        elif use_quantization and quantization_type == 'nf4':
            logger.info("Using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            logger.info("Loading model without quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            **model_kwargs
        )
        if quantization_type == 'int2' or quantization_type == 'int4':
            model.enable_input_require_grads()
        # Setup PEFT (only if not doing full fine-tuning)
        if not full_finetune:
            logger.info("Setting up LoRA...")
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
            
            model.config.use_cache = False
            
            # Only prepare for k-bit training if using bitsandbytes quantization
            if use_quantization and quantization_type == 'nf4':
                model = prepare_model_for_kbit_training(model)
            else:
                # For models that are non-quantized, or quantized to INT2 or INT4, simply enable gradient checkpointing
                model.gradient_checkpointing_enable()
            
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            logger.info("Setting up full fine-tuning...")
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        
        # Tokenize data
        logger.info("Tokenizing data...")
        from datasets import Dataset
        
        def tokenize_function(examples):
            # Tokenize the texts
            model_inputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            
            # For causal language modeling, labels are the same as input_ids
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        # Create dataset and tokenize
        dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"],
            desc="Tokenizing data"
        )
        
        # Filter out sequences that are too short (less than 10 tokens)
        def filter_short_sequences(example):
            return len(example["input_ids"]) >= 10
        
        tokenized_dataset = tokenized_dataset.filter(filter_short_sequences)
        logger.info(f"Dataset size after filtering short sequences: {len(tokenized_dataset)}")
        
        # Check tokenized data
        logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
        if len(tokenized_dataset) > 0:
            # Check sequence lengths
            lengths = [len(example["input_ids"]) for example in tokenized_dataset]
            logger.info(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
            
            sample = tokenized_dataset[0]
            logger.info(f"Sample tokenized length: {len(sample['input_ids'])}")
            # Check for duplicate begin tokens
            input_ids = sample['input_ids']
            if len(input_ids) > 1 and input_ids[0] == input_ids[1] == 128000:
                logger.warning("Found duplicate <|begin_of_text|> tokens!")
            
            # Decode to check format
            decoded = tokenizer.decode(input_ids[:100], skip_special_tokens=False)
            logger.info(f"Sample decoded (first 100 tokens): {decoded}")
        
        if len(tokenized_dataset) == 0:
            logger.error("No valid tokenized examples found!")
            return
        
        # Setup training
        logger.info("Setting up training...")
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        # Configure training arguments based on precision
        training_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "logging_steps": 10,
            "save_steps": 500,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "report_to": "none",
        }
        
        # Set precision-specific arguments
        if precision == "fp32":
            # No special precision flags for FP32
            pass
        elif precision == "fp16":
            training_kwargs["fp16"] = True
        else:  # bf16
            training_kwargs["bf16"] = True
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Custom data collator that handles variable length sequences properly
        from transformers.data.data_collator import DataCollatorMixin
        from dataclasses import dataclass
        from typing import Any, Dict, List, Union
        import torch
        
        @dataclass
        class DataCollatorForCausalLM(DataCollatorMixin):
            """
            Data collator for causal language modeling that properly handles padding and labels.
            """
            tokenizer: Any
            pad_to_multiple_of: int = None
            return_tensors: str = "pt"
            
            def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
                # Handle the input_ids and labels
                batch = {}
                
                # Get all input_ids and labels
                input_ids = [example["input_ids"] for example in examples]
                labels = [example["labels"] for example in examples]
                
                # Pad sequences to the same length
                batch["input_ids"] = self._pad_sequences(input_ids, self.tokenizer.pad_token_id)
                batch["labels"] = self._pad_sequences(labels, -100)  # -100 is ignored in CrossEntropy loss
                
                # Create attention mask
                batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
                
                return batch
            
            def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
                """Pad sequences to the same length."""
                max_length = max(len(seq) for seq in sequences)
                
                # Pad to multiple if specified
                if self.pad_to_multiple_of is not None:
                    max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
                
                padded_sequences = []
                for seq in sequences:
                    padded_seq = seq + [pad_value] * (max_length - len(seq))
                    padded_sequences.append(padded_seq)
                
                return torch.tensor(padded_sequences, dtype=torch.long)
        
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Check training setup
        train_dataloader = trainer.get_train_dataloader()
        logger.info(f"Training dataloader batches: {len(train_dataloader)}")
        logger.info(f"Total training steps: {len(train_dataloader) * num_train_epochs}")
        
        # Print memory info if using CUDA
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        if len(train_dataloader) == 0:
            logger.error("No training batches! Check your data and batch size.")
            return
        
        # Start training
        logger.info("Starting training...")
        try:
            result = trainer.train()
            logger.info("Training completed successfully!")
            
            # Save model
            trainer.save_model()
            logger.info(f"Model saved to {output_dir}")
            
            # Print final memory usage
            if torch.cuda.is_available():
                logger.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.info(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Print metrics
            if result and hasattr(result, 'metrics'):
                logger.info(f"Final metrics: {result.metrics}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    train()
    return None  # Explicitly return None at the end of trainOp
