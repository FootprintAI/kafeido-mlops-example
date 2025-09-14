
def trainOp(
    data_name: str = 'my_voice_cloning_dataset',
    model_relative_path: str = 'xtts_model',
    model_name: str = 'xtts_finetuned',
    model_dir: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
    num_train_epochs: float = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 0.0001,
    precision: str = 'fp16',
    use_quantization: bool = False,
    full_finetune: bool = False,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    quantization_type: str = None
):
    """
    Fine-tunes the XTTS v2 model for voice cloning with a post-training verification step.
    
    This function prepares a dataset for a specific speaker, fine-tunes the XTTS model,
    and then generates sample audio using the fine-tuned model for verification.

    Args:
        data_name (str): The name of the dataset folder, which should contain both a
            'train' and 'test' subdirectory. The 'test' subdirectory is used for
            post-training verification. Defaults to 'my_voice_cloning_dataset'.
        model_relative_path (str): The relative path within the home directory
            to save the fine-tuned model. Defaults to 'xtts_model'.
        model_name (str): The name of the fine-tuned model directory.
            Defaults to 'xtts_finetuned'.
        model_dir (str): The pre-trained model ID for the XTTS base model.
            Defaults to 'tts_models/multilingual/multi-dataset/xtts_v2'.
        num_train_epochs (float): The number of training epochs. Defaults to 1.
        per_device_train_batch_size (int): The number of examples per device in
            a training batch. Defaults to 4.
        gradient_accumulation_steps (int): The number of steps to accumulate
            gradients. Defaults to 1.
        learning_rate (float): The learning rate for the optimizer.
            Defaults to 0.0001.
        precision (str): The training precision, either 'fp16' or 'bf16'.
            Defaults to 'fp16'.
        use_quantization (bool): Whether to use quantization. Not typically used.
            Defaults to False.
        full_finetune (bool): Whether to perform full fine-tuning. LoRA is
            recommended. Defaults to False.
        lora_rank (int): LoRA attention dimension (the "rank"). Defaults to 64.
        lora_alpha (int): The alpha parameter for LoRA scaling. Defaults to 128.
        quantization_type (str): The type of quantization. Not used.
            Defaults to None.

    Returns:
        None: The function saves the trained model and verification audio and
            does not return a value.
    """
    import os
    import logging
    import torch
    from pathlib import Path
    from TTS.api import TTS

    home = '/home/jovyan'
    input_dir = os.path.join(home, 'data', data_name)
    output_dir = os.path.join(home, model_relative_path, model_name)
    verification_output_dir = os.path.join(output_dir, 'verification_samples')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(verification_output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Data and Model Setup ---
    try:
        if not os.path.exists(os.path.join(input_dir, 'train')) or not os.path.exists(os.path.join(input_dir, 'test')):
            raise FileNotFoundError(f"Dataset directory must contain 'train' and 'test' subdirectories: {input_dir}")
        
        logger.info(f"Using dataset from: {input_dir}")

    except FileNotFoundError as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"An error occurred during data setup: {e}")
        return

    # Initialize the TTS model
    tts = TTS(model_dir)

    # --- Training Configuration ---
    config = {
        "model_path": model_dir,
        "run_name": model_name,
        "output_path": output_dir,
        "epochs": int(num_train_epochs),
        "batch_size": per_device_train_batch_size,
        "grad_accum": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "use_fp16": (precision == 'fp16'),
        "use_bf16": (precision == 'bf16'),
    }

    # XTTS `train` method handles the training loop
    try:
        logger.info("Starting XTTS fine-tuning...")
        tts.train(
            dataset_path=Path(input_dir),
            config=config,
        )
        logger.info("XTTS fine-tuning completed successfully!")

    except Exception as e:
        logger.error(f"XTTS training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # --- Verification Step ---
    logger.info("Starting verification step...")
    try:
        # Load the fine-tuned model for verification
        # The `tts.load_checkpoint` method loads the new LoRA weights
        fine_tuned_model_path = os.path.join(output_dir, 'XTTS_finetune_1.pth')
        tts.load_checkpoint(fine_tuned_model_path)
        
        test_file_path = os.path.join(input_dir, 'test', 'metadata.csv')
        if not os.path.exists(test_file_path):
            logger.warning("No test metadata file found. Skipping verification.")
            return

        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_lines = f.readlines()[:5]  # Take a few lines for quick testing
            
        for i, line in enumerate(test_lines):
            try:
                wav_file, transcript = line.strip().split('|')
                
                # Get the speaker embedding from a reference audio in the test set
                reference_wav_path = os.path.join(input_dir, 'test', 'wavs', wav_file)
                
                # Generate a sample audio
                output_wav_path = os.path.join(verification_output_dir, f'sample_{i}.wav')
                
                logger.info(f"Synthesizing sample {i+1} for verification...")
                tts.tts_with_vc_to_file(
                    text=transcript,
                    file_path=reference_wav_path,
                    speaker=None, # The model automatically uses the embedding from the reference audio
                    file_out=output_wav_path,
                    language='en' # Specify the language of your dataset
                )
                logger.info(f"Verification sample saved to: {output_wav_path}")
                
            except Exception as e:
                logger.error(f"Failed to synthesize sample {i}: {e}")
                continue

        logger.info("Verification step completed. Please listen to the samples for performance evaluation.")

    except Exception as e:
        logger.error(f"An error occurred during the verification step: {e}")
        import traceback
        traceback.print_exc()

    return None
