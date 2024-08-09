from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPImageProcessor, CLIPVisionModelWithProjection
import torch

def load_and_process_data(split='test', num_samples=10000, batch_size=64):
    ds = load_dataset("nlphuji/flickr30k", split=split).select(range(num_samples))

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to('cuda')

    def process_text(examples):
        all_captions = [caption for image_captions in examples["caption"] for caption in image_captions]
        tokenized_text = tokenizer(all_captions, padding=True, truncation=True, return_tensors="pt")
        tokenized_text = {k: v.to('cuda') for k, v in tokenized_text.items()}
        
        with torch.no_grad():
            text_outputs = text_model(**tokenized_text)
        
        text_embeds = text_outputs.text_embeds.cpu().numpy()
        num_images = len(examples["caption"])
        reshaped_embeds = text_embeds.reshape(num_images, 5, -1)
        
        return {"text_embed": reshaped_embeds}

    def process_images(examples):
        images = examples["image"]
        processed_images = image_processor(images, return_tensors="pt")
        processed_images = {k: v.to('cuda') for k, v in processed_images.items()}
        
        with torch.no_grad():
            image_outputs = vision_model(**processed_images)
        
        return {"image_embed": image_outputs.last_hidden_state.cpu().numpy()}

    ds = ds.map(process_text, batched=True, batch_size=32, remove_columns=["caption"])
    ds = ds.map(process_images, batched=True, batch_size=32, remove_columns=["image"])
    ds.set_format(type="torch")

    dataset = ds.train_test_split(test_size=0.2, seed=42)
    train_val_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

    final_dataset = DatasetDict({
        'train': train_val_dataset['train'],
        'validation': train_val_dataset['test'],
        'test': dataset['test']
    })

    train_loader = DataLoader(final_dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(final_dataset['validation'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(final_dataset['test'], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader