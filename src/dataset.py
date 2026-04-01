import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import torchvision.transforms as T
import nltk
from sklearn.model_selection import train_test_split

# Download NLTK punkt tokenizer (needs to be run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        self.idx = 4
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return nltk.tokenize.word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1
                
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi.get(token, self.stoi["<unk>"]) 
            for token in tokenized_text
        ]


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None, freq_threshold=5, split='train', test_size=0.2, val_size=0.1, random_state=42):
        self.img_dir = img_dir
        
        # Check if the captions file exists
        if not os.path.exists(captions_file):
            print(f"Warning: {captions_file} not found. Please download the dataset.")
            self.df = pd.DataFrame(columns=['image', 'caption'])
            self.vocab = Vocabulary(freq_threshold)
            return

        self.df = pd.read_csv(captions_file)
        # Some versions of Flickr8k use image,caption header, some use other formats. Assumes standard comma separated.
        if 'image' not in self.df.columns or 'caption' not in self.df.columns:
            # Fallback if it's the standard txt format separated by commas
            self.df = pd.read_csv(captions_file, sep=",")
            
        self.transform = transform
        
        # Split by image to avoid data leakage
        unique_images = self.df['image'].unique()
        
        # Train / Temp (Val+Test) split
        train_imgs, temp_imgs = train_test_split(
            unique_images, test_size=(test_size + val_size), random_state=random_state
        )
        
        # Val / Test split
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=test_size/(test_size + val_size), random_state=random_state
        )
        
        if split == 'train':
            split_imgs = train_imgs
        elif split == 'val':
            split_imgs = val_imgs
        else: # test
            split_imgs = test_imgs
            
        # Filter dataframe for the current split
        self.df = self.df[self.df['image'].isin(split_imgs)].reset_index(drop=True)
        
        # Build vocabulary using ONLY training captions
        self.vocab = Vocabulary(freq_threshold)
        if split == 'train':
            self.vocab.build_vocabulary(self.df['caption'].astype(str).tolist())
            
        self.imgs = self.df['image'].tolist()
        self.captions = self.df['caption'].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        
        img_path = os.path.join(self.img_dir, img_id)
        # Handle cases where the dataset isn't loaded yet gracefully
        if not os.path.exists(img_path):
            img = Image.new('RGB', (224, 224), color = 'black')
        else:
            img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])
        
        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )
        
        return imgs, targets


def get_data_loaders(img_dir, captions_file, transform=None, batch_size=32, num_workers=2):
    if transform is None:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    train_dataset = Flickr8kDataset(img_dir, captions_file, transform=transform, split='train')
    val_dataset = Flickr8kDataset(img_dir, captions_file, transform=transform, split='val')
    test_dataset = Flickr8kDataset(img_dir, captions_file, transform=transform, split='test')
    
    # We use the train vocabulary for validation and testing
    val_dataset.vocab = train_dataset.vocab
    test_dataset.vocab = train_dataset.vocab
    
    pad_idx = train_dataset.vocab.stoi["<pad>"]
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab

