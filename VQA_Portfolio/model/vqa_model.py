import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

class VQADataset(Dataset):
    """Dataset dla zadania Visual Question Answering"""
    
    def __init__(self, csv_path, img_dir, transform=None, max_length=128, 
                 answer_map_path=None, is_test=False):
        """
        Inicjalizacja datasetu VQA.
        
        Args:
            csv_path (str): Ścieżka do pliku CSV z danymi
            img_dir (str): Ścieżka do folderu z obrazami
            transform (callable, optional): Opcjonalne transformacje do zastosowania na obrazach
            max_length (int, optional): Maksymalna długość tokenizowanego pytania
            answer_map_path (str, optional): Ścieżka do pliku z mapowaniem odpowiedzi na indeksy
            is_test (bool, optional): Czy to zbiór testowy (nie oczekujemy odpowiedzi)
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_test = is_test
        self.transform = transform if transform else self._get_default_transform()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
        # Wczytanie mapowania odpowiedzi na indeksy (jeśli podano)
        if answer_map_path and os.path.exists(answer_map_path):
            with open(answer_map_path, 'r') as f:
                self.answer_to_idx = json.load(f)
            print(f"Wczytano mapowanie dla {len(self.answer_to_idx)} odpowiedzi.")
        else:
            self.answer_to_idx = None
            print("Brak mapowania odpowiedzi - wykorzystane zostaną indeksy z kolumny answer_idx.")
    
    def _get_default_transform(self):
        """Zwraca domyślne transformacje dla obrazów"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Zwraca pojedynczy element datasetu"""
        row = self.df.iloc[idx]
        img_id = row['image_id']
        question = row['question']
        
        # Ścieżka do obrazu
        img_path = os.path.join(self.img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        
        # Wczytywanie obrazu
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Błąd wczytywania obrazu {img_path}: {e}")
            # Zastępczy obraz (czarny)
            image = torch.zeros((3, 224, 224))
        
        # Tokenizacja pytania
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Przygotowanie danych wyjściowych
        output = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'question_id': row['question_id'],
            'question': question
        }
        
        # Dodanie odpowiedzi, jeśli to nie jest zbiór testowy
        if not self.is_test:
            if 'answer_idx' in row:
                output['answer_idx'] = torch.tensor(row['answer_idx'], dtype=torch.long)
            elif 'multiple_choice_answer' in row and self.answer_to_idx:
                answer = row['multiple_choice_answer']
                answer_idx = self.answer_to_idx.get(answer, len(self.answer_to_idx))
                output['answer_idx'] = torch.tensor(answer_idx, dtype=torch.long)
        
        return output

class VQAModel(nn.Module):
    """Model dla zadania Visual Question Answering łączący cechy obrazu i tekstu"""
    
    def __init__(self, num_classes=1000, pretrained=True):
        """
        Inicjalizacja modelu VQA.
        
        Args:
            num_classes (int, optional): Liczba klas (możliwych odpowiedzi)
            pretrained (bool, optional): Czy używać wstępnie wytrenowanych wag dla modeli
        """
        super(VQAModel, self).__init__()
        
        # Model obrazu (ResNet-50)
        self.image_model = models.resnet50(pretrained=pretrained)
        # Usunięcie ostatniej warstwy klasyfikacyjnej
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
        
        # Model tekstu (BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Wymiary cech
        self.image_features_dim = 2048
        self.text_features_dim = 768
        
        # Warstwy fuzji
        self.fusion = nn.Sequential(
            nn.Linear(self.image_features_dim + self.text_features_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, input_ids, attention_mask):
        """
        Forward pass modelu.
        
        Args:
            image (torch.Tensor): Tensor obrazu [batch_size, 3, 224, 224]
            input_ids (torch.Tensor): Token IDs pytania [batch_size, max_length]
            attention_mask (torch.Tensor): Maska uwagi dla pytania [batch_size, max_length]
            
        Returns:
            torch.Tensor: Logity dla każdej klasy odpowiedzi [batch_size, num_classes]
        """
        # Ekstrakcja cech obrazu
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Ekstrakcja cech tekstu
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Konkatenacja cech
        combined_features = torch.cat((image_features, text_features), dim=1)
        
        # Fuzja i klasyfikacja
        output = self.fusion(combined_features)
        
        return output

if __name__ == "__main__":
    main()
    
def get_transforms():
    """Zwraca transformacje dla obrazów używane w treningu i walidacji"""
    
    # Transformacje dla zbioru treningowego (z augmentacją)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformacje dla zbioru walidacyjnego/testowego (bez augmentacji)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def test_model_architecture():
    """Testuje architekturę modelu z przykładowymi danymi wejściowymi"""
    
    # Tworzenie przykładowych danych wejściowych
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (batch_size, 128))  # 30522 to rozmiar słownika BERT
    attention_mask = torch.ones(batch_size, 128)
    
    # Inicjalizacja modelu
    model = VQAModel(num_classes=1000)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(image, input_ids, attention_mask)
    
    # Wyświetlenie informacji o wyjściu
    print(f"Kształt wyjścia modelu: {outputs.shape}")
    print(f"Przykładowe logity: {outputs[0, :5]}")
    
    # Rysowanie architektury modelu (uproszczone)
    print("\nArchitektura modelu VQA:")
    print("1. Obraz --> ResNet-50 --> 2048-wymiarowy wektor cech")
    print("2. Pytanie --> BERT --> 768-wymiarowy wektor cech")
    print("3. Konkatenacja [2048 + 768] --> 2816-wymiarowy wektor")
    print("4. Fully Connected (2816 -> 1024) --> ReLU --> Dropout(0.5)")
    print("5. Fully Connected (1024 -> 512) --> ReLU --> Dropout(0.5)")
    print("6. Fully Connected (512 -> num_classes)")
    
    return model

def main():
    """Główna funkcja do testowania modelu"""
    
    # Testowanie architektury modelu
    model = test_model_architecture()
    
    # Zapisanie podsumowania modelu
    save_model_summary(model)
    
    print("\nModel VQA został pomyślnie zdefiniowany i przetestowany!")

def save_model_summary(model, output_path="VQA_Portfolio/model/model_summary.txt"):
    """Zapisuje podsumowanie modelu do pliku"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Zapisanie podstawowych informacji o modelu
        f.write("=== Model VQA ===\n\n")
        f.write("Architektura:\n")
        f.write("1. Ekstraktor cech obrazu: ResNet-50 (bez warstwy klasyfikacyjnej)\n")
        f.write("2. Ekstraktor cech tekstu: BERT base-uncased\n")
        f.write("3. Fuzja cech: Konkatenacja + MLP\n\n")
        
        # Zapisanie informacji o wymiarach
        f.write("Wymiary cech:\n")
        f.write(f"- Wymiar cech obrazu: {model.image_features_dim}\n")
        f.write(f"- Wymiar cech tekstu: {model.text_features_dim}\n")
        f.write(f"- Całkowity wymiar po konkatenacji: {model.image_features_dim + model.text_features_dim}\n\n")
        
        # Informacje o warstwach fuzji
        f.write("Warstwy fuzji:\n")
        for i, layer in enumerate(model.fusion):
            f.write(f"- Warstwa {i}: {layer}\n")
    
    print(f"Zapisano podsumowanie modelu do {output_path}")
