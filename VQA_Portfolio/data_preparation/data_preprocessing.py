import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import shutil

def load_data(base_path):
    """Wczytuje dane pytań i adnotacji z plików JSON."""
    
    # Ścieżki do plików z pytaniami i adnotacjami
    questions_path = os.path.join(base_path, "v2_Questions_Train_mscoco", "v2_mscoco_train2014_questions.json")
    annotations_path = os.path.join(base_path, "v2_Annotations_Train_mscoco", "v2_mscoco_train2014_annotations.json")
    
    # Wczytywanie danych pytań
    print("Wczytywanie danych pytań...")
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
    
    # Wczytywanie danych adnotacji
    has_annotations = True
    try:
        print("Wczytywanie danych adnotacji...")
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print("Brak pliku adnotacji. Kontynuujemy tylko z pytaniami.")
        has_annotations = False
    
    # Konwersja na DataFrame pandas
    questions_df = pd.DataFrame(questions_data['questions'])
    print(f"Liczba pytań: {len(questions_df)}")
    print(questions_df.head())
    
    if has_annotations:
        annotations_df = pd.DataFrame(annotations_data['annotations'])
        print(f"Liczba adnotacji: {len(annotations_df)}")
        print(annotations_df.head())
        
        # Łączenie pytań z adnotacjami po question_id
        merged_df = questions_df.merge(annotations_df, on='question_id')
        print(f"Liczba połączonych rekordów: {len(merged_df)}")
        print(merged_df.head())
        return merged_df
    else:
        return questions_df

def preprocess_data(df, sample_size=None, save_path=None):
    """Przetwarza dane i opcjonalnie zapisuje podzbiór."""
    
    # Losowy wybór podzbioru danych (jeśli określono sample_size)
    if sample_size and sample_size < len(df):
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"Wybrano losowy podzbiór {sample_size} rekordów.")
    else:
        sampled_df = df
    
    # Dodawanie typów pytań (jeśli nie ma w danych)
    if 'question_type' not in sampled_df.columns and 'question' in sampled_df.columns:
        # Prosta heurystyka do określania typu pytania
        def get_question_type(question):
            question = question.lower()
            if question.startswith('what'):
                return 'what'
            elif question.startswith('how'):
                return 'how'
            elif question.startswith('is'):
                return 'yes/no'
            elif question.startswith('are'):
                return 'yes/no'
            elif question.startswith('does'):
                return 'yes/no'
            elif question.startswith('do'):
                return 'yes/no'
            elif question.startswith('where'):
                return 'where'
            elif question.startswith('when'):
                return 'when'
            elif question.startswith('who'):
                return 'who'
            elif question.startswith('which'):
                return 'which'
            else:
                return 'other'
        
        sampled_df['question_type'] = sampled_df['question'].apply(get_question_type)
    
    # Zapisywanie przetworzonego zbioru danych
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sampled_df.to_csv(save_path, index=False)
        print(f"Zapisano przetworzony zbiór danych do {save_path}")
    
    return sampled_df

def show_examples(df, img_dir, num_examples=5):
    """Wyświetla przykładowe obrazy z pytaniami i odpowiedziami."""
    
    if len(df) == 0:
        print("Brak danych do wyświetlenia.")
        return
    
    # Wybór losowych przykładów
    sample_indices = random.sample(range(len(df)), min(num_examples, len(df)))
    
    for i, idx in enumerate(sample_indices):
        sample = df.iloc[idx]
        img_id = sample['image_id']
        question = sample['question']
        
        # Ścieżka do obrazu
        img_path = os.path.join(img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        
        # Wczytywanie obrazu
        try:
            img = Image.open(img_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            
            # Wyświetlanie informacji
            plt.title(f"Pytanie: {question}")
            if 'multiple_choice_answer' in sample:
                plt.suptitle(f"Odpowiedź: {sample['multiple_choice_answer']}")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Błąd wczytywania obrazu {img_path}: {e}")

def analyze_data(df):
    """Przeprowadza analizę danych VQA."""
    
    print("\n=== Analiza danych VQA ===")
    
    # Statystyki pytań
    print("\nStatystyki dotyczące pytań:")
    df['question_length'] = df['question'].apply(len)
    print(f"Średnia długość pytania: {df['question_length'].mean():.2f} znaków")
    print(f"Mediana długości pytania: {df['question_length'].median()} znaków")
    print(f"Najkrótsze pytanie: {df['question_length'].min()} znaków")
    print(f"Najdłuższe pytanie: {df['question_length'].max()} znaków")
    
    # Typy pytań (jeśli są dostępne)
    if 'question_type' in df.columns:
        print("\nRozkład typów pytań:")
        question_type_counts = df['question_type'].value_counts()
        print(question_type_counts)
        
        # Wykres typów pytań
        plt.figure(figsize=(12, 6))
        question_type_counts.plot(kind='bar')
        plt.title('Rozkład typów pytań')
        plt.xlabel('Typ pytania')
        plt.ylabel('Liczba pytań')
        plt.tight_layout()
        plt.show()
    
    # Analiza odpowiedzi (jeśli są dostępne)
    if 'multiple_choice_answer' in df.columns:
        print("\nAnaliza odpowiedzi:")
        answer_counts = df['multiple_choice_answer'].value_counts()
        print(f"Liczba unikalnych odpowiedzi: {len(answer_counts)}")
        
        print("\nNajczęstsze odpowiedzi:")
        print(answer_counts.head(20))
        
        # Wykres najczęstszych odpowiedzi
        plt.figure(figsize=(12, 8))
        answer_counts.head(20).plot(kind='bar')
        plt.title('Top 20 najczęstszych odpowiedzi')
        plt.xlabel('Odpowiedź')
        plt.ylabel('Liczba wystąpień')
        plt.tight_layout()
        plt.show()

def prepare_data_for_model(df, output_dir, img_dir, num_classes=1000):
    """Przygotowuje dane do treningu modelu, w tym mapowanie odpowiedzi."""
    
    # Katalog wyjściowy
    os.makedirs(output_dir, exist_ok=True)
    
    # Jeśli mamy odpowiedzi, wybieramy najczęstsze
    if 'multiple_choice_answer' in df.columns:
        answer_counts = df['multiple_choice_answer'].value_counts()
        top_answers = answer_counts.head(num_classes).index.tolist()
        
        # Mapowanie odpowiedzi na indeksy
        answer_to_idx = {ans: idx for idx, ans in enumerate(top_answers)}
        idx_to_answer = {idx: ans for idx, ans in enumerate(top_answers)}
        
        # Zapisywanie mapowania
        with open(os.path.join(output_dir, 'answer_to_idx.json'), 'w') as f:
            json.dump(answer_to_idx, f)
        with open(os.path.join(output_dir, 'idx_to_answer.json'), 'w') as f:
            json.dump(idx_to_answer, f)
        
        # Funkcja mapująca odpowiedzi na indeksy (z obsługą nieznanych odpowiedzi)
        def map_answer(answer):
            return answer_to_idx.get(answer, num_classes)  # num_classes jako indeks dla nieznanych
        
        df['answer_idx'] = df['multiple_choice_answer'].apply(map_answer)
        
        # Informacje o mapowaniu
        print(f"Utworzono mapowanie dla {len(top_answers)} najczęstszych odpowiedzi.")
        print(f"Przykładowe mapowanie: {list(answer_to_idx.items())[:5]}")
    
    # Opcjonalnie - kopiowanie próbki obrazów do folderu processed dla łatwiejszego dostępu
    sample_img_dir = os.path.join(output_dir, 'sample_images')
    os.makedirs(sample_img_dir, exist_ok=True)
    
    # Wybór kilku obrazów do próbki (np. 10)
    sample_img_ids = df['image_id'].sample(10).tolist()
    
    for img_id in sample_img_ids:
        src_path = os.path.join(img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        dst_path = os.path.join(sample_img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        try:
            shutil.copy(src_path, dst_path)
            print(f"Skopiowano obraz {img_id} do folderu próbek.")
        except Exception as e:
            print(f"Błąd kopiowania obrazu {img_id}: {e}")
    
    # Zapisywanie przetworzonego zbioru danych
    df.to_csv(os.path.join(output_dir, 'vqa_processed.csv'), index=False)
    print(f"Zapisano przetworzony zbiór danych z {len(df)} rekordami.")
    
    return df

def main():
    # Ścieżki do folderów
    base_dir = "VQA_Portfolio"
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    images_dir = os.path.join(raw_data_dir, "train2014")
    
    # Sprawdzenie, czy foldery istnieją
    if not os.path.exists(raw_data_dir):
        print(f"Folder {raw_data_dir} nie istnieje. Upewnij się, że uruchomiłeś wcześniej data_download.py.")
        return
    
    # Wczytywanie danych
    df = load_data(raw_data_dir)
    
    # Przetwarzanie danych
    # Używamy mniejszej próbki danych dla przyspieszenia - możesz zwiększyć tę wartość
    sample_size = 5000
    print(f"\nPrzetwarzanie danych (próbka {sample_size} rekordów)...")
    processed_df = preprocess_data(
        df, 
        sample_size=sample_size, 
        save_path=os.path.join(processed_dir, "vqa_sample.csv")
    )
    
    # Analiza danych
    analyze_data(processed_df)
    
    # Przygotowanie danych do modelu
    prepare_data_for_model(
        processed_df, 
        output_dir=processed_dir, 
        img_dir=images_dir, 
        num_classes=1000
    )
    
    # Wyświetlenie przykładów
    print("\nWyświetlanie przykładowych obrazów z pytaniami i odpowiedziami...")
    if os.path.exists(images_dir):
        show_examples(processed_df, images_dir, num_examples=3)
    else:
        print(f"Folder {images_dir} nie istnieje. Nie można wyświetlić przykładów.")
    
    print("\nPrzetwarzanie danych zakończone!")

if __name__ == "__main__":
    main()
