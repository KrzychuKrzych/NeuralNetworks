# Visual Question Answering (VQA) Portfolio Project

Ten projekt stanowi implementację systemu Visual Question Answering (VQA) - modelu AI, który odpowiada na pytania dotyczące zawartości obrazów. Projekt został zrealizowany jako element portfolio w dziedzinie uczenia maszynowego i przetwarzania obrazów.

## Struktura projektu

```
VQA_Portfolio/
├── data_preparation/
│   ├── data_download.py - Pobieranie danych VQA
│   └── data_preprocessing.py - Przetwarzanie i przygotowanie danych
├── model/
│   ├── vqa_model.py - Definicja architektury modelu
│   └── training.py - Trening modelu
├── evaluation/
│   └── evaluate_model.py - Ewaluacja wydajności modelu
├── utils/
│   └── visualization.py - Wizualizacje i analizy
├── data/
│   ├── raw/ - Surowe dane pobrane z zewnętrznych źródeł
│   ├── processed/ - Przetworzone dane gotowe do treningu
│   ├── models/ - Zapisane modele i historia treningu
│   ├── results/ - Wyniki ewaluacji modelu
│   └── visualizations/ - Wygenerowane wizualizacje i dashboardy
└── README.md - Ten plik
```

## O projekcie

Visual Question Answering to zadanie AI, które łączy przetwarzanie obrazów (computer vision) z przetwarzaniem języka naturalnego (NLP). Model przyjmuje obraz oraz pytanie w języku naturalnym i generuje odpowiedź na to pytanie w oparciu o zawartość obrazu.

### Architektura modelu

Model VQA składa się z trzech głównych komponentów:

1. **Ekstraktor cech obrazu** - Wykorzystuje architekturę ResNet-50 do ekstrakcji cech z obrazów.
2. **Encoder tekstu** - Wykorzystuje model BERT do przetwarzania pytań.
3. **Moduł fuzji** - Łączy cechy z obu modalności (obraz i tekst) i wykonuje klasyfikację do przewidzenia najbardziej prawdopodobnej odpowiedzi.

### Zbiór danych

Projekt wykorzystuje zbiór danych VQA v2, który zawiera:
- Obrazy z kolekcji COCO
- Pytania dotyczące tych obrazów
- Adnotacje z odpowiedziami

## Uruchomienie projektu

### Wymagania

Przed uruchomieniem projektu upewnij się, że masz zainstalowane następujące biblioteki:

```
torch
torchvision
transformers
pandas
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

Możesz zainstalować wszystkie wymagane zależności korzystając z polecenia:

```bash
pip install -r requirements.txt
```

### Kroki do uruchomienia

1. **Pobieranie danych**:
   ```bash
   python data_preparation/data_download.py
   ```

2. **Przetwarzanie danych**:
   ```bash
   python data_preparation/data_preprocessing.py
   ```

3. **Trenowanie modelu**:
   ```bash
   python model/training.py
   ```

4. **Ewaluacja modelu**:
   ```bash
   python evaluation/evaluate_model.py
   ```

5. **Generowanie wizualizacji**:
   ```bash
   python utils/visualization.py
   ```

## Wyniki

Po uruchomieniu wszystkich skryptów, w katalogu `data/` znajdziesz następujące rezultaty:

- **Wytrenowane modele** w `data/models/`
- **Wyniki ewaluacji** w `data/results/`
- **Wizualizacje i dashboard** w `data/visualizations/`

## Integracja z Databricks

Projekt został zaprojektowany z myślą o możliwości przeniesienia go do środowiska Databricks. Aby przenieść projekt do Databricks:

1. Prześlij wszystkie pliki do repozytorium GitHub
2. W Databricks, przejdź do sekcji "Repos" i podłącz swoje repozytorium
3. Utwórz klaster z odpowiednimi bibliotekami (PyTorch, Transformers, itp.)
4. Wykonaj skrypty w kolejności przedstawionej powyżej

## Dalszy rozwój

Możliwe kierunki rozwoju projektu:

- Implementacja bardziej zaawansowanych architektur dla VQA
- Dodanie generacji odpowiedzi w formie zdań zamiast klasyfikacji
- Eksploracja technik uwagi (attention) dla lepszej interpretacji modelu
- Adaptacja modelu do specyficznych domen (np. analiza medyczna, rozpoznawanie produktów)

## Licencja

Ten projekt jest udostępniany na licencji MIT. Szczegóły znajdują się w pliku LICENSE.

## Autor

[Twoje Imię] - projekt portfolio z zakresu uczenia maszynowego i przetwarzania obrazów.
