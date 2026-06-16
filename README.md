# LeNet-5 – klasyfikacja figur konturowych (32×32, grayscale)

Sieć neuronowa typu **LeNet-5** (PyTorch) rozpoznająca 7 figur konturowych
renderowanych jako obrazy **32×32 piksele w skali szarości (0–255)**:

| glif | klasa | glif | klasa | glif | klasa |
|:---:|:---|:---:|:---|:---:|:---|
| ○ | circle (koło) | □ | square (kwadrat) | △ | triangle (trójkąt) |
| ◇ | diamond (romb) | ☆ | star (gwiazda) | ⬠ | pentagon (pięciokąt) |
| ⬡ | hexagon (sześciokąt) | | | | |

## Co realizuje projekt

1. **Generator danych** – dla każdej figury tworzy **po 100 różnych wariantów**
   z losowymi zakłóceniami: **szum**, **przesunięcie**, **rozmycie**,
   **pochylenie (shear)**, obrót, zmiana skali, grubości konturu i kontrastu.
   Figury rysowane są geometrycznie z antyaliasingiem (super-sampling), dzięki
   czemu obrazy są prawdziwie szaroskalowe (0–255), a nie binarne (0/1).
2. **Sieć LeNet-5** – klasyczna architektura LeCun et al. (C1→S2→C3→S4→C5→F6→out).
3. **Podział 50% / 50%** – z każdej klasy losowo 50 obrazów do uczenia i 50 do
   testowania (bez zbioru walidacyjnego).
4. **Krzywe uczenia** – wykres straty i dokładności (train vs. test) wg epok.
5. **Obrazy wejściowe** – zapisane jako **jedna strona PDF**.
6. **Wyniki klasyfikacji** – po 20 losowych obrazów z bazy (czyste) oraz 20 z
   dodanym szumem, z naniesioną etykietą prawdziwą i przewidzianą.

## Struktura

```
src/dataset.py        rysowanie figur, augmentacje, budowa i podział zbioru
src/model.py          definicja sieci LeNet-5
generate_dataset.py   generuje zbiór (dataset.npz) i PDF z obrazami wejściowymi
train_lenet5.py       trenuje sieć, rysuje krzywe uczenia i wyniki klasyfikacji
requirements.txt      zależności
outputs/              wyniki (PDF, PNG, model)
```

## Instalacja

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow matplotlib
```

## Uruchomienie

```bash
# 1) wygeneruj zbiór danych (700 obrazów) + PDF z obrazami wejściowymi
python generate_dataset.py

# 2) wytrenuj LeNet-5 i wygeneruj krzywe uczenia oraz wyniki klasyfikacji
python train_lenet5.py
```

Najważniejsze opcje: `--per-class` (domyślnie 100), `--epochs` (domyślnie 70),
`--batch-size`, `--lr`, `--seed`.

## Pliki wynikowe (`outputs/`)

| plik | opis |
|---|---|
| `input_images.pdf` | jedna strona PDF z obrazami wejściowymi (wzorce + warianty) |
| `learning_curves.png` | krzywe uczenia (strata i dokładność, train/test) |
| `classification_clean.png` | 20 losowych obrazów z bazy + wynik klasyfikacji |
| `classification_noisy.png` | 20 losowych obrazów z bazy z dodanym szumem + klasyfikacja |
| `dataset.npz` | wygenerowany zbiór (regenerowalny) |
| `lenet5_shapes.pt` | wagi wytrenowanej sieci (regenerowalne) |

## Uwagi o wynikach

Przy podziale 50/50 (po 50 obrazów uczących na klasę) sieć osiąga ok. **75%**
dokładności na zbiorze testowym, a ok. **70%** przy dodatkowo zaszumionych
obrazach (σ=45). Większość błędów dotyczy figur naturalnie podobnych przy małej
rozdzielczości i rozmyciu (koło ↔ pięciokąt ↔ sześciokąt). Wyniki są
powtarzalne dzięki ustawionym ziarnom losowości (`--seed`).
