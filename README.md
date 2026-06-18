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
train_lenet5.py       trenuje sieć LeNet-5, rysuje krzywe uczenia i wyniki klasyfikacji
train_svm.py          to samo co train_lenet5.py, ale klasyfikator SVM (RBF)
requirements.txt      zależności
outputs/              wyniki (PDF, PNG, modele)
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

# 2a) wariant z siecią neuronową LeNet-5
python train_lenet5.py

# 2b) wariant z klasyfikatorem SVM (RBF) – ten sam zbiór, podział i wykresy
python train_svm.py
```

Najważniejsze opcje LeNet-5: `--per-class` (domyślnie 100), `--epochs`
(domyślnie 70), `--batch-size`, `--lr`, `--seed`.
Najważniejsze opcje SVM: `--per-class`, `--aug-copies` (domyślnie 20, liczba
dodatkowych augmentowanych kopii obrazu uczącego), `--kernel` (domyślnie
`linear`), `--C` (domyślnie 0.01), `--gamma`, `--seed`.

### LeNet-5 vs SVM

Oba skrypty używają tego samego zbioru i podziału 50/50 oraz generują krzywe
uczenia i te same demonstracje klasyfikacji (20 obrazów czystych + 20 z szumem).
Różni je tylko klasyfikator. W krzywej uczenia SVM oś X to liczba próbek
uczących (SVM nie ma epok). W zbiorze uczącym SVM dokładana jest augmentacja
(domyślnie 20 kopii na obraz) jako odpowiednik augmentacji on-the-fly z sieci.

**Unikanie overfittingu.** Jądro RBF z dużym `C` na surowych pikselach (1024
wymiary) szybko „zapamiętuje" mały zbiór uczący – dokładność uczenia przyklejona
do 100%, duża luka train–test. Dlatego domyślny SVM to **jądro liniowe z silną
regularizacją** (`C=0.01`) uczone na obficie augmentowanym zbiorze. Dzięki temu
krzywa uczenia jest „zdrowa": linia train schodzi poniżej 1.0 i maleje, test
rośnie, a obie krzywe się zbiegają (luka ~13 pp zamiast ~17 pp przy RBF). Test
utrzymuje się na poziomie ok. **80%**. Jądro RBF nadal jest dostępne przez
`--kernel rbf --C 10` (wyższy test ~83%, ale wyraźny overfitting train=100%).

## Pliki wynikowe (`outputs/`)

| plik | opis |
|---|---|
| `input_images.pdf` | jedna strona PDF z obrazami wejściowymi (wzorce + warianty) |
| `learning_curves.png` | krzywe uczenia (strata i dokładność, train/test) |
| `classification_clean.png` | 20 losowych obrazów z bazy + wynik klasyfikacji (LeNet-5) |
| `classification_noisy.png` | 20 losowych obrazów z bazy z dodanym szumem + klasyfikacja (LeNet-5) |
| `learning_curves_svm.png` | krzywe uczenia SVM (dokładność/błąd vs. liczba próbek) |
| `classification_clean_svm.png` | 20 losowych obrazów z bazy + klasyfikacja (SVM) |
| `classification_noisy_svm.png` | 20 losowych obrazów z bazy z szumem + klasyfikacja (SVM) |
| `dataset.npz` | wygenerowany zbiór (regenerowalny) |
| `lenet5_shapes.pt` | wagi wytrenowanej sieci (regenerowalne) |
| `svm_shapes.joblib` | wytrenowany model SVM (regenerowalny) |

## Uwagi o wynikach

Przy podziale 50/50 (po 50 obrazów uczących na klasę) sieć osiąga ok. **75%**
dokładności na zbiorze testowym, a ok. **70%** przy dodatkowo zaszumionych
obrazach (σ=45). Większość błędów dotyczy figur naturalnie podobnych przy małej
rozdzielczości i rozmyciu (koło ↔ pięciokąt ↔ sześciokąt). Wyniki są
powtarzalne dzięki ustawionym ziarnom losowości (`--seed`).
