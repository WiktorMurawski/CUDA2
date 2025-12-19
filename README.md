# Dokumentacja wstępna projektu 2: Znajdowanie par wektorów binarnych o odległości Hamminga równej 1

## 1. Opis problemu

Zadanie polega na znalezieniu wszystkich par wektorów binarnych (ciągów bitów o stałej długości `l`) z danego zbioru `n` wektorów, takich że odległość Hamminga między nimi wynosi dokładnie 1 (różnią się dokładnie na jednej pozycji bitowej).

**Wejście:**
- Plik tekstowy zawierający na pierwszej linii dwie liczby oddzielone przecinkiem: `n` (liczba wektorów) i `l` (długość każdego wektora w bitach).
- Następnie `n` linii z ciągami składającymi się z `l` znaków `'0'` i `'1'`.

**Wyjście:**
- Liczba par wektorów binarnych z oryginalnego zbioru o odległość Hamminga = 1.
- Wyświetlone powyższe pary (dla opcji `verbose`)

## 2. Podejście sekwencyjne (CPU – istniejąca implementacja)

Istniejąca wersja CPU wykorzystuje **drzewo radixowe** (binary trie) do efektywnego przechowywania wektorów binarnych.

### Budowa drzewa (na CPU)
- Tworzone jest drzewo radixowe, gdzie każdy poziom odpowiada kolejnemu bitowi wektora (od indeksu 0).
- Każdy węzeł ma dwa dzieci: dla bitu `0` i `1`.
- Liście przechowują indeks oryginalnego wektora (lub `-1`, jeśli węzeł nie jest liściem).
- Wstawianie `n` wektorów: złożoność czasowa `O(n ⋅ l)`.

### Wyszukiwanie par (na CPU)
- Dla każdego wektora zapytania `v` (indeks `i`):
  - Generujemy `l` zmodyfikowanych wersji `v`, w których flipujemy dokładnie jeden bit (każdy po kolei).
  - Dla każdej takiej wersji wykonujemy dokładne wyszukiwanie w drzewie (przechodzenie po bitach).
  - Jeśli znaleziony wektor istnieje i jego indeks ≠ `i`, to jest to sąsiad.
- Zbieramy pary z `i < j`, aby uniknąć duplikatów.
- Złożoność: `O(nl²)`, ale ponieważ zazwyczaj `n > l` jest znacznie lepsza niż naiwne porównywanie wszystkich par (`O(n²l)`).

## 3. Planowane podejście równoległe (CUDA GPU)

Aby przyspieszyć obliczenia, stosujemy hybrydowe podejście:
- **Budowa drzewa radixowego na CPU** - złożoność obliczeniowa: `O(nl)`
- **Zrównoleglone wyszukiwanie sąsiadów na GPU** - złożoność obliczeniowa:`O(nl²)`, podział na `n` wątków

### Transfer danych na GPU
- Przeniesienie tablicy węzłów drzewa (`std::vector<RadixNode>`) do pamięci GPU.
- Przeniesienie tablicy bitów wektorów (`data.bits`).

### Kernel CUDA
- Uruchamiamy jeden wątek na każdy wektor zapytania (łącznie `n` wątków).
- Każdy wątek odpowiada za wektor o indeksie `i` i wykonuje:
  1. Pobranie swojego wektora `v`.
  2. Dla każdego z `l` możliwych flipów bitu `j` `(k = 0, ..., l-1)`:
     - Tymczasowe flipowanie bitu `j`.
     - Przechodzenie przez drzewo od korzenia zgodnie ze zmodyfikowanymi bitami.
     - Jeśli dojdzie do liścia z ważnym `vectorIndex` != `i`, dodaje parę `(i, foundIndex)` do globalnej listy wyników.
- Aby uniknąć duplikatów: para dodawana tylko gdy `i < foundIndex`.

### Zbieranie wyników
- Użycie atomowych operacji do inkrementacji licznika wyników i zapisu par do preallokowanej tablicy (górne oszacowanie liczby par: `nl / 2`).
- Alternatywnie: zapis wyników per-wątek do oddzielnych buforów, a następnie kompaktowanie.

### Planowane optymalizacje
- Wykorzystanie shared memory do przyspieszenia przechodzenia przez drzewo (jeśli będzie to możliwe).

## 4. Etapy implementacji

1. Alokacja pamięci na GPU i przeniesienie drzewa oraz danych wejściowych.
2. Implementacja kernela wyszukiwania sąsiadów.
3. Mechanizm zbierania wyników (operacje atomowe/zbieranie wyników częściowych).
4. Przeniesienie wyników z GPU na host oraz ewentualne sortowanie/usuwanie duplikatów.
5. Pomiar czasu przenoszenia drzewa na GPU oraz działania kernela za pomocą `cudaEvent`, pomiar całości obliczeń za pomocą `chrono` i porównanie z wersją CPU.
