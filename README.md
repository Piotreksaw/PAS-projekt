# PAS-projekt

## 1. Cel projektu

Głównym celem niniejszego projektu jest przeprowadzenie zaawansowanej analizy częstotliwościowej sygnału zmienności rytmu zatokowego (ang. \textit{Heart Rate Variability} – HRV)~. Projekt zakłada identyfikację istotnych statystycznie różnic w parametrach widmowych pomiędzy osobami młodymi a seniorami. Efektem końcowym ma być stworzenie stabilnego modelu klasyfikacyjnego, zdolnego do automatycznego rozróżniania grup wiekowych na podstawie cech wyekstrahowanych z sygnału EKG.

## 2. Wymagania
- Baza danych:
    
    Wykorzystanie zbioru danych **Fantasia Database** (PhysioNet), zawierającego zbiór 120-minutowych zapisów EKG osób zdrowych w grupach: _Young_ (21–34 lata) oraz _Elderly_ (68–85 lat). Częstotliwość próbkowania sygnału wynosi $f_s = 250$ Hz.

- Dziedzina analizy:
    
    Skupienie się na parametrach częstotliwościowych (widmowych).
- Pasma częstotliwości:

    Analiza mocy w zakresach $LF$ (0.04--0.15 Hz) oraz $HF$ (0.15--0.4 Hz).
- Narzędzia:

    Implementacja algorytmów w środowisku MATLAB z wykorzystaniem dedykowanych bibliotek do przetwarzania sygnałów i uczenia maszynowego.
