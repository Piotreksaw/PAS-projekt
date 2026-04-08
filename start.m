clear all; close all; clc;

%% 1. Parametry i wczytywanie danych
recordName = 'f1y01'; % Nazwa pacjenta (f1y01 - young, f1e01 - elderly)

% rdsamp - wczytuje sygnał EKG
% [sygnal, fs] = rdsamp(nazwa_pliku, kanal)
[signal, fs] = rdsamp(recordName, 1); 

% rdann - wczytuje adnotacje (pozycje załamków R zaznaczone przez ekspertów)
% 'ecg' to rozszerzenie pliku z adnotacjami w bazie Fantasia
[ann, type, subtype, chan, num, comments] = rdann(recordName, 'ecg');

%% 2. Konwersja na jednostki czasu
time = (0:length(signal)-1) / fs; % wektor czasu w sekundach
r_peaks_samples = ann;            % numery próbek, w których są załamki R
r_peaks_time = time(ann);         % czas wystąpienia załamków R w sekundach

%% 3. Wizualizacja kontrolna
figure;
plot(time, signal);
hold on;
plot(r_peaks_time, signal(ann), 'ro', 'MarkerSize', 8); % Zaznaczenie R na czerwono
title(['Sygnał EKG z bazy Fantasia - Pacjent: ', recordName]);
xlabel('Czas [s]');
ylabel('Amplituda [mV]');
legend('EKG', 'Wykryte załamki R');
grid on;
xlim([0 500]); % Wyświetlamy tylko pierwsze 500 sekund dla czytelności