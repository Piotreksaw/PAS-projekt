clear all; close all; clc;

%% 1. Parametry i wczytywanie danych
recordName = 'fantasia-database-1.0.0/f1y01'; % Nazwa pacjenta (f1y01 - young, f1e01 - elderly)

% rdsamp - wczytuje sygnał EKG
% [sygnal, fs] = rdsamp(nazwa_pliku, kanal)
[signal, fs] = rdsamp(recordName); 

ecg = signal(:,2); % zamiast 1
% rdann - wczytuje adnotacje (pozycje załamków R zaznaczone przez ekspertów)
% 'ecg' to rozszerzenie pliku z adnotacjami w bazie Fantasia
[ann, type, subtype, chan, num, comments] = rdann(recordName, 'ecg');

%% 2. Konwersja na jednostki czasu
time = (0:length(ecg)-1) / fs; % wektor czasu w sekundach
r_peaks_samples = ann;            % numery próbek, w których są załamki R
r_peaks_time = time(ann);         % czas wystąpienia załamków R w sekundach

%% 3. Wizualizacja kontrolna
figure;
plot(time, ecg);
hold on;
plot(r_peaks_time, ecg(ann), 'ro', 'MarkerSize', 8); % Zaznaczenie R na czerwono
title(['Sygnał EKG z bazy Fantasia - Pacjent: ', recordName]);
xlabel('Czas [s]');
ylabel('Amplituda [mV]');
legend('EKG', 'Wykryte załamki R');
grid on;
xlim([0 10]); % Wyświetlamy tylko pierwsze 10 sekund dla czytelności

%% wyznaczenie parametrow LF, HF i LF/HF
[LF, HF, LF_HF, f, PSD] = compute_hrv_freq(r_peaks_time);

fprintf('LF = %.4f\n', LF);
fprintf('HF = %.4f\n', HF);
fprintf('LF/HF = %.4f\n', LF_HF);

figure;
plot(f, PSD);
xlim([0 0.5]);
xlabel('Frequency [Hz]');
ylabel('PSD');
title('HRV Power Spectrum');
grid on;