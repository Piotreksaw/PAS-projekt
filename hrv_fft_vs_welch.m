clear all; close all; clc;

%% =========================================================
%  Porównanie metod PSD: FFT vs Welch
%  dla klasyfikacji HRV young vs elderly (Fantasia Database)
% =========================================================

db_path     = 'fantasia-database-1.0.0/';
young_ids   = {'f1y01','f1y02','f1y03','f1y04','f1y05', ...
               'f1y06','f1y07','f1y08','f1y09','f1y10'};
elderly_ids = {'f1o01','f1o02','f1o03','f1o04','f1o05', ...
               'f1o06','f1o07','f1o08','f1o09','f1o10'};

all_ids    = [young_ids,   elderly_ids];
all_labels = [repmat({'young'},1,10), repmat({'elderly'},1,10)];

%% 1. Ekstrakcja cech obiema metodami
N = numel(all_ids);
feats_fft   = nan(N, 3);   % [LF, HF, LF_HF]
feats_welch = nan(N, 3);

fprintf('%-10s  %6s %6s %7s  |  %6s %6s %7s\n', ...
    'Rekord','LF_fft','HF_fft','LF/HF','LF_wlch','HF_wlch','LF/HF');
fprintf('%s\n', repmat('-',1,62));

for i = 1:N
    rec = fullfile(db_path, all_ids{i});
    try
        [signal, fs] = rdsamp(rec);
        ann          = rdann(rec, 'ecg');
        time         = (0:length(signal)-1) / fs;
        r_peaks_t    = time(ann);

        % Wspólny preprocessing
        RR        = diff(r_peaks_t);
        t_RR      = r_peaks_t(2:end);
        fs_interp = 4;
        t_interp  = t_RR(1):1/fs_interp:t_RR(end);
        RR_interp = interp1(t_RR, RR, t_interp, 'spline');
        RR_interp = detrend(RR_interp);

        % --- FFT ---
        [LF_f, HF_f, LFHF_f, ~, ~] = psd_fft(RR_interp, fs_interp);

        % --- Welch ---
        [LF_w, HF_w, LFHF_w, ~, ~] = psd_welch(RR_interp, fs_interp);

        feats_fft(i,:)   = [LF_f,  HF_f,  LFHF_f];
        feats_welch(i,:) = [LF_w,  HF_w,  LFHF_w];

        fprintf('%-10s  %6.4f %6.4f %7.4f  |  %6.4f %6.4f %7.4f\n', ...
            all_ids{i}, LF_f, HF_f, LFHF_f, LF_w, HF_w, LFHF_w);

    catch ME
        warning('Błąd dla %s: %s', all_ids{i}, ME.message);
    end
end

%% 2. Klasyfikacja SVM dla obu metod
Y = categorical(all_labels');

rng(42);
fprintf('\n=== Klasyfikacja SVM – 10-fold cross-validation ===\n');

[acc_fft,   cm_fft]   = classify_svm(feats_fft,   Y);
[acc_welch, cm_welch] = classify_svm(feats_welch, Y);

fprintf('Accuracy FFT:   %.1f%%\n', acc_fft);
fprintf('Accuracy Welch: %.1f%%\n', acc_welch);

%% 3. Porównanie widm FFT vs Welch – jeden przykład
example_rec = fullfile(db_path, 'f1y01');
[signal, fs] = rdsamp(example_rec);
ann          = rdann(example_rec, 'ecg');
time         = (0:length(signal)-1) / fs;
r_peaks_t    = time(ann);

RR        = diff(r_peaks_t);
t_RR      = r_peaks_t(2:end);
fs_interp = 4;
t_interp  = t_RR(1):1/fs_interp:t_RR(end);
RR_interp = interp1(t_RR, RR, t_interp, 'spline');
RR_interp = detrend(RR_interp);

[~,~,~, f_fft,   psd_fft_v]   = psd_fft(RR_interp,   fs_interp);
[~,~,~, f_welch, psd_welch_v] = psd_welch(RR_interp, fs_interp);

figure('Name','FFT vs Welch – widmo HRV', ...
       'Units','centimeters','Position',[2 2 22 9]);

subplot(1,2,1);
plot_psd(f_fft, psd_fft_v, 'FFT (periodogram)', [0.2 0.4 0.9]);

subplot(1,2,2);
plot_psd(f_welch, psd_welch_v, 'Welch', [0.1 0.7 0.3]);

sgtitle('Porównanie metod estymacji PSD – f1y01');

%% 4. Wykres słupkowy Accuracy
figure('Name','Accuracy FFT vs Welch', ...
       'Units','centimeters','Position',[2 2 10 8]);
bar([acc_fft, acc_welch], 0.5, 'FaceColor','flat', ...
    'CData', [0.2 0.4 0.9; 0.1 0.7 0.3]);
xticklabels({'FFT','Welch'});
ylabel('Accuracy [%]');
ylim([0 100]);
title('Skuteczność klasyfikacji SVM');
grid on; box off;
for k = 1:2
    val = [acc_fft, acc_welch];
    text(k, val(k)+2, sprintf('%.1f%%', val(k)), ...
        'HorizontalAlignment','center','FontWeight','bold');
end

%% 5. Macierze pomyłek obok siebie
figure('Name','Macierze pomyłek', ...
       'Units','centimeters','Position',[2 2 24 9]);

subplot(1,2,1);
confusionchart(Y, predict_all(feats_fft, Y), ...
    'Title','Macierz pomyłek – FFT', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

subplot(1,2,2);
confusionchart(Y, predict_all(feats_welch, Y), ...
    'Title','Macierz pomyłek – Welch', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% =========================================================
%  FUNKCJE LOKALNE
% =========================================================

function [LF, HF, LF_HF, f, PSD] = psd_fft(RR_interp, fs)
%PSD_FFT  Periodogram (klasyczne FFT) do estymacji PSD.
    N   = length(RR_interp);
    win = hann(N)';                     % okno Hanna – redukuje przeciek
    X   = fft(RR_interp .* win);
    PSD = (1/(fs*N)) * abs(X).^2;
    PSD = PSD(1:floor(N/2)+1);
    PSD(2:end-1) = 2*PSD(2:end-1);     % jednostronne
    f   = (0:floor(N/2)) * fs / N;

    LF_band = f >= 0.04 & f < 0.15;
    HF_band = f >= 0.15 & f < 0.4;
    LF      = trapz(f(LF_band), PSD(LF_band));
    HF      = trapz(f(HF_band), PSD(HF_band));
    LF_HF   = LF / HF;
end

% ---------------------------------------------------------

function [LF, HF, LF_HF, f, PSD] = psd_welch(RR_interp, fs)
%PSD_WELCH  Metoda Welcha (uśrednianie segmentów).
    [PSD, f] = pwelch(RR_interp, [], [], [], fs);
    LF_band  = f >= 0.04 & f < 0.15;
    HF_band  = f >= 0.15 & f < 0.4;
    LF       = trapz(f(LF_band), PSD(LF_band));
    HF       = trapz(f(HF_band), PSD(HF_band));
    LF_HF    = LF / HF;
end

% ---------------------------------------------------------

function [acc, cm] = classify_svm(feats, Y)
%CLASSIFY_SVM  SVM RBF z 10-fold CV; zwraca accuracy i macierz pomyłek.
    valid = ~any(isnan(feats), 2);
    X     = feats(valid, :);
    Yv    = Y(valid);

    % Standaryzacja
    mu    = mean(X);
    sigma = std(X);
    Xz    = (X - mu) ./ sigma;

    cv  = cvpartition(Yv, 'KFold', 10, 'Stratify', true);
    mdl = fitcsvm(Xz, Yv, ...
        'KernelFunction', 'rbf', ...
        'KernelScale',    'auto', ...
        'BoxConstraint',  1, ...
        'CVPartition',    cv);

    Y_pred = kfoldPredict(mdl);
    acc    = mean(Y_pred == Yv) * 100;
    cm     = confusionmat(Yv, Y_pred);
end

% ---------------------------------------------------------

function Y_pred = predict_all(feats, Y)
%PREDICT_ALL  Trenuje na wszystkich danych, zwraca predykcje (do macierzy pomyłek).
    valid  = ~any(isnan(feats), 2);
    X      = feats(valid, :);
    Yv     = Y(valid);
    mu     = mean(X);  sigma = std(X);
    Xz     = (X - mu) ./ sigma;
    mdl    = fitcsvm(Xz, Yv, 'KernelFunction','rbf','KernelScale','auto');
    Y_pred = predict(mdl, Xz);
end

% ---------------------------------------------------------

function plot_psd(f, PSD, method_name, col)
%PLOT_PSD  Rysuje widmo z zaznaczonymi pasmami LF i HF.
    area(f, PSD, 'FaceColor', col, 'FaceAlpha', 0.15, ...
         'EdgeColor', col, 'LineWidth', 1.5);
    hold on;
    LF_mask = f >= 0.04 & f < 0.15;
    HF_mask = f >= 0.15 & f < 0.4;
    area(f(LF_mask), PSD(LF_mask), 'FaceColor',[0.1 0.3 1], ...
         'FaceAlpha', 0.5, 'EdgeColor','none');
    area(f(HF_mask), PSD(HF_mask), 'FaceColor',[1 0.2 0.1], ...
         'FaceAlpha', 0.5, 'EdgeColor','none');
    xlim([0 0.5]);
    xlabel('Częstotliwość [Hz]');
    ylabel('PSD [s²/Hz]');
    title(['PSD – ', method_name]);
    legend('PSD','LF (0.04–0.15 Hz)','HF (0.15–0.4 Hz)', ...
           'Location','northeast');
    grid on;
end
