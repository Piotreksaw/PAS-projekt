clear all; close all; clc;

%% =========================================================
%  Porównanie metod PSD: FFT vs Welch (Segmentacja 5-min)
%  dla klasyfikacji HRV young vs elderly (Fantasia Database)
% =========================================================
db_path     = 'fantasia-database-1.0.0/';
young_ids   = {'f1y01','f1y02','f1y03','f1y04','f1y05', ...
               'f1y06','f1y07','f1y08','f1y09','f1y10'};
elderly_ids = {'f1o01','f1o02','f1o03','f1o04','f1o05', ...
               'f1o06','f1o07','f1o08','f1o09','f1o10'};
all_ids    = [young_ids,   elderly_ids];
all_labels = [repmat({'young'},1,10), repmat({'elderly'},1,10)];

%% 1. Ekstrakcja cech z wycinka 5-minutowego (300 sekund)
window_len_s = 300; % Dokładnie 5 minut, tak jak w kodzie głównym

N = numel(all_ids);
feats_fft   = nan(N, 2);   % Przechowujemy [log(LF), log(HF)]
feats_welch = nan(N, 2);

fprintf('%-10s  %6s %6s  |  %6s %6s\n', 'Rekord', 'log_LF_f', 'log_HF_f', 'log_LF_w', 'log_HF_w');
fprintf('%s\n', repmat('-',1,50));

for i = 1:N
    rec = fullfile(db_path, all_ids{i});
    try
        [signal, fs] = rdsamp(rec); 
        ann          = rdann(rec, 'ecg'); 
        time         = (0:length(signal)-1) / fs;
        r_peaks_t    = time(ann); 
        
        % OGRANICZENIE DO PRZEBIEGU 5 MINUT
        r_peaks_segment = r_peaks_t(r_peaks_t >= 0 & r_peaks_t < window_len_s);
        
        % Preprocessing dla wycinka
        RR        = diff(r_peaks_segment);
        t_RR      = r_peaks_segment(2:end);
        fs_interp = 4;
        t_interp  = t_RR(1):1/fs_interp:t_RR(end);
        RR_interp = interp1(t_RR, RR, t_interp, 'spline');
        RR_interp = detrend(RR_interp);
        
        % --- FFT ---
        [LF_f, HF_f, ~, ~, ~] = psd_fft(RR_interp, fs_interp);
        % --- Welch ---
        [LF_w, HF_w, ~, ~, ~] = psd_welch(RR_interp, fs_interp);
        
        % Zapisujemy zlogarytmowane cechy (ln)
        feats_fft(i,:)   = [log(LF_f),  log(HF_f)];
        feats_welch(i,:) = [log(LF_w),  log(HF_w)];
        
        fprintf('%-10s  %6.4f %6.4f  |  %6.4f %6.4f\n', ...
            all_ids{i}, log(LF_f), log(HF_f), log(LF_w), log(HF_w));
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

%% 3. Generowanie wykresów widmowych na 5-minutowych przebiegach
% --- PACJENT MŁODY (f1y01) ---
rec_y = fullfile(db_path, 'f1y01');
[sig_y, fs_y] = rdsamp(rec_y); ann_y = rdann(rec_y, 'ecg'); 
r_peaks_y = ((0:length(sig_y)-1)/fs_y)'; r_peaks_y = r_peaks_y(ann_y);
r_seg_y = r_peaks_y(r_peaks_y >= 0 & r_peaks_y < window_len_s);
RR_y = diff(r_seg_y); t_RR_y = r_seg_y(2:end);
t_int_y = t_RR_y(1):1/4:t_RR_y(end);
RR_int_y = detrend(interp1(t_RR_y, RR_y, t_int_y, 'spline'));
[~,~,~, f_fft_y, psd_fft_y] = psd_fft(RR_int_y, 4);
[~,~,~, f_wlch_y, psd_wlch_y] = psd_welch(RR_int_y, 4);

fig_young = figure('Name','Wykres Widmo - Młody', 'Units','centimeters','Position',[2 2 22 9]);
subplot(1,2,1); plot_psd_unified(f_fft_y, psd_fft_y, 'FFT (periodogram)');
subplot(1,2,2); plot_psd_unified(f_wlch_y, psd_wlch_y, 'Welch');
sgtitle('Porównanie metod estymacji PSD – młody pacjent (f1y01, segment 5-min)', 'Interpreter', 'none');

% --- PACJENT STARSZY (f1o01) ---
rec_o = fullfile(db_path, 'f1o01');
[sig_o, fs_o] = rdsamp(rec_o); ann_o = rdann(rec_o, 'ecg'); 
r_peaks_o = ((0:length(sig_o)-1)/fs_o)'; r_peaks_o = r_peaks_o(ann_o);
r_seg_o = r_peaks_o(r_peaks_o >= 0 & r_peaks_o < window_len_s);
RR_o = diff(r_seg_o); t_RR_o = r_seg_o(2:end);
t_int_o = t_RR_o(1):1/4:t_RR_o(end);
RR_int_o = detrend(interp1(t_RR_o, RR_o, t_int_o, 'spline'));
[~,~,~, f_fft_o, psd_fft_o] = psd_fft(RR_int_o, 4);
[~,~,~, f_wlch_o, psd_wlch_o] = psd_welch(RR_int_o, 4);

fig_elderly = figure('Name','Wykres Widmo - Starszy', 'Units','centimeters','Position',[3 3 22 9]);
subplot(1,2,1); plot_psd_unified(f_fft_o, psd_fft_o, 'FFT (periodogram)');
subplot(1,2,2); plot_psd_unified(f_wlch_o, psd_wlch_o, 'Welch');
sgtitle('Porównanie metod estymacji PSD – starszy pacjent (f1o01, segment 5-min)', 'Interpreter', 'none');

%% 4. Wykres słupkowy Accuracy
fig_accuracy = figure('Name','Accuracy FFT vs Welch', ...
                      'Units','centimeters','Position',[2 2 10 8]);
bar([acc_fft, acc_welch], 0.5, 'FaceColor','flat', ...
    'CData', [0 0.8 0.4; 0 0.8 0.4]); 
xticklabels({'FFT','Welch'});
ylabel('Accuracy [%]', 'Interpreter', 'none');
ylim([0 100]);
title('Skuteczność klasyfikacji SVM', 'Interpreter', 'none');
grid on; box off;
for k = 1:2
    val = [acc_fft, acc_welch];
    text(k, val(k)+2, sprintf('%.1f%%', val(k)), ...
        'HorizontalAlignment','center','FontWeight','bold', 'Interpreter', 'none');
end

%% 5. Macierze pomyłek obok siebie
fig_cm = figure('Name','Macierze pomyłek', ...
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

%% --- EXPORT DO PLIKÓW PDF DLA LATEXA ---
output_dir = 'wykresy';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Utworzono folder: "%s"\n', output_dir);
end

set(findobj(fig_young, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig_elderly, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig_accuracy, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig_cm, '-property', 'FontName'), 'FontName', 'Arial');

fprintf('Zapisywanie wykresów metodycznych do plików .pdf...\n');
exportgraphics(fig_young, fullfile(output_dir, 'wykres_widmo_young.pdf'), 'ContentType', 'vector');
exportgraphics(fig_elderly, fullfile(output_dir, 'wykres_widmo_elderly.pdf'), 'ContentType', 'vector');
exportgraphics(fig_accuracy, fullfile(output_dir, 'metody_accuracy_bar.pdf'), 'ContentType', 'vector');
exportgraphics(fig_cm, fullfile(output_dir, 'metody_macierze_pomylek.pdf'), 'ContentType', 'vector');
fprintf('Wszystkie pliki PDF zostały pomyślnie zapisane w folderze "%s/".\n', output_dir);


%% =========================================================
%  FUNKCJE LOKALNE
% =========================================================
function [LF, HF, LF_HF, f, PSD] = psd_fft(RR_interp, fs)
    N   = length(RR_interp);
    win = hann(N)';                     
    X   = fft(RR_interp .* win);
    PSD = (1/(fs*N)) * abs(X).^2;
    PSD = PSD(1:floor(N/2)+1);
    PSD(2:end-1) = 2*PSD(2:end-1);     
    f   = (0:floor(N/2)) * fs / N;
    LF_band = f >= 0.04 & f < 0.15;
    HF_band = f >= 0.15 & f < 0.4;
    LF      = trapz(f(LF_band), PSD(LF_band));
    HF      = trapz(f(HF_band), PSD(HF_band));
    LF_HF   = LF / HF;
end

% ---------------------------------------------------------
function [LF, HF, LF_HF, f, PSD] = psd_welch(RR_interp, fs)
    [PSD, f] = pwelch(RR_interp, [], [], [], fs);
    LF_band  = f >= 0.04 & f < 0.15;
    HF_band  = f >= 0.15 & f < 0.4;
    LF       = trapz(f(LF_band), PSD(LF_band));
    HF       = trapz(f(HF_band), PSD(HF_band));
    LF_HF    = LF / HF;
end

% ---------------------------------------------------------
function [acc, cm] = classify_svm(feats, Y)
    valid = ~any(isnan(feats), 2);
    X     = feats(valid, :);
    Yv    = Y(valid);
    mu    = mean(X); sigma = std(X);
    Xz    = (X - mu) ./ sigma;
    cv  = cvpartition(Yv, 'KFold', 10, 'Stratify', true);
    mdl = fitcsvm(Xz, Yv, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', 1, 'CVPartition', cv);
    Y_pred = kfoldPredict(mdl);
    acc    = mean(Y_pred == Yv) * 100;
    cm     = confusionmat(Yv, Y_pred);
end

% ---------------------------------------------------------
function Y_pred = predict_all(feats, Y)
    valid  = ~any(isnan(feats), 2);
    X      = feats(valid, :);
    Yv     = Y(valid);
    mu     = mean(X);  sigma = std(X);
    Xz     = (X - mu) ./ sigma;
    mdl    = fitcsvm(Xz, Yv, 'KernelFunction','rbf','KernelScale','auto');
    Y_pred = predict(mdl, Xz);
end

% ---------------------------------------------------------
% NOWA FUNKCJA: Rysuje zunifikowany wykres, gdzie pasma stanowią integralną część jednej warstwy
function plot_psd_unified(f, PSD, method_name)
    % 1. Tworzymy maski dla poszczególnych pasm częstotliwości
    LF_mask = f >= 0.04 & f < 0.15;
    HF_mask = f >= 0.15 & f < 0.40;
    
    % 2. Rysujemy bazowy wykres obszarowy (cały na zielono)
    green_col = [0 0.8 0.4]; 
    h = area(f, PSD, 'FaceColor', green_col, 'FaceAlpha', 0.15, ...
         'EdgeColor', green_col, 'LineWidth', 1.5);
    hold on;
    
    % Modyfikacja właściwości graficznych area (dostęp do surowych wierzchołków)
    % Pobieramy domyślne kolory RGB dla każdego punktu na osi X
    drawnow; % Wymuszenie przetworzenia grafiki przez MATLAB
    sampled_colors = repmat(green_col, length(f), 1);
    
    % Podmieniamy kolory wewnątrz odpowiednich indeksów tej samej macierzy wierzchołków
    sampled_colors(LF_mask, :) = repmat([0.2 0.4 1], sum(LF_mask), 1); % Niebieski dla LF
    sampled_colors(HF_mask, :) = repmat([1 0.2 0.2], sum(HF_mask), 1); % Czerwony dla HF
    
    % Do poprawnego działania przezroczystości wielokolorowych powierzchni w MATLAB
    % tworzymy dwa dodatkowe, przezroczyste wykresy area wyłącznie w celach legendy,
    % aby nie psuć głównego przebiegu funkcji area.
    h_lf_dummy = area(f(LF_mask), PSD(LF_mask), 'FaceColor', [0.2 0.4 1], 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'Visible', 'off');
    h_hf_dummy = area(f(HF_mask), PSD(HF_mask), 'FaceColor', [1 0.2 0.2], 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'Visible', 'off');
    
    % Nadpisujemy kolory w głównym wykresie (naprawia odcięcia i łączy warstwy)
    area(f(LF_mask), PSD(LF_mask), 'FaceColor', [0.2 0.4 1], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
    area(f(HF_mask), PSD(HF_mask), 'FaceColor', [1 0.2 0.2], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
    
    xlim([0 0.5]);
    xlabel('Częstotliwość [Hz]', 'Interpreter', 'none');
    ylabel('PSD [s^2/Hz]', 'Interpreter', 'none');
    title(['PSD – ', method_name], 'Interpreter', 'none');
    
    legend([h, h_lf_dummy, h_hf_dummy], ...
           {'Moc widmowa (PSD)', 'Pasmo LF (0.04–0.15 Hz)', 'Pasmo HF (0.15–0.4 Hz)'}, ...
           'Location', 'northeast', 'Interpreter', 'none');
    grid on;
end