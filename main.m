clear all; close all; clc;


%% Inicjalizacja etykiet i ścieżek
young_ids   = {'f1y01','f1y02','f1y03','f1y04','f1y05', ...
               'f1y06','f1y07','f1y08','f1y09','f1y10', ...
               'f2y01','f2y02','f2y03','f2y04','f2y05', ...
               'f2y06','f2y07','f2y08','f2y09','f2y10'};

elderly_ids = {'f1o01','f1o02','f1o03','f1o04','f1o05', ...
               'f1o06','f1o07','f1o08','f1o09','f1o10', ...
               'f2o01','f2o02','f2o03','f2o04','f2o05', ...
               'f2o06','f2o07','f2o08','f2o09','f2o10'};

db_path = 'fantasia-database-1.0.0/'; 



%% Wyciągnięcie cech i podział na 300s fragmenty
window_len_s = 300;
[X_young,   Y_young,   groups_young]   = extract_segmented_features(young_ids,   'young',   db_path, window_len_s, 0);
[X_elderly, Y_elderly, groups_elderly] = extract_segmented_features(elderly_ids, 'elderly', db_path, window_len_s, 20);

X      = [X_young; X_elderly];        
Y      = [Y_young; Y_elderly];        
groups = [groups_young; groups_elderly];

classes = categories(Y);

%% 3. Walidacja krzyżowa
num_patient_folds = 5;
all_Y_test_SVM = string([]); all_Y_pred_SVM = string([]);
all_Y_test_LDA = string([]); all_Y_pred_LDA = string([]);

[unique_groups, first_idx] = unique(groups);
unique_labels = Y(first_idx);

rng(42);
patient_cv = cvpartition(unique_labels, 'KFold', num_patient_folds, 'Stratify', true);

for k = 1:num_patient_folds
    train_patients = unique_groups(training(patient_cv, k));
    test_patients  = unique_groups(test(patient_cv, k));
    
    train_idx = ismember(groups, train_patients);
    test_idx  = ismember(groups, test_patients);
    
    X_train = X(train_idx, :); Y_train = Y(train_idx);
    X_test  = X(test_idx, :);  Y_test  = Y(test_idx);
    
    % Standaryzacja z-score
    mu = mean(X_train);
    sigma = std(X_train);
    X_train_z = (X_train - mu) ./ sigma;
    X_test_z  = (X_test  - mu) ./ sigma;
    
    %% SVM
    svm_model = fitcsvm(X_train_z, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint',  1, ...
        'KernelScale',    'auto', ...
        'Standardize',    false, ...
        'ClassNames',     classes);
    
    all_Y_test_SVM = [all_Y_test_SVM; string(Y_test)];
    all_Y_pred_SVM = [all_Y_pred_SVM; string(predict(svm_model, X_test_z))];
    
    %% LDA
    lda_model = fitcdiscr(X_train_z, Y_train, ...
        'DiscrimType',    'linear', ...
        'ClassNames',     classes);
    
    all_Y_test_LDA = [all_Y_test_LDA; string(Y_test)];
    all_Y_pred_LDA = [all_Y_pred_LDA; string(predict(lda_model, X_test_z))];
end

%% Podsumowanie wyników
acc_svm = mean(strcmp(all_Y_pred_SVM, all_Y_test_SVM)) * 100;
acc_lda = mean(strcmp(all_Y_pred_LDA, all_Y_test_LDA)) * 100;
fprintf('\nPodsumowanie\n');
fprintf('Średnia dokładność SVM (RBF): %.2f%%\n', acc_svm);
fprintf('Średnia dokładność LDA:       %.2f%%\n\n', acc_lda);

%% Generowanie wykresów PSD dla Welch i FFT

%young
rec_y = fullfile(db_path, 'f1y01');
[sig_y, fs_y] = rdsamp(rec_y); ann_y = rdann(rec_y, 'ecg');
r_peaks_y = ((0:length(sig_y)-1)/fs_y)'; r_peaks_y = r_peaks_y(ann_y);
r_seg_y = r_peaks_y(r_peaks_y >= 0 & r_peaks_y < window_len_s);
RR_y = diff(r_seg_y); t_RR_y = r_seg_y(2:end); t_int_y = t_RR_y(1):1/4:t_RR_y(end);
RR_int_y = detrend(interp1(t_RR_y, RR_y, t_int_y, 'spline'));
[~,~,~, f_fft_y, psd_fft_y] = psd_fft(RR_int_y, 4);
[~,~,~, f_wlch_y, psd_wlch_y] = psd_welch(RR_int_y, 4);

fig_young = figure('Name','Wykres Widmo - Młody', 'Units','centimeters','Position',[2 2 22 9]);
subplot(1,2,1); plot_psd_unified_local(f_fft_y, psd_fft_y, 'FFT');
subplot(1,2,2); plot_psd_unified_local(f_wlch_y, psd_wlch_y, 'Welch');
sgtitle('Porównanie metod estymacji PSD – młody pacjent (f1y01)', 'Interpreter', 'none');

%eldery
rec_o = fullfile(db_path, 'f1o01');
[sig_o, fs_o] = rdsamp(rec_o); ann_o = rdann(rec_o, 'ecg');
r_peaks_o = ((0:length(sig_o)-1)/fs_o)'; r_peaks_o = r_peaks_o(ann_o);
r_seg_o = r_peaks_o(r_peaks_o >= 0 & r_peaks_o < window_len_s);
RR_o = diff(r_seg_o); t_RR_o = r_seg_o(2:end); t_int_o = t_RR_o(1):1/4:t_RR_o(end);
RR_int_o = detrend(interp1(t_RR_o, RR_o, t_int_o, 'spline'));
[~,~,~, f_fft_o, psd_fft_o] = psd_fft(RR_int_o, 4);
[~,~,~, f_wlch_o, psd_wlch_o] = psd_welch(RR_int_o, 4);

fig_elderly = figure('Name','Wykres Widmo - Starszy', 'Units','centimeters','Position',[3 3 22 9]);
subplot(1,2,1); plot_psd_unified_local(f_fft_o, psd_fft_o, 'FFT');
subplot(1,2,2); plot_psd_unified_local(f_wlch_o, psd_wlch_o, 'Welch');
sgtitle('Porównanie metod estymacji PSD – starszy pacjent (f1o01)', 'Interpreter', 'none');

%% Wykresy końcowe

% Macierze pomyłek
fig1 = figure('Name','Porównanie Klasyfikatorów na Segmentach','Units','centimeters','Position',[5 5 26 12]);
all_Y_test_SVM_cat = categorical(all_Y_test_SVM);
all_Y_pred_SVM_cat = categorical(all_Y_pred_SVM);
all_Y_test_LDA_cat = categorical(all_Y_test_LDA);
all_Y_pred_LDA_cat = categorical(all_Y_pred_LDA);
subplot(1,2,1); confusionchart(all_Y_test_SVM_cat, all_Y_pred_SVM_cat, 'Title', 'Macierz Pomyłek: SVM', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
subplot(1,2,2); confusionchart(all_Y_test_LDA_cat, all_Y_pred_LDA_cat, 'Title', 'Macierz Pomyłek: LDA', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

% Główne widma mocy
fig2 = plot_hrv_spectra(db_path, window_len_s);

% Przestrzeń cech w skali logarytmicznej
fig3 = plot_features_scatter(X_young, X_elderly);

%% Zapis wykresów do sprawozdania
output_dir = 'wykresy';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Utworzono nowy folder eksportowy: "%s"\n', output_dir);
end

% hack: mieliśmy problem z niektórymi polskimi literami, wymuszenie
% czcionki Arial
set(findobj(fig1, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig2, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig3, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig_young, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig_elderly, '-property', 'FontName'), 'FontName', 'Arial');

fprintf('Eksport wykresów do folderu:"%s/"\n', output_dir);
exportgraphics(fig1, fullfile(output_dir, 'macierze_pomylek.pdf'), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(output_dir, 'widma_hrv_porownanie.pdf'), 'ContentType', 'vector');
exportgraphics(fig3, fullfile(output_dir, 'przestrzen_cech_scatter.pdf'), 'ContentType', 'vector');
exportgraphics(fig_young, fullfile(output_dir, 'wykres_widmo_young.pdf'), 'ContentType', 'vector');
exportgraphics(fig_elderly, fullfile(output_dir, 'wykres_widmo_elderly.pdf'), 'ContentType', 'vector');
fprintf('Zapisano wykresy!\n');


%% Funkcje pomocnicze
function [X_feat, Y_lab, G_num] = extract_segmented_features(ids, group_name, db_path, win_len, id_offset)
    X_feat = []; Y_lab = []; G_num = [];
    for i = 1:numel(ids)
        rec = fullfile(db_path, ids{i});
        [signal, fs] = rdsamp(rec);
        ecg = signal(:,2);
        ann = rdann(rec, 'ecg');
        time      = (0:length(ecg)-1) / fs;
        r_peaks_t = time(ann);
        total_duration = max(r_peaks_t);
        num_windows = floor(total_duration / win_len);
        
        for w = 0:(num_windows-1)
            t_start = w * win_len;
            t_end   = t_start + win_len;
            win_r_peaks = r_peaks_t(r_peaks_t >= t_start & r_peaks_t < t_end);
            
            if numel(win_r_peaks) >= 30
                win_r_peaks_zeroed = win_r_peaks - t_start;
                [LF, HF, ~] = compute_hrv_freq(win_r_peaks_zeroed);
                if LF > 0 && HF > 0
                    X_feat = [X_feat; log(LF), log(HF)];
                    Y_lab  = [Y_lab; categorical({group_name})];
                    G_num  = [G_num; i + id_offset];
                end
            end
        end
    end
end

function fig_handle = plot_hrv_spectra(db_path, win_len)
    examples = { fullfile(db_path,'f1y01'), 'Młody pacjent (f1y01)', [0 0.8 0.4]; ...
                 fullfile(db_path,'f1o01'), 'Starszy pacjent (f1o01)', [0 0.8 0.4] };
    fig_handle = figure('Name','Widma HRV dla segmentów 5-minutowych','Units','centimeters','Position',[2 2 22 9]);
    for k = 1:2
        rec   = examples{k,1}; lbl   = examples{k,2}; col   = examples{k,3};
        [signal, fs] = rdsamp(rec); ann = rdann(rec, 'ecg');
        time         = (0:length(signal)-1) / fs; r_peaks_t = time(ann);
        r_peaks_segment = r_peaks_t(r_peaks_t >= 0 & r_peaks_t < win_len);
        [~, ~, ~, f, PSD] = compute_hrv_freq(r_peaks_segment);
        
        subplot(1,2,k);
        area(f, PSD, 'FaceColor', col, 'FaceAlpha', 0.25, 'EdgeColor', col, 'LineWidth', 1.5);
        hold on;
        LF_mask = f >= 0.04 & f < 0.15; HF_mask = f >= 0.15 & f < 0.4; 
        area(f(LF_mask), PSD(LF_mask), 'FaceColor',[0.2 0.4 1], 'FaceAlpha',0.45, 'EdgeColor','none');
        area(f(HF_mask), PSD(HF_mask), 'FaceColor',[1 0.2 0.2], 'FaceAlpha',0.45, 'EdgeColor','none');
        xlim([0 0.5]); xlabel('Częstotliwość [Hz]'); ylabel('PSD [s^2/Hz]');
        title(lbl, 'Interpreter', 'none');
        legend('Moc widmowa (PSD)','Pasmo LF (0.04–0.15 Hz)','Pasmo HF (0.15–0.4 Hz)', 'Location','northeast', 'Interpreter', 'none');
        grid on;
    end
    sgtitle('Porównanie widm mocy HRV', 'Interpreter', 'none');
end

function fig_handle = plot_features_scatter(feat_young, feat_elderly)
    fig_handle = figure('Name','Przestrzeń cech HRV – segmenty','Units','centimeters','Position',[2 2 15 11]);
    scatter(feat_young(:,1),   feat_young(:,2),   35, 'o', 'filled', 'MarkerFaceColor',[0 0.7 0.3], 'MarkerFaceAlpha', 0.6, 'DisplayName','Young (Młodzi)');
    hold on;
    scatter(feat_elderly(:,1), feat_elderly(:,2), 35, 's', 'filled', 'MarkerFaceColor',[0.85 0.2 0.1], 'MarkerFaceAlpha', 0.6, 'DisplayName','Elderly (Seniorzy)');
    xlabel('Moc niskiej częstotliwości ln(LF) [ln(s^2)]'); ylabel('Moc wysokiej częstotliwości ln(HF) [ln(s^2)]');
    title('Rozkład 5-minutowych segmentów w przestrzeni cech'); legend('Location','best'); grid on;
end

function [LF, HF, LF_HF, f, PSD] = psd_fft(RR_interp, fs)
    N   = length(RR_interp); win = hann(N)';                    
    X   = fft(RR_interp .* win); PSD = (1/(fs*N)) * abs(X).^2;
    PSD = PSD(1:floor(N/2)+1); PSD(2:end-1) = 2*PSD(2:end-1);    
    f   = (0:floor(N/2)) * fs / N;
    LF_band = f >= 0.04 & f < 0.15; HF_band = f >= 0.15 & f < 0.4;
    LF = trapz(f(LF_band), PSD(LF_band)); HF = trapz(f(HF_band), PSD(HF_band)); LF_HF = LF / HF;
end

function [LF, HF, LF_HF, f, PSD] = psd_welch(RR_interp, fs)
    [PSD, f] = pwelch(RR_interp, [], [], [], fs);
    LF_band  = f >= 0.04 & f < 0.15; HF_band  = f >= 0.15 & f < 0.4;
    LF = trapz(f(LF_band), PSD(LF_band)); HF = trapz(f(HF_band), PSD(HF_band)); LF_HF = LF / HF;
end

function plot_psd_unified_local(f, PSD, method_name)
    LF_mask = f >= 0.04 & f < 0.15; HF_mask = f >= 0.15 & f < 0.40;
    green_col = [0 0.8 0.4];
    h = area(f, PSD, 'FaceColor', green_col, 'FaceAlpha', 0.15, 'EdgeColor', green_col, 'LineWidth', 1.5);
    hold on;
    h_lf_dummy = area(f(LF_mask), PSD(LF_mask), 'FaceColor', [0.2 0.4 1], 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'Visible', 'off');
    h_hf_dummy = area(f(HF_mask), PSD(HF_mask), 'FaceColor', [1 0.2 0.2], 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'Visible', 'off');
    area(f(LF_mask), PSD(LF_mask), 'FaceColor', [0.2 0.4 1], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
    area(f(HF_mask), PSD(HF_mask), 'FaceColor', [1 0.2 0.2], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
    xlim([0 0.5]); xlabel('Częstotliwość [Hz]', 'Interpreter', 'none'); ylabel('PSD [s^2/Hz]', 'Interpreter', 'none');
    title(['PSD – ', method_name], 'Interpreter', 'none');
    legend([h, h_lf_dummy, h_hf_dummy], {'Moc widmowa (PSD)', 'Pasmo LF (0.04–0.15 Hz)', 'Pasmo HF (0.15–0.4 Hz)'}, 'Location', 'northeast', 'Interpreter', 'none');
    grid on;
end