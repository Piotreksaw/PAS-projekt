clear all; close all; clc;

%% =========================================================
%  HRV Classification – Fantasia Database (Wersja z Segmentacją)
%  Podział na okna 5-minutowe + Walidacja Group K-Fold (Bezpieczna)
% =========================================================

%% 1. Definicja rekordów
young_ids   = {'f1y01','f1y02','f1y03','f1y04','f1y05', ...
               'f1y06','f1y07','f1y08','f1y09','f1y10', ...
               'f2y01','f2y02','f2y03','f2y04','f2y05', ...
               'f2y06','f2y07','f2y08','f2y09','f2y10'};

elderly_ids = {'f1o01','f1o02','f1o03','f1o04','f1o05', ...
               'f1o06','f1o07','f1o08','f1o09','f1o10', ...
               'f2o01','f2o02','f2o03','f2o04','f2o05', ...
               'f2o06','f2o07','f2o08','f2o09','f2o10'};

db_path = 'fantasia-database-1.0.0/';  

%% 2. Ekstrakcja cech z podziałem na segmenty
window_len_s = 300; % 5 minut = 300 sekund (Standard dla short-term HRV)
fprintf('Rozpoczynam segmentację i ekstrakcję cech (okna %d-sekundowe)...\n', window_len_s);
[X_young,   Y_young,   groups_young]   = extract_segmented_features(young_ids,   'young',   db_path, window_len_s, 0);
[X_elderly, Y_elderly, groups_elderly] = extract_segmented_features(elderly_ids, 'elderly', db_path, window_len_s, 20);

% Łączenie bazy (Grupy pomagają nam odróżnić, z którego pacjenta pochodzi dany wycinek)
X      = [X_young; X_elderly];         % Macierz cech [Wszystkie_okienka x 2] (LF i HF)
Y      = [Y_young; Y_elderly];         % Etykiety klasy [Wszystkie_okienka x 1]
groups = [groups_young; groups_elderly]; % ID pacjenta dla każdego okienka [Wszystkie_okienka x 1]

classes = categories(Y);
fprintf('Wyekstrahowano łącznie %d segmentów od %d pacjentów.\n', height(X), numel(unique(groups)));

%% 3. Walidacja krzyżowa oparta na grupach (Group K-Fold CV)
num_patient_folds = 5; 
all_Y_test_SVM = string([]); all_Y_pred_SVM = string([]);
all_Y_test_LDA = string([]); all_Y_pred_LDA = string([]);

% Do stratyfikacji pacjentów: wyciągamy unikalnych pacjentów i ich etykiety
[unique_groups, first_idx] = unique(groups);
unique_labels = Y(first_idx);

rng(42); % Powtarzalność
patient_cv = cvpartition(unique_labels, 'KFold', num_patient_folds, 'Stratify', true);

for k = 1:num_patient_folds
    % Wyznaczamy, którzy pacjenci idą do treningu, a którzy do testu
    train_patients = unique_groups(training(patient_cv, k));
    test_patients  = unique_groups(test(patient_cv, k));
    
    % Mapujemy pacjentów na odpowiadające im 5-minutowe segmenty w głównej macierzy
    train_idx = ismember(groups, train_patients);
    test_idx  = ismember(groups, test_patients);
    
    X_train = X(train_idx, :); Y_train = Y(train_idx);
    X_test  = X(test_idx, :);  Y_test  = Y(test_idx);
    
    %% Standaryzacja z-score (BEZ wycieku danych)
    mu = mean(X_train);
    sigma = std(X_train);
    X_train_z = (X_train - mu) ./ sigma;
    X_test_z  = (X_test  - mu) ./ sigma;
    
    %% --- Klasyfikator 1: SVM (RBF) ---
    svm_model = fitcsvm(X_train_z, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint',  1, ...
        'KernelScale',    'auto', ...
        'Standardize',    false, ...
        'ClassNames',     classes);
    
    all_Y_test_SVM = [all_Y_test_SVM; string(Y_test)];
    all_Y_pred_SVM = [all_Y_pred_SVM; string(predict(svm_model, X_test_z))];
    
    %% --- Klasyfikator 2: LDA ---
    lda_model = fitcdiscr(X_train_z, Y_train, ...
        'DiscrimType',    'linear', ...
        'ClassNames',     classes);
    
    all_Y_test_LDA = [all_Y_test_LDA; string(Y_test)];
    all_Y_pred_LDA = [all_Y_pred_LDA; string(predict(lda_model, X_test_z))];
end

%% 4. Wyświetlenie poprawnych statystyk końcowych
acc_svm = mean(strcmp(all_Y_pred_SVM, all_Y_test_SVM)) * 100;
acc_lda = mean(strcmp(all_Y_pred_LDA, all_Y_test_LDA)) * 100;
fprintf('\n================ WYNIKI KOŃCOWE (Zrównoważone Group K-Fold) ================\n');
fprintf('Średnia dokładność SVM (RBF): %.2f%%\n', acc_svm);
fprintf('Średnia dokładność LDA:       %.2f%%\n', acc_lda);
fprintf('============================================================================\n');

%% 5. Wykresy: Zbiorcze Macierze Pomyłek
fig1 = figure('Name','Porównanie Klasyfikatorów na Segmentach','Units','centimeters','Position',[5 5 26 12]);
all_Y_test_SVM_cat = categorical(all_Y_test_SVM);
all_Y_pred_SVM_cat = categorical(all_Y_pred_SVM);
all_Y_test_LDA_cat = categorical(all_Y_test_LDA);
all_Y_pred_LDA_cat = categorical(all_Y_pred_LDA);

subplot(1,2,1);
confusionchart(all_Y_test_SVM_cat, all_Y_pred_SVM_cat, ...
    'Title', 'Macierz Pomyłek: SVM (Segmenty 5-min)', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

subplot(1,2,2);
confusionchart(all_Y_test_LDA_cat, all_Y_pred_LDA_cat, ...
    'Title', 'Macierz Pomyłek: LDA (Segmenty 5-min)', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

%% --- WYWOŁANIE WYKRESÓW PORÓWNAWCZYCH ---
fig2 = plot_hrv_spectra(db_path, window_len_s);
fig3 = plot_features_scatter(X_young, X_elderly);

%% --- POPRAWKA: AUTOMATYCZNY ZAPIS WYKRESÓW DO PLIKÓW PDF DLA LATEXA ---
output_dir = 'wykresy';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Utworzono nowy folder: "%s"\n', output_dir);
end

set(findobj(fig1, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig2, '-property', 'FontName'), 'FontName', 'Arial');
set(findobj(fig3, '-property', 'FontName'), 'FontName', 'Arial');

fprintf('Zapisywanie wykresów wektorowych w formacie .pdf...\n');
% exportgraphics automatycznie docina białe marginesy wokół osi wykresu
exportgraphics(fig1, fullfile(output_dir, 'macierze_pomylek.pdf'), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(output_dir, 'widma_hrv_porownanie.pdf'), 'ContentType', 'vector');
exportgraphics(fig3, fullfile(output_dir, 'przestrzen_cech_scatter.pdf'), 'ContentType', 'vector');
fprintf('Wykresy .pdf zostały pomyślnie zapisane w folderze "%s/".\n', output_dir);


%% =========================================================
%  FUNKCJE POMOCNICZE DLA SEGMENTACJI
% =========================================================
function [X_feat, Y_lab, G_num] = extract_segmented_features(ids, group_name, db_path, win_len, id_offset)
    X_feat = []; Y_lab = []; G_num = [];
    
    for i = 1:numel(ids)
        rec = fullfile(db_path, ids{i});
        try
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
        catch ME
            warning('Błąd segmentacji dla %s: %s', ids{i}, ME.message);
        end
    end
end

function fig_handle = plot_hrv_spectra(db_path, win_len)
%PLOT_HRV_SPECTRA  Widmo HRV dla jednego young i jednego elderly (wycinek 5-min).
    % POPRAWKA: Zmieniono kolory w macierzy na zielony [0 0.8 0.4] dla obu wykresów
    examples = { fullfile(db_path,'f1y01'), 'Młody pacjent – segment 5-min (f1y01)', [0 0.8 0.4]; ...
                 fullfile(db_path,'f1o01'), 'Starszy pacjent – segment 5-min (f1o01)', [0 0.8 0.4] };
    
    fig_handle = figure('Name','Widma HRV dla segmentów 5-minutowych','Units','centimeters','Position',[2 2 22 9]);
    
    for k = 1:2
        rec   = examples{k,1};
        lbl   = examples{k,2};
        col   = examples{k,3};
        
        [signal, fs] = rdsamp(rec); 
        ann          = rdann(rec, 'ecg'); 
        time         = (0:length(signal)-1) / fs;
        r_peaks_t    = time(ann); 
        
        r_peaks_segment = r_peaks_t(r_peaks_t >= 0 & r_peaks_t < win_len);
        
        [~, ~, ~, f, PSD] = compute_hrv_freq(r_peaks_segment); 
        
        subplot(1,2,k);
        % Główny wykres PSD i jego krawędź będą teraz zielone dla obu pacjentów
        area(f, PSD, 'FaceColor', col, 'FaceAlpha', 0.25, ...
             'EdgeColor', col, 'LineWidth', 1.5);
        hold on;
        
        % Podświetlenie pasm LF i HF (nakładane na zielony wykres) zostaje bez zmian
        LF_mask = f >= 0.04 & f < 0.15; 
        HF_mask = f >= 0.15 & f < 0.4;  
        area(f(LF_mask), PSD(LF_mask), 'FaceColor',[0.2 0.4 1], 'FaceAlpha',0.45, 'EdgeColor','none');
        area(f(HF_mask), PSD(HF_mask), 'FaceColor',[1 0.2 0.2], 'FaceAlpha',0.45, 'EdgeColor','none');
        
        xlim([0 0.5]);
        xlabel('Częstotliwość [Hz]');
        ylabel('PSD [s^2/Hz]');
        title(lbl, 'Interpreter', 'none'); % Zabezpieczenie przed gubieniem ogonków
        legend('Moc widmowa (PSD)','Pasmo LF (0.04–0.15 Hz)','Pasmo HF (0.15–0.4 Hz)', ...
               'Location','northeast', 'Interpreter', 'none');
        grid on;
    end
    sgtitle('Porównanie widm mocy HRV (Metoda Welcha) dla grup wiekowych', 'Interpreter', 'none');
end

% ---------------------------------------------------------
function fig_handle = plot_features_scatter(feat_young, feat_elderly)
    fig_handle = figure('Name','Przestrzeń cech HRV – segmenty','Units','centimeters','Position',[2 2 15 11]);
    
    scatter(feat_young(:,1),   feat_young(:,2),   35, ...
        'o', 'filled', 'MarkerFaceColor',[0 0.7 0.3], 'MarkerFaceAlpha', 0.6, 'DisplayName','Young (Młodzi)');
    hold on;
    scatter(feat_elderly(:,1), feat_elderly(:,2), 35, ...
        's', 'filled', 'MarkerFaceColor',[0.85 0.2 0.1], 'MarkerFaceAlpha', 0.6, 'DisplayName','Elderly (Seniorzy)');
    
    xlabel('Moc niskiej częstotliwości ln(LF) [ln(s^2)]');
    ylabel('Moc wysokiej częstotliwości ln(HF) [ln(s^2)]');
    title('Rozkład 5-minutowych segmentów w przestrzeni cech');
    legend('Location','best');
    grid on;
end