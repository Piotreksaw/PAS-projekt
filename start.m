clear all; close all; clc;

%% =========================================================
%  HRV Classification – Fantasia Database
%  Grupy: young (f1y01–f1y10) vs elderly (f1e01–f1e10)
%  Cechy: LF, HF, LF/HF
%  Klasyfikator: SVM (fitcsvm)
% =========================================================

%% 1. Definicja rekordów
young_ids   = {'f1y01','f1y02','f1y03','f1y04','f1y05', ...
               'f1y06','f1y07','f1y08','f1y09','f1y10'};
elderly_ids = {'f1o01','f1o02','f1o03','f1o04','f1o05', ...
               'f1o06','f1o07','f1o08','f1o09','f1o10'};

db_path = 'fantasia-database-1.0.0/';  % <-- dostosuj jeśli trzeba

%% 2. Ekstrakcja cech HRV
[features_young,   labels_young]   = extract_features(young_ids,   'young',   db_path);
[features_elderly, labels_elderly] = extract_features(elderly_ids, 'elderly', db_path);

X = [features_young;   features_elderly];   % macierz cech [N x 3]
Y = [labels_young;     labels_elderly];     % etykiety (categorical)

fprintf('\nWczytano %d rekordów young + %d elderly.\n', ...
    height(features_young), height(features_elderly));

%% 3. Podział train / test (hold-out 80/20, stratyfikowany)
rng(42);
cv = cvpartition(Y, 'HoldOut', 0.2, 'Stratify', true);
X_train = X(training(cv), :);   Y_train = Y(training(cv));
X_test  = X(test(cv), :);       Y_test  = Y(test(cv));

%% 4. Standaryzacja cech (ważne dla SVM!)
mu    = mean(X_train);
sigma = std(X_train);
X_train_z = (X_train - mu) ./ sigma;
X_test_z  = (X_test  - mu) ./ sigma;

%% 5. Trening SVM (jądro RBF)
svm_model = fitcsvm(X_train_z, Y_train, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint',  1, ...
    'KernelScale',    'auto', ...
    'Standardize',    false, ...   % już znormalizowane ręcznie
    'ClassNames',     categorical({'young','elderly'}));

%% 6. Predykcja i ocena
Y_pred = predict(svm_model, X_test_z);

acc = mean(Y_pred == Y_test) * 100;
fprintf('\n=== Wyniki klasyfikacji SVM ===\n');
fprintf('Accuracy: %.1f%%\n', acc);

%% 7. Macierz pomyłek
figure('Name','Macierz pomyłek','Units','centimeters','Position',[2 2 14 11]);
cm = confusionchart(Y_test, Y_pred, ...
    'Title',           'Macierz pomyłek – SVM (RBF)', ...
    'RowSummary',      'row-normalized', ...
    'ColumnSummary',   'column-normalized');
cm.FontSize = 12;

%% 8. Wykresy widm HRV dla przykładowych rekordów
plot_hrv_spectra(db_path);

%% 9. Wykres cech LF vs HF (scatter)
plot_features_scatter(features_young, features_elderly);

%% =========================================================
%  FUNKCJE POMOCNICZE
% =========================================================

function [feat_table, labels] = extract_features(ids, group_name, db_path)
%EXTRACT_FEATURES  Wczytuje rekordy, liczy LF/HF/LF_HF dla każdego pacjenta.
    N = numel(ids);
    LF_vec    = nan(N,1);
    HF_vec    = nan(N,1);
    LF_HF_vec = nan(N,1);
    valid     = true(N,1);

    for i = 1:N
        rec = fullfile(db_path, ids{i});
        try
            [signal, fs] = rdsamp(rec);
            ecg = signal(:,2);
            ann = rdann(rec, 'ecg');

            % Odrzuć jeśli za mało załamków R
            if numel(ann) < 20
                warning('Za mało R-peaks w %s – pomijam.', ids{i});
                valid(i) = false;
                continue
            end

            time        = (0:length(ecg)-1) / fs;
            r_peaks_t   = time(ann);

            [LF, HF, LF_HF] = compute_hrv_freq(r_peaks_t);

            LF_vec(i)    = LF;
            HF_vec(i)    = HF;
            LF_HF_vec(i) = LF_HF;

            fprintf('[%s] %s – LF=%.4f  HF=%.4f  LF/HF=%.4f\n', ...
                group_name, ids{i}, LF, HF, LF_HF);
        catch ME
            warning('Błąd dla %s: %s', ids{i}, ME.message);
            valid(i) = false;
        end
    end

    % Usuń rekordy z błędami
    LF_vec    = LF_vec(valid);
    HF_vec    = HF_vec(valid);
    LF_HF_vec = LF_HF_vec(valid);

    feat_table = [LF_vec, HF_vec, LF_HF_vec];
    labels     = repmat(categorical({group_name}), sum(valid), 1);
end

% ---------------------------------------------------------

function plot_hrv_spectra(db_path)
%PLOT_HRV_SPECTRA  Widmo HRV dla jednego young i jednego elderly.
    examples = { fullfile(db_path,'f1y01'), 'Young (f1y01)', [0 0.8 0.4]; ...
                 fullfile(db_path,'f1o01'), 'Elderly (f1o01)', [0.85 0.3 0.1] };

    figure('Name','Widma HRV','Units','centimeters','Position',[2 2 20 8]);
    for k = 1:2
        rec   = examples{k,1};
        lbl   = examples{k,2};
        col   = examples{k,3};

        [signal, fs] = rdsamp(rec);
        ann          = rdann(rec, 'ecg');
        time         = (0:length(signal)-1) / fs;
        r_peaks_t    = time(ann);

        [~, ~, ~, f, PSD] = compute_hrv_freq(r_peaks_t);

        subplot(1,2,k);
        area(f, PSD, 'FaceColor', col, 'FaceAlpha', 0.25, ...
             'EdgeColor', col, 'LineWidth', 1.5);
        hold on;

        % Zaznacz pasma LF i HF
        LF_mask = f >= 0.04 & f < 0.15;
        HF_mask = f >= 0.15 & f < 0.4;
        area(f(LF_mask), PSD(LF_mask), 'FaceColor',[0.2 0.4 1], 'FaceAlpha',0.45, 'EdgeColor','none');
        area(f(HF_mask), PSD(HF_mask), 'FaceColor',[1 0.2 0.2], 'FaceAlpha',0.45, 'EdgeColor','none');

        xlim([0 0.5]);
        xlabel('Częstotliwość [Hz]');
        ylabel('PSD [s^2/Hz]');
        title(['Widmo HRV – ', lbl]);
        legend('PSD','LF (0.04–0.15 Hz)','HF (0.15–0.4 Hz)', ...
               'Location','northeast');
        grid on;
    end
    sgtitle('Porównanie widm HRV: Young vs Elderly');
end

% ---------------------------------------------------------

function plot_features_scatter(feat_young, feat_elderly)
%PLOT_FEATURES_SCATTER  Wykres rozproszenia LF vs HF z podziałem na grupy.
    figure('Name','Cechy HRV – scatter','Units','centimeters','Position',[2 2 14 11]);

    scatter(feat_young(:,1),   feat_young(:,2),   80, ...
        'o', 'filled', 'MarkerFaceColor',[0 0.7 0.3], 'DisplayName','Young');
    hold on;
    scatter(feat_elderly(:,1), feat_elderly(:,2), 80, ...
        's', 'filled', 'MarkerFaceColor',[0.85 0.2 0.1], 'DisplayName','Elderly');

    xlabel('LF [s^2]');
    ylabel('HF [s^2]');
    title('Przestrzeń cech HRV (LF vs HF)');
    legend('Location','best');
    grid on;
end