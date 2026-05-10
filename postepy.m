clear all; close all; clc;

% Autorzy: Piotr Sawicki 319003, Michał Odziemkowski 311267

%% przygotowanie folderów do zapisu zdjec
out_dir = 'wykresy';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Utworzono folder: %s\n\n', out_dir);
end

db_path = 'fantasia-database-1.0.0/';

%% przygotowanie struktur do prezentacji  wynikow
records = struct();
records(1).id    = 'f1y01';
records(1).label = 'Young (f1y01)';
records(1).color = [0.10 0.60 0.20];

records(2).id    = 'f1o01';
records(2).label = 'Elderly (f1o01)';
records(2).color = [0.85 0.25 0.10];

fs_interp = 4;   % Hz – standard analizy HRV

for r = 1:2
    rec = fullfile(db_path, records(r).id);
    fprintf('Wczytywanie: %s\n', records(r).id);

    [signal, fs]   = rdsamp(rec);
    ann            = rdann(rec, 'ecg');

    records(r).fs      = fs;
    records(r).ecg     = signal(:,2);
    records(r).time    = (0:length(signal)-1) / fs;
    records(r).ann     = ann;
    records(r).r_time  = records(r).time(ann);

    % Szereg RR + interpolacja
    RR       = diff(records(r).r_time);
    t_RR     = records(r).r_time(2:end);
    t_interp = t_RR(1) : 1/fs_interp : t_RR(end);
    RR_interp = interp1(t_RR, RR, t_interp, 'spline');
    RR_interp = detrend(RR_interp);

    % FFT
    [LF_f, HF_f, LFHF_f, f_fft, psd_fft] = compute_psd_fft(RR_interp, fs_interp);

    % Welch
    [LF_w, HF_w, LFHF_w, f_welch, psd_welch] = compute_psd_welch(RR_interp, fs_interp);

    records(r).f_fft    = f_fft;
    records(r).psd_fft  = psd_fft;
    records(r).f_welch  = f_welch;
    records(r).psd_welch = psd_welch;
    records(r).LF_f     = LF_f;   records(r).HF_f  = HF_f;   records(r).LFHF_f  = LFHF_f;
    records(r).LF_w     = LF_w;   records(r).HF_w  = HF_w;   records(r).LFHF_w  = LFHF_w;

    fprintf('  fs = %d Hz | R-peaks: %d | Czas: %.1f min\n', ...
        fs, numel(ann), records(r).time(end)/60);
    fprintf('  FFT   → LF=%.4f  HF=%.4f  LF/HF=%.4f\n', LF_f, HF_f, LFHF_f);
    fprintf('  Welch → LF=%.4f  HF=%.4f  LF/HF=%.4f\n\n', LF_w, HF_w, LFHF_w);
end

%% Wykres ekg z zaznaczonymi peakami z pliku .ecg
t_window = 60;  

fig1 = figure('Name','EKG z R-peaks', ...
    'Units','centimeters', 'Position', [1 1 24 12] );
for r = 1:2
    subplot(2,1,r);

    mask_ecg = records(r).time <= t_window;
    mask_ann = records(r).r_time <= t_window;

    plot(records(r).time(mask_ecg), records(r).ecg(mask_ecg), ...
        'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
    hold on;
    plot(records(r).r_time(mask_ann), ...
         records(r).ecg(records(r).ann(mask_ann)), ...
         'o', 'Color', records(r).color, ...
         'MarkerFaceColor', records(r).color, 'MarkerSize', 5);

    xlabel('Czas [s]');
    ylabel('Amplituda [mV]');
    title(['Sygnał EKG – ', records(r).label,sprintf(' ', records(r).fs)]);
    legend('EKG', 'Załamki R', 'Location','northeast');
    xlim([0 t_window]);
    grid on;
end

sgtitle('Fragment sygnału EKG z zaznaczonymi załamkami R');
save_pdf(fig1, fullfile(out_dir, 'wykres_ekg_rpeaks.pdf'));

%% Porownanie FFT z Welch na przykladzie young
fig2 = figure('Name','FFT vs Welch – young', ...
    'Units','centimeters','Position',[1 1 24 9]);

subplot(1,2,1);
plot_spectrum(records(1).f_fft, records(1).psd_fft, ...
    'FFT', records(1).color);

subplot(1,2,2);
plot_spectrum(records(1).f_welch, records(1).psd_welch, ...
    'Welch', records(1).color);

sgtitle(['Porównanie metod PSD – ', records(1).label]);
save_pdf(fig2, fullfile(out_dir, 'wykres_widmo_young.pdf'));

%% Porownanie FFT z Welch na przykladzie eldery
fig3 = figure('Name','FFT vs Welch – elderly', ...
    'Units','centimeters','Position',[1 1 24 9]);

subplot(1,2,1);
plot_spectrum(records(2).f_fft, records(2).psd_fft, ...
    'FFT', records(2).color);

subplot(1,2,2);
plot_spectrum(records(2).f_welch, records(2).psd_welch, ...
    'Welch', records(2).color);

sgtitle(['Porównanie metod PSD – ', records(2).label]);
save_pdf(fig3, fullfile(out_dir, 'wykres_widmo_elderly.pdf'));

%% podsumowanie
fprintf('=== Podsumowanie: FFT vs Welch ===\n');
fprintf('%-10s  %-6s  %6s  %6s  %7s\n', 'Rekord','Metoda','LF','HF','LF/HF');
fprintf('%s\n', repmat('-',1,44));
for r = 1:2
    fprintf('%-10s  %-6s  %6.4f  %6.4f  %7.4f\n', ...
        records(r).id, 'FFT',   records(r).LF_f, records(r).HF_f, records(r).LFHF_f);
    fprintf('%-10s  %-6s  %6.4f  %6.4f  %7.4f\n', ...
        records(r).id, 'Welch', records(r).LF_w, records(r).HF_w, records(r).LFHF_w);
    fprintf('%s\n', repmat('-',1,44));
end
%% Lokalne funkcje

function [LF, HF, LF_HF, f, PSD] = compute_psd_fft(RR_interp, fs)
    N   = length(RR_interp);
    win = hann(N)';
    X   = fft(RR_interp .* win);
    PSD = (1/(fs*N)) * abs(X).^2;
    PSD = PSD(1:floor(N/2)+1);
    PSD(2:end-1) = 2 * PSD(2:end-1);
    f   = (0:floor(N/2)) * fs / N;

    LF    = trapz(f(f>=0.04 & f<0.15), PSD(f>=0.04 & f<0.15));
    HF    = trapz(f(f>=0.15 & f<0.40), PSD(f>=0.15 & f<0.40));
    LF_HF = LF / HF;
end

function [LF, HF, LF_HF, f, PSD] = compute_psd_welch(RR_interp, fs)
    [PSD, f] = pwelch(RR_interp, [], [], [], fs);
    LF    = trapz(f(f>=0.04 & f<0.15), PSD(f>=0.04 & f<0.15));
    HF    = trapz(f(f>=0.15 & f<0.40), PSD(f>=0.15 & f<0.40));
    LF_HF = LF / HF;
end

function plot_spectrum(f, PSD, method_name, col)
    area(f, PSD, 'FaceColor', col, 'FaceAlpha', 0.15, ...
         'EdgeColor', col, 'LineWidth', 1.5);
    hold on;
    area(f(f>=0.04 & f<0.15), PSD(f>=0.04 & f<0.15), ...
         'FaceColor',[0.15 0.35 1.00], 'FaceAlpha', 0.55, 'EdgeColor','none');
    area(f(f>=0.15 & f<0.40), PSD(f>=0.15 & f<0.40), ...
         'FaceColor',[1.00 0.20 0.10], 'FaceAlpha', 0.55, 'EdgeColor','none');
    xlim([0 0.5]);
    xlabel('Częstotliwość [Hz]');
    ylabel('PSD [s²/Hz]');
    title(method_name);
    legend('PSD', 'LF (0.04–0.15 Hz)', 'HF (0.15–0.40 Hz)', ...
           'Location','northeast');
    grid on;
end

function add_params_text(LF, HF, LFHF)
    ax = gca;
    txt = sprintf('LF = %.4f\nHF = %.4f\nLF/HF = %.4f', LF, HF, LFHF);
    text(ax.XLim(2)*0.55, ax.YLim(2)*0.75, txt, ...
        'FontSize', 9, 'BackgroundColor', [1 1 1 0.7], ...
        'EdgeColor',[0.6 0.6 0.6]);
end

function save_pdf(fig, filename)
    set(fig, 'PaperUnits', 'centimeters');
    sz = get(fig, 'Position');
    set(fig, 'PaperSize',     [sz(3) sz(4)]);
    set(fig, 'PaperPosition', [0 0 sz(3) sz(4)]);
    print(fig, filename, '-dpdf', '-vector');
    fprintf('Zapisano: %s\n', filename);
end
