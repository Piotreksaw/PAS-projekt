function [LF, HF, LF_HF, f, PSD] = compute_hrv_freq(r_peaks_time)

    %% 1. RR intervals
    RR = diff(r_peaks_time);              % w sekundach
    t_RR = r_peaks_time(2:end);           % przypisanie czasu do RR

    %% 2. Interpolacja (ważne!)
    fs_interp = 4; % standard w HRV
    t_interp = t_RR(1):1/fs_interp:t_RR(end);

    RR_interp = interp1(t_RR, RR, t_interp, 'spline');

    %% 3. Usunięcie trendu
    RR_interp = detrend(RR_interp);

    %% 4. PSD – metoda Welcha
    [PSD, f] = pwelch(RR_interp, [], [], [], fs_interp);

    %% 5. Zakresy częstotliwości
    LF_band = (f >= 0.04 & f < 0.15);
    HF_band = (f >= 0.15 & f < 0.4);

    %% 6. Moc w pasmach (całkowanie)
    LF = trapz(f(LF_band), PSD(LF_band));
    HF = trapz(f(HF_band), PSD(HF_band));

    %% 7. Stosunek
    LF_HF = LF / HF;

end