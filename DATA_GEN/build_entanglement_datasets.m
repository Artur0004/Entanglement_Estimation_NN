function build_entanglement_datasets()
% Builds dataset_F1.csv (6 features + label) and dataset_F2.csv (9 features + label)

clc;

rng(7);  % reproducibility

% -------------------------
% Configuration
% -------------------------
num_rand = 40000;   % random mixed states
num_sep  = 40000;   % separable (convex mixtures of product states)
num_pure = 20000;   % pure states
total    = num_rand + num_sep + num_pure;

% mixture size for separable states (randomly chosen in this range)
K_min = 2; K_max = 5;

eps_herm  = 1e-12;   % tolerance for Hermiticity
eps_psd   = 1e-12;   % tolerance for minimum eigenvalue
eps_trace = 1e-12;   % tolerance for trace 1
eps_feat  = 1e-8;    % tolerance for imaginary residuals in features

% -------------------------
% Pauli matrices
% -------------------------
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];
P = {sx, sy, sz};

% 9 two-qubit Pauli observables in table order: 
% {xx, xy, xz, yx, yy, yz, zx, zy, zz}
Pauli2 = cell(3,3);
for a = 1:3
    for b = 1:3
        Pauli2{a,b} = kron(P{a}, P{b});
    end
end

% Index helper to pull F1 from F2 order
% F2 order: xx(1), xy(2), xz(3), yx(4), yy(5), yz(6), zx(7), zy(8), zz(9)
idx_F1 = [1,2,3,5,6,9];  % xx,xy,xz,yy,yz,zz

% Pre-allocate
X_F1 = zeros(total, 6);
X_F2 = zeros(total, 9);
y_C  = zeros(total, 1);

row = 1;

fprintf('Building dataset (%d total rows)...\n', total);

% -------------------------
% 1) Random mixed states
% -------------------------
for i = 1:num_rand
    rho = rand_rho_d(4);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli_features(rho, Pauli2, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    c = concurrence_2q(rho, sy);
    y_C(row,1) = max(0, c);  % guard tiny negatives
    row = row + 1;

    if mod(i,5000)==0, fprintf('Random: %d/%d\n', i, num_rand); end
end

% -------------------------
% 2) Separable states (convex mixtures of product states)
%     rho = sum_t w_t (rhoA_t ⊗ rhoB_t)
% -------------------------
for i = 1:num_sep
    K = randi([K_min, K_max]);
    w = rand(1,K); w = w/sum(w);
    rho = zeros(4);
    for t = 1:K
        rhoA = rand_rho_d(2);
        rhoB = rand_rho_d(2);
        rho  = rho + w(t)*kron(rhoA, rhoB);
    end
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli_features(rho, Pauli2, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    c = concurrence_2q(rho, sy);           % should be 0 for separable
    if c < 1e-10, c = 0; end               % clip numerical fuzz
    y_C(row,1) = c;
    row = row + 1;

    if mod(i,5000)==0, fprintf('Separable: %d/%d\n', i, num_sep); end
end

% -------------------------
% 3) Pure states
% -------------------------
for i = 1:num_pure
    rho = rand_pure_d(4);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli_features(rho, Pauli2, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    c = concurrence_2q(rho, sy);
    y_C(row,1) = max(0, c);
    row = row + 1;

    if mod(i,5000)==0, fprintf('Pure: %d/%d\n', i, num_pure); end
end

% -------------------------
% Final sanity: sizes and label range
% -------------------------
assert(size(X_F1,1) == total && size(X_F1,2) == 6);
assert(size(X_F2,1) == total && size(X_F2,2) == 9);
assert(all(y_C >= -1e-12 & y_C <= 1+1e-12));

% -------------------------
% Write CSVs (features only + label)
% -------------------------
hdr_F1 = {'F1_xx','F1_xy','F1_xz','F1_yy','F1_yz','F1_zz','Concurrence'};
hdr_F2 = {'F2_xx','F2_xy','F2_xz','F2_yx','F2_yy','F2_yz','F2_zx','F2_zy','F2_zz','Concurrence'};

T1 = array2table([X_F1, y_C], 'VariableNames', hdr_F1);
T2 = array2table([X_F2, y_C], 'VariableNames', hdr_F2);

writetable(T1, 'dataset_F1.csv');
writetable(T2, 'dataset_F2.csv');

fprintf('Done.\nFiles written: dataset_F1.csv (6+1 cols), dataset_F2.csv (9+1 cols)\n');

% ======================================================================
% Nested helpers
% ======================================================================

    function rho = rand_rho_d(d)
        % Use QETLAB if available
        if exist('RandomDensityMatrix','file') == 2
            rho = RandomDensityMatrix(d);
        else
            % Ginibre ensemble: sigma*sigma^† / Tr(sigma*sigma^†)
            G = (randn(d) + 1i*randn(d))/sqrt(2);
            sigma = G;
            rho = sigma*sigma';
            rho = rho/trace(rho);
        end
    end

    function rho = rand_pure_d(d)
        psi = randn(d,1) + 1i*randn(d,1);
        psi = psi / norm(psi);
        rho = psi*psi';
    end

    function c = concurrence_2q(rho, sy_local)
        if exist('Concurrence','file') == 2
            c = Concurrence(rho);
            return;
        end
        YY = kron(sy_local, sy_local);
        R  = rho * YY * conj(rho) * YY;
        ev = sort(real(eig(R)), 'descend');
        ev(ev<0) = 0; 
        s = sqrt(ev);
        c = max(0, s(1) - s(2) - s(3) - s(4));
    end

    function rho = sanitize_rho(rho, eps_herm_, eps_psd_, eps_trace_)
        % Hermitize
        rho = (rho + rho')/2;
        % Trace -> 1
        tr = real(trace(rho));
        if abs(tr - 1) > eps_trace_
            rho = rho / tr;
        end
        % Project tiny negative eigenvalues to zero if needed
        [V,D] = eig((rho+rho')/2);
        lam = real(diag(D));
        lam(lam < 0 & lam > -1e-10) = 0;   % clip tiny negatives
        if any(lam < -eps_psd_)
            error('PSD check failed: min eig = %.3e', min(lam));
        end
        rho = V*diag(lam)*V';
        % Re-normalize trace (numerical drift)
        rho = rho / trace(rho);
    end

    function assert_valid_rho(rho, eps_herm_, eps_psd_, eps_trace_)
        % Hermitian
        if norm(rho - rho','fro') > eps_herm_
            error('Hermiticity check failed (||rho - rho^†|| > tol).');
        end
        % PSD
        lam = eig(rho);
        if min(real(lam)) < -eps_psd_
            error('PSD check failed: min eig = %.3e', min(real(lam)));
        end
        % Trace 1
        if abs(real(trace(rho)) - 1) > eps_trace_
            error('Trace check failed: Tr(rho)=%.15f', trace(rho));
        end
        % Purity <= 1
        pur = real(trace(rho*rho));
        if pur > 1 + 1e-10
            error('Purity check failed: Tr(rho^2)=%.15f', pur);
        end
    end

    function [f2, f1] = pauli_features(rho, Pauli2_local, idx_F1_local, eps_feat_)
        f2 = zeros(1,9);
        k = 1;
        for a = 1:3
            for b = 1:3
                val = trace(rho * Pauli2_local{a,b});
                if abs(imag(val)) > eps_feat_
                    error('Feature imaginary residue too large: %.3e', imag(val));
                end
                r = real(val);
                if abs(r) > 1 + 5e-3
                    error('Feature out of range: %.3f', r);
                end
                f2(k) = max(-1, min(1, r));  % clamp tiny overshoot
                k = k + 1;
            end
        end
        f1 = f2(idx_F1_local);
    end

end
