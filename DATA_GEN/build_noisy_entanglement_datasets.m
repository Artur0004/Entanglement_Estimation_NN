function build_noisy_entanglement_datasets()
% Builds a noisy, partially observed measurement dataset for NN training.
%
% Files written:
%   - dataset_noisy_measurements.csv
%       Columns:
%           meas_1, ..., meas_M          (noisy measurement values in [-1,1])
%           mask_1, ..., mask_M          (0 = missing, 1 = measured)
%           F2_xx, F2_xy, ..., F2_zz     (clean Pauli expectation values)
%           Concurrence                  (true concurrence in [0,1])
%
%   - measurement_directions.csv
%       Columns:
%           meas_id, nAx, nAy, nAz, nBx, nBy, nBz
%       describing the single-qubit Bloch directions used to define
%       σ(n_A) ⊗ σ(n_B) for each measurement index.
%
% QETLAB usage:
%   - If available, RandomDensityMatrix(d) is used to sample random states.
%   - If available, Concurrence(rho) is used to compute concurrence.

clc;

rng(7);  % reproducibility

% -------------------------
% Configuration: states
% -------------------------
num_rand = 40000;   % random mixed states
num_sep  = 40000;   % separable (convex mixtures of product states)
num_pure = 20000;   % pure states
total    = num_rand + num_sep + num_pure;

% mixture size for separable states
K_min_sep = 2; 
K_max_sep = 5;

eps_herm  = 1e-12;   % tolerance for Hermiticity
eps_psd   = 1e-12;   % tolerance for minimum eigenvalue
eps_trace = 1e-12;   % tolerance for trace 1
eps_feat  = 1e-8;    % tolerance for imaginary residuals in features

% -------------------------
% Configuration: measurements (realistic scenario)
% -------------------------
M             = 10;      % total possible measurement settings (σ(n_A)⊗σ(n_B))
K_min_meas    = 3;       % min number of actually performed measurements per state
K_max_meas    = 5;       % max number of actually performed measurements per state
noise_sigma   = 0.05;    % Gaussian noise std dev added to measured expectations

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

% -------------------------
% Define global measurement directions σ(nA)⊗σ(nB)
% "Arbitrary directions" on Bloch sphere, fixed across all states.
% -------------------------
nA = zeros(M, 3);
nB = zeros(M, 3);
MeasOps = cell(M,1);   % measurement operators in 4x4 space

for m = 1:M
    vA = randn(3,1); vA = vA / norm(vA);
    vB = randn(3,1); vB = vB / norm(vB);
    nA(m,:) = vA.';
    nB(m,:) = vB.';
    
    % σ(n) = n_x σ_x + n_y σ_y + n_z σ_z
    sigA = vA(1)*sx + vA(2)*sy + vA(3)*sz;
    sigB = vB(1)*sx + vB(2)*sy + vB(3)*sz;
    
    MeasOps{m} = kron(sigA, sigB);  % 4x4 Hermitian observable
end

% -------------------------
% Pre-allocate arrays
% -------------------------
X_meas = zeros(total, M);   % noisy measurements
Mask   = zeros(total, M);   % mask (0=missing, 1=present)
X_F2   = zeros(total, 9);   % clean Pauli expectations
y_C    = zeros(total, 1);   % concurrence

row = 1;

fprintf('Building noisy dataset (%d total rows)...\n', total);

% -------------------------
% 1) Random mixed states
% -------------------------
for i = 1:num_rand
    rho = rand_rho_d(4);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    % clean Pauli F2 features
    f2 = pauli_F2_features(rho, Pauli2, eps_feat);
    X_F2(row,:) = f2;

    % noisy, partial measurements in arbitrary bases
    [noisy_meas, mask_vec] = noisy_measurements( ...
        rho, MeasOps, M, K_min_meas, K_max_meas, noise_sigma);
    X_meas(row,:) = noisy_meas;
    Mask(row,:)   = mask_vec;

    % concurrence
    c = concurrence_2q(rho, sy);
    y_C(row,1) = max(0, c);
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Random: %d/%d\n', i, num_rand);
    end
end

% -------------------------
% 2) Separable states (convex mixtures of product states)
%     rho = sum_t w_t (rhoA_t ⊗ rhoB_t)
% -------------------------
for i = 1:num_sep
    K = randi([K_min_sep, K_max_sep]);
    w = rand(1,K); w = w/sum(w);
    rho = zeros(4);
    for t = 1:K
        rhoA = rand_rho_d(2);
        rhoB = rand_rho_d(2);
        rho  = rho + w(t)*kron(rhoA, rhoB);
    end
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    f2 = pauli_F2_features(rho, Pauli2, eps_feat);
    X_F2(row,:) = f2;

    [noisy_meas, mask_vec] = noisy_measurements( ...
        rho, MeasOps, M, K_min_meas, K_max_meas, noise_sigma);
    X_meas(row,:) = noisy_meas;
    Mask(row,:)   = mask_vec;

    c = concurrence_2q(rho, sy);   % should be 0 for separable
    if c < 1e-10, c = 0; end       % clip numerical fuzz
    y_C(row,1) = c;
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Separable: %d/%d\n', i, num_sep);
    end
end

% -------------------------
% 3) Pure states
% -------------------------
for i = 1:num_pure
    rho = rand_pure_d(4);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    f2 = pauli_F2_features(rho, Pauli2, eps_feat);
    X_F2(row,:) = f2;

    [noisy_meas, mask_vec] = noisy_measurements( ...
        rho, MeasOps, M, K_min_meas, K_max_meas, noise_sigma);
    X_meas(row,:) = noisy_meas;
    Mask(row,:)   = mask_vec;

    c = concurrence_2q(rho, sy);
    y_C(row,1) = max(0, c);
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Pure: %d/%d\n', i, num_pure);
    end
end

% -------------------------
% Final sanity checks
% -------------------------
assert(size(X_meas,1) == total && size(X_meas,2) == M);
assert(size(Mask,   1) == total && size(Mask,   2) == M);
assert(size(X_F2,   1) == total && size(X_F2,   2) == 9);
assert(all(y_C >= -1e-12 & y_C <= 1+1e-12));

% -------------------------
% Write CSV: main dataset (inputs + targets)
% -------------------------
hdr_meas = arrayfun(@(k) sprintf('meas_%d', k), 1:M, 'UniformOutput', false);
hdr_mask = arrayfun(@(k) sprintf('mask_%d', k), 1:M, 'UniformOutput', false);
hdr_F2   = {'F2_xx','F2_xy','F2_xz', ...
            'F2_yx','F2_yy','F2_yz', ...
            'F2_zx','F2_zy','F2_zz'};
hdr_all  = [hdr_meas, hdr_mask, hdr_F2, {'Concurrence'}];

T = array2table([X_meas, Mask, X_F2, y_C], 'VariableNames', hdr_all);
writetable(T, 'dataset_noisy_measurements.csv');

% -------------------------
% Write CSV: measurement directions
% -------------------------
MeasDirs = table((1:M).', ...
    nA(:,1), nA(:,2), nA(:,3), ...
    nB(:,1), nB(:,2), nB(:,3), ...
    'VariableNames', {'meas_id','nAx','nAy','nAz','nBx','nBy','nBz'});

writetable(MeasDirs, 'measurement_directions.csv');

fprintf('Done.\n');
fprintf('Files written:\n');
fprintf('  dataset_noisy_measurements.csv  (noisy inputs + clean targets)\n');
fprintf('  measurement_directions.csv      (Bloch directions for each meas_k)\n');

% ======================================================================
% Nested helper functions
% ======================================================================

    function rho = rand_rho_d(d)
        % Random mixed state in dimension d
        if exist('RandomDensityMatrix','file') == 2
            rho = RandomDensityMatrix(d);  % QETLAB
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
        % Two-qubit concurrence, prefer QETLAB if available
        if exist('Concurrence','file') == 2
            c = Concurrence(rho);  % QETLAB function
            return;
        end
        % Fallback: Wootters formula
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

    function f2 = pauli_F2_features(rho, Pauli2_local, eps_feat_)
        % Compute clean 9 Pauli expectation values in the order:
        % xx, xy, xz, yx, yy, yz, zx, zy, zz
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
    end

    function [noisy_meas, mask_vec] = noisy_measurements( ...
            rho, MeasOps_local, M_local, K_min_, K_max_, noise_sigma_)
        % For a given rho, compute:
        %   - clean expectation values for all M_local measurement operators
        %   - choose K ~ Uniform{K_min_,...,K_max_} of them to actually "measure"
        %   - add Gaussian noise to the measured ones
        %   - unmeasured ones are set to 0 with mask=0
        clean_meas = zeros(1, M_local);
        for m_ = 1:M_local
            val = trace(rho * MeasOps_local{m_});
            clean_meas(m_) = real(val);
        end
        
        noisy_meas = zeros(1, M_local);
        mask_vec   = zeros(1, M_local);
        
        K = randi([K_min_, K_max_]);
        idx = randperm(M_local, K);
        
        for m_ = idx
            v = clean_meas(m_) + noise_sigma_*randn;
            v = max(-1, min(1, v));   % keep in [-1,1]
            noisy_meas(m_) = v;
            mask_vec(m_)   = 1;
        end
    end

end
