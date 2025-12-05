%% 3-qubit entanglement dataset (F1 & F2) using QETLAB
% Produces:
%   - dataset3_F1.csv  (10 features + GlobalEntanglement)
%   - dataset3_F2.csv  (27 features + GlobalEntanglement)

clear; clc;
rng(7);  % reproducibility

% -------------------------
% Configuration
% -------------------------
num_rand = 40000;   % random mixed 3-qubit states
num_sep  = 40000;   % separable mixed (convex mixtures of product states)
num_pure = 20000;   % pure 3-qubit states
total    = num_rand + num_sep + num_pure;

% mixture size for separable states (randomly chosen in this range)
K_min = 2; K_max = 5;

eps_herm  = 1e-12;
eps_psd   = 1e-12;
eps_trace = 1e-12;
eps_feat  = 1e-8;    % tolerance for imaginary parts of features

% -------------------------
% Pauli matrices
% -------------------------
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];
P  = {sx, sy, sz};   % {x, y, z}

% 27 three-qubit Pauli observables in lexicographic order:
% indices: 1->x, 2->y, 3->z
% order: (1,1,1), (1,1,2), ..., (3,3,3)
Pauli3 = cell(3,3,3);
for a = 1:3
    for b = 1:3
        for c = 1:3
            Pauli3{a,b,c} = kron(P{a}, kron(P{b}, P{c}));
        end
    end
end

% F2 order: all 27 in (a,b,c) lexicographic
% F1 subset: sorted index triples (combinations with repetition):
% (1,1,1)=xxx, (1,1,2)=xxy, (1,1,3)=xxz, (1,2,2)=xyy, (1,2,3)=xyz,
% (1,3,3)=xzz, (2,2,2)=yyy, (2,2,3)=yyz, (2,3,3)=yzz, (3,3,3)=zzz
idx_F1_triples = [
    1 1 1;
    1 1 2;
    1 1 3;
    1 2 2;
    1 2 3;
    1 3 3;
    2 2 2;
    2 2 3;
    2 3 3;
    3 3 3
];

% We'll map (a,b,c) -> linear index in F2 order
% F2 linear index: (a-1)*9 + (b-1)*3 + c
idx_F1 = zeros(size(idx_F1_triples,1),1);
for k = 1:size(idx_F1_triples,1)
    a = idx_F1_triples(k,1);
    b = idx_F1_triples(k,2);
    c = idx_F1_triples(k,3);
    idx_F1(k) = (a-1)*9 + (b-1)*3 + c;
end

% -------------------------
% Pre-allocate
% -------------------------
X_F1 = zeros(total, 10);    % 10 F1 features
X_F2 = zeros(total, 27);    % 27 F2 features
y_E  = zeros(total, 1);     % Global entanglement label

dims = [2 2 2];             % local dimensions for 3 qubits
row = 1;

fprintf('Building 3-qubit dataset (%d total rows)...\n', total);

% -------------------------
% Helper: random dxd density matrix
% -------------------------
function rho = rand_rho_d(d)
    if exist('RandomDensityMatrix','file') == 2
        rho = RandomDensityMatrix(d);
    else
        G = (randn(d) + 1i*randn(d))/sqrt(2);
        sigma = G;
        rho = sigma*sigma';
        rho = rho/trace(rho);
    end
end

% -------------------------
% Helper: random pure state density matrix of dim d
% -------------------------
function rho = rand_pure_d(d)
    psi = randn(d,1) + 1i*randn(d,1);
    psi = psi / norm(psi);
    rho = psi*psi';
end

% -------------------------
% Helper: sanitize rho numerically
% -------------------------
function rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace)
    rho = (rho + rho')/2;  % Hermitize
    tr = real(trace(rho));
    if abs(tr - 1) > eps_trace
        rho = rho / tr;
    end
    [V,D] = eig((rho + rho')/2);
    lam = real(diag(D));
    lam(lam < 0 & lam > -1e-10) = 0; % clip tiny negatives
    if any(lam < -eps_psd)
        error('PSD check failed: min eig = %.3e', min(lam));
    end
    rho = V*diag(lam)*V';
    rho = rho / trace(rho);
end

% -------------------------
% Helper: asserts for valid rho
% -------------------------
function assert_valid_rho(rho, eps_herm, eps_psd, eps_trace)
    if norm(rho - rho','fro') > eps_herm
        error('Hermiticity check failed.');
    end
    lam = eig(rho);
    if min(real(lam)) < -eps_psd
        error('PSD check failed: min eig = %.3e', min(real(lam)));
    end
    if abs(real(trace(rho)) - 1) > eps_trace
        error('Trace check failed: Tr(rho)=%.15f', trace(rho));
    end
    pur = real(trace(rho*rho));
    if pur > 1 + 1e-10
        error('Purity check failed: Tr(rho^2)=%.15f', pur);
    end
end

% -------------------------
% Helper: compute F2 and F1 features
% -------------------------
function [f2, f1] = pauli3_features(rho, Pauli3, idx_F1, eps_feat)
    f2 = zeros(1,27);
    k  = 1;
    for a = 1:3
        for b = 1:3
            for c = 1:3
                val = trace(rho * Pauli3{a,b,c});
                if abs(imag(val)) > eps_feat
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
    f1 = f2(idx_F1);
end

% -------------------------
% Helper: global entanglement label
%   E = 1 - (Tr(rhoA^2) + Tr(rhoB^2) + Tr(rhoC^2))/3
% -------------------------
function E = global_entanglement(rho, dims)
    % QETLAB PartialTrace: PartialTrace(rho, sys, dims)
    % sys = which subsystems to trace out
    rhoA = PartialTrace(rho, [2 3], dims); % keep qubit 1
    rhoB = PartialTrace(rho, [1 3], dims); % keep qubit 2
    rhoC = PartialTrace(rho, [1 2], dims); % keep qubit 3

    pA = real(trace(rhoA*rhoA));
    pB = real(trace(rhoB*rhoB));
    pC = real(trace(rhoC*rhoC));

    E = 1 - (pA + pB + pC)/3;
    % Clamp to [0,1] in case of tiny numerical overshoot
    if E < 0 && E > -1e-10
        E = 0;
    end
    if E > 1 && E < 1+1e-10
        E = 1;
    end
end

% -------------------------
% 1) Random mixed 3-qubit states
% -------------------------
for i = 1:num_rand
    rho = rand_rho_d(8);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli3_features(rho, Pauli3, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    y_E(row,1)  = global_entanglement(rho, dims);
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Random: %d/%d\n', i, num_rand);
    end
end

% -------------------------
% 2) Separable mixed 3-qubit states
%     rho = sum_t w_t (rhoA_t ⊗ rhoB_t ⊗ rhoC_t)
% -------------------------
for i = 1:num_sep
    K = randi([K_min, K_max]);
    w = rand(1,K); w = w/sum(w);
    rho = zeros(8);
    for t = 1:K
        rhoA = rand_rho_d(2);
        rhoB = rand_rho_d(2);
        rhoC = rand_rho_d(2);
        rho  = rho + w(t)*kron(rhoA, kron(rhoB, rhoC));
    end
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli3_features(rho, Pauli3, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    y_E(row,1)  = global_entanglement(rho, dims);
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Separable: %d/%d\n', i, num_sep);
    end
end

% -------------------------
% 3) Pure 3-qubit states
% -------------------------
for i = 1:num_pure
    rho = rand_pure_d(8);
    rho = sanitize_rho(rho, eps_herm, eps_psd, eps_trace);
    assert_valid_rho(rho, eps_herm, eps_psd, eps_trace);

    [f2, f1] = pauli3_features(rho, Pauli3, idx_F1, eps_feat);
    X_F1(row,:) = f1;
    X_F2(row,:) = f2;
    y_E(row,1)  = global_entanglement(rho, dims);
    row = row + 1;

    if mod(i,5000)==0
        fprintf('Pure: %d/%d\n', i, num_pure);
    end
end

% -------------------------
% Final sanity
% -------------------------
assert(size(X_F1,1) == total && size(X_F1,2) == 10);
assert(size(X_F2,1) == total && size(X_F2,2) == 27);
assert(all(y_E >= -1e-6 & y_E <= 1+1e-6));

% -------------------------
% Write CSVs
% -------------------------

% F1 column names (sorted triples)
F1_labels = {
    'xxx','xxy','xxz','xyy','xyz',...
    'xzz','yyy','yyz','yzz','zzz'
};
hdr_F1 = [strcat('F1_', F1_labels), {'GlobalEntanglement'}];

% F2 column names (all triples in lexicographic order)
axes = {'x','y','z'};
F2_labels = cell(1,27);
k = 1;
for a = 1:3
    for b = 1:3
        for c = 1:3
            F2_labels{k} = [axes{a} axes{b} axes{c}];
            k = k + 1;
        end
    end
end
hdr_F2 = [strcat('F2_', F2_labels), {'GlobalEntanglement'}];

T1 = array2table([X_F1, y_E], 'VariableNames', hdr_F1);
T2 = array2table([X_F2, y_E], 'VariableNames', hdr_F2);

writetable(T1, 'dataset3_F1.csv');
writetable(T2, 'dataset3_F2.csv');

fprintf('Done.\nFiles written: dataset3_F1.csv (10+1 cols), dataset3_F2.csv (27+1 cols)\n');
