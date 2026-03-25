%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bayesian Inference based Interference Estimation
%=================================================
% In this work, we proposed an improved resource allocation algorithm for
% URLLC.
% In legacy systems, the interference for the next transmission is
% estimated as an weighted average of the past interference values. This is
% not suitable for URLLC transmission requiring high reliability
% We propose to estimate the interference using Bayesian Inference
% framework, where the possible interference value is discretized and a
% probability is attached to each level.
% ===================================================
% This work implements the proposed scheme based on the second-order Markov
% chain along with the first-order Markov chain and a baseline scheme
% In the baseline scheme, the estimated interference value 
% I_est(t+1) = w*I_est(t) + (1-w)*I(t-1)


clc;
clearvars;
close all;

nrOfIterations = 1e1; %1e3; %% to calculate average due to random INR
nrOfPreSamples = 1e5; %% number of samples used for training
nrOfRuns = 1e3; %1e6; %% number of samples for prediction
nrOfInterferers = 10; 
meanSnr_dB = 20;
meanSnr_lin = 10.^(meanSnr_dB/10);
minInr_dB = -10;
maxInr_dB = 5;

confidenceLevel = 0.95; % this is the confidence level in the interference estimate 
tgtOutageProbVector = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];

wt_AR = 0.99;
packetLength = 50;

transDuration = 10;
activProb = 0.5;

maxResources = inf; %1e5;% maximum number of resource elements

temp_actualRU = zeros(length(tgtOutageProbVector),nrOfIterations);
temp_RU_AR = temp_actualRU;
temp_RU_prop = temp_actualRU;
temp_RU_prop_2nd = temp_actualRU;
temp_meanOutage_AR = temp_actualRU;
temp_meanOutage_prop_2nd = temp_actualRU;
temp_meanOutage_prop = temp_actualRU;
temp_outage_AR = temp_actualRU;
temp_outage_prop = temp_actualRU;
temp_outage_prop_2nd = temp_actualRU;

actualRU = zeros(length(tgtOutageProbVector),1); % RU:resource usage
RU_AR = actualRU;
RU_prop = actualRU;
RU_prop_2nd = actualRU;

outage_AR = actualRU;
outage_prop = actualRU;
outage_prop_2nd = actualRU;

meanOutage_AR = actualRU;
meanOutage_prop = actualRU;
meanOutage_prop_2nd = actualRU;

outProbIdx = 0;
for tgtOutageProb = tgtOutageProbVector
    outProbIdx = outProbIdx + 1;
    qInvPe = qfuncinv(tgtOutageProb);

% inr_db = [5, 2, 0, -3, -10]; % mean INRs of the interferers in dB
for iter = 1:nrOfIterations
inr_db = (maxInr_dB - minInr_dB)*rand(1,nrOfInterferers) + minInr_dB;
inr_lin = 10.^(inr_db/10);


filterLength = 100; %% to make correlation between samples
filterCoefs = 1/filterLength * ones(filterLength,1);
filterStateI = zeros(filterLength - 1, nrOfInterferers);
filterStateQ = zeros(filterLength - 1, nrOfInterferers);

rng default
% rng(0,'v5uniform');

samplingFreq = 4e6;
sampleTime = 1/samplingFreq;

time = sampleTime*(0:nrOfPreSamples-1);
sum_interference = 0;

for ii = 1:nrOfInterferers
    actInterferers = zeros(nrOfPreSamples,1);
    num = 1;
    while num <= nrOfPreSamples
        var = rand;
        if var <= activProb
            actInterferers(num:min(num + transDuration-1, nrOfPreSamples)) = 1;
            num = num + transDuration;
        else
            actInterferers(num) = 0;
            num = num + 1;
        end
    end

    noiseI = randn(nrOfPreSamples,1) .* actInterferers;
    noiseQ = randn(nrOfPreSamples,1) .* actInterferers;
    
    [noiseI, stateI] = filter(filterCoefs, 1, noiseI, filterStateI(:,ii));
    [noiseQ, stateQ] = filter(filterCoefs, 1, noiseQ, filterStateQ(:,ii));
    filterStateI(:, ii) = stateI;
    filterStateQ(:, ii) = stateQ;
    
    nn = noiseI + 1i*noiseQ;
    interfSignal = (real(nn).^2 + imag(nn).^2);
    interfSignal = interfSignal/mean(interfSignal)*inr_lin(ii);
    
    sum_interference = sum_interference + interfSignal;
end

mx_interf = max(sum_interference);
mn_interf = min(sum_interference);

% discretize the interference into 'nrOfInterfLevels' levels.
nrOfInterfLevels = 15; % number of bins to discretize the interference level
[~,b] = hist((sum_interference).^2,nrOfInterfLevels); %% for the purpose of unequal bins
bin_gap = (b(2) - b(1))/2;
bin_edges = sqrt([b(1) - bin_gap, b + bin_gap]);
bin_edges_old = bin_edges;
bin_edges_old(end) = 2*bin_edges_old(end - 1) - bin_edges_old(end - 2);  
bin_edges(1) = 0;
bin_edges(end) = mx_interf*10;
interf_levels = discretize(sum_interference, bin_edges);

% Generate the transition probability matrix of 1st order Markov chain
transition_matrix = zeros(nrOfInterfLevels); % instances of transitioning from 
% state K (column) to state L (row)
transition_prob_matrix = transition_matrix;
interf_level_count = zeros(nrOfInterfLevels, 1);
for ii = 2:nrOfPreSamples
    transition_matrix(interf_levels(ii), interf_levels(ii-1)) = ...
        transition_matrix(interf_levels(ii), interf_levels(ii-1)) + 1;
    interf_level_count(interf_levels(ii-1)) = ...
        interf_level_count(interf_levels(ii-1)) + 1; 
end

for jj = 1:nrOfInterfLevels
    transition_prob_matrix(:,jj) = ...
        transition_matrix(:,jj)/interf_level_count(jj);
end

transition_prob_matrix_initial = transition_prob_matrix;
meanSumInt = mean(sum_interference);

%%%%%% Generate the transition probability matrix of 2nd order Markov chain
transition_matrix_2nd = zeros(nrOfInterfLevels,nrOfInterfLevels,nrOfInterfLevels); % instances of
% transitioning from state (j,K) to state L
transition_prob_matrix_2nd = transition_matrix_2nd;
interf_state_count = zeros(nrOfInterfLevels,nrOfInterfLevels);
for ii = 3:nrOfPreSamples
    transition_matrix_2nd(interf_levels(ii), interf_levels(ii-1), interf_levels(ii-2)) = ...
        transition_matrix_2nd(interf_levels(ii), interf_levels(ii-1), interf_levels(ii-2)) + 1;
    interf_state_count(interf_levels(ii-1),interf_levels(ii-2)) = ...
        interf_state_count(interf_levels(ii-1),interf_levels(ii-2)) + 1; 
end

for jj = 1:nrOfInterfLevels
    for kk = 1:nrOfInterfLevels
        if interf_state_count(kk,jj) == 0
            transition_prob_matrix_2nd(:,kk,jj) = 0; 
        else
            transition_prob_matrix_2nd(:,kk,jj) = ...
                transition_matrix_2nd(:,kk,jj)/interf_state_count(kk,jj);
        end
    end
end

transition_prob_matrix_2nd_initial = transition_prob_matrix_2nd;

%%%%%%%%%%%%%%%%%%%%%%%%% Algorithm implementation %%%%%%%%%%%%%%%%%%%%%%%%

    outageProb_AR = zeros(nrOfRuns, 1);
    outageProb_prop = outageProb_AR;
    outageProb_prop_2nd = outageProb_AR;
    
    resourceUsage_actual = outageProb_AR;
    resourceUsage_AR = outageProb_AR;
    resourceUsage_prop = outageProb_AR;
    resourceUsage_prop_2nd = outageProb_AR;

    estimatedInterf_AR = meanSumInt;
    transition_prob_matrix = transition_prob_matrix_initial;
    transition_cdf_matrix = cumsum(transition_prob_matrix);

    transition_prob_matrix_2nd = transition_prob_matrix_2nd_initial;
    transition_cdf_matrix_2nd = cumsum(transition_prob_matrix_2nd);
    
    %%%%% Generate the correlated signals (desired and interference) beforehand
    filterStateI = zeros(filterLength - 1, nrOfInterferers);
    filterStateQ = zeros(filterLength - 1, nrOfInterferers);
    
    rng('shuffle');
    
    time = sampleTime*(0:nrOfRuns-1);
    sum_interference = 0;
    
    for ii = 1:nrOfInterferers
        actInterferers = zeros(nrOfRuns,1);
        num = 1;
        while num <= nrOfRuns
            var = rand;
            if var <= activProb
                actInterferers(num:min(num + transDuration-1, nrOfRuns)) = 1;
                num = num + transDuration;
            else
                actInterferers(num) = 0;
                num = num + 1;
            end
        end

        noiseI = randn(nrOfRuns,1) .* actInterferers;
        noiseQ = randn(nrOfRuns,1) .* actInterferers;
        
        [noiseI, stateI] = filter(filterCoefs, 1, noiseI, filterStateI(:,ii));
        [noiseQ, stateQ] = filter(filterCoefs, 1, noiseQ, filterStateQ(:,ii));
        filterStateI(:, ii) = stateI;
        filterStateQ(:, ii) = stateQ;
        
        nn = noiseI + 1i*noiseQ;
        interfSignal = (real(nn).^2 + imag(nn).^2);
        interfSignal = interfSignal/mean(interfSignal)*inr_lin(ii);
        
        sum_interference = sum_interference + interfSignal;
    end
    
    noiseI = randn(nrOfRuns,1);
    noiseQ = randn(nrOfRuns,1);
    
    [noiseI, stateI] = filter(filterCoefs, 1, noiseI, filterStateI(:,ii));
    [noiseQ, stateQ] = filter(filterCoefs, 1, noiseQ, filterStateQ(:,ii));
    filterStateI(:, ii) = stateI;
    filterStateQ(:, ii) = stateQ;
    
    nn = noiseI + 1i*noiseQ;
    desiredSignal = (real(nn).^2 + imag(nn).^2);
    desiredSignal = desiredSignal/mean(desiredSignal)*meanSnr_lin;

    currentInterfLevel = sum(sum_interference(2) >= bin_edges);
    pastInterfLevel = sum(sum_interference(1) >= bin_edges);

    error = 0;
    consec_error = [];
    error_2nd = 0;
    consec_error_2nd = [];

    for runIdx = 3:nrOfRuns
        achievedSnr = desiredSignal(runIdx);
        actualInterf = sum_interference(runIdx);
        actualSir = achievedSnr/actualInterf;
        C_actualSir = log2(1 + actualSir);
        V_actualSir = 1/(log(2)^2)*(1 - 1/((1 + actualSir)^2));
        resourceUsage_actual(runIdx) = min(packetLength/C_actualSir + qInvPe^2*V_actualSir/(2*C_actualSir^2)*...
            (1 + sqrt(1 + 4*packetLength*C_actualSir/(qInvPe^2*V_actualSir))),maxResources);
        
        % AR based estimate
        estimatedSir_AR = achievedSnr/estimatedInterf_AR;
        C_est_AR = log2(1 + estimatedSir_AR);
        V_est_AR = 1/(log(2)^2)*(1 - 1/((1 + estimatedSir_AR)^2));
        resourceUsage_AR(runIdx) = min(packetLength/C_est_AR + ...
            qInvPe^2*V_est_AR/(2*C_est_AR^2)*...
            (1 + sqrt(1 + 4*packetLength*C_est_AR/(qInvPe^2*V_est_AR))),maxResources);
        outageProb_AR(runIdx) = qfunc((resourceUsage_AR(runIdx)*C_actualSir - packetLength)/...
            sqrt(resourceUsage_AR(runIdx)*V_actualSir));
     
        % proposed estimation based on 1st order Markov chain
        estimatedInterfLevel = min(sum(transition_cdf_matrix(:,currentInterfLevel) ...
            <= confidenceLevel) + 1, nrOfInterfLevels); 
        estimatedInterf_prop = bin_edges_old(estimatedInterfLevel+1);
        estimatedSir_prop = achievedSnr/estimatedInterf_prop;
        C_est_prop = log2(1 + estimatedSir_prop);
        V_est_prop = 1/(log(2)^2)*(1 - 1/((1 + estimatedSir_prop)^2));
        resourceUsage_prop(runIdx) = min(packetLength/C_est_prop + ...
            qInvPe^2*V_est_prop/(2*C_est_prop^2)*...
            (1 + sqrt(1 + 4*packetLength*C_est_prop/(qInvPe^2*V_est_prop))),maxResources);
        outageProb_prop(runIdx) = qfunc((resourceUsage_prop(runIdx)*C_actualSir - packetLength)/...
            sqrt(resourceUsage_prop(runIdx)*V_actualSir));

        %%%%%%%%%%%%%%% proposed estimation based on 2nd order Markov chain
        if interf_state_count(currentInterfLevel,pastInterfLevel) < nrOfInterfLevels
            estimatedInterfLevel_2nd = min(sum(transition_cdf_matrix(:,currentInterfLevel) ...
                <= confidenceLevel) + 1, nrOfInterfLevels);
        else
            estimatedInterfLevel_2nd = min(sum(transition_cdf_matrix_2nd(:,currentInterfLevel,pastInterfLevel) ...
                <= confidenceLevel) + 1, nrOfInterfLevels);
        end
        estimatedInterf_prop_2nd = bin_edges_old(estimatedInterfLevel_2nd+1);
        estimatedSir_prop_2nd = achievedSnr/estimatedInterf_prop_2nd;
        C_est_prop_2nd = log2(1 + estimatedSir_prop_2nd);
        V_est_prop_2nd = 1/(log(2)^2)*(1 - 1/((1 + estimatedSir_prop_2nd)^2));
        resourceUsage_prop_2nd(runIdx) = min(packetLength/C_est_prop_2nd + ...
            qInvPe^2*V_est_prop_2nd/(2*C_est_prop_2nd^2)*...
            (1 + sqrt(1 + 4*packetLength*C_est_prop_2nd/(qInvPe^2*V_est_prop_2nd))),maxResources);
        outageProb_prop_2nd(runIdx) = qfunc((resourceUsage_prop_2nd(runIdx)*C_actualSir - packetLength)/...
            sqrt(resourceUsage_prop_2nd(runIdx)*V_actualSir));
        
        % update AR estimate
        estimatedInterf_AR = wt_AR*estimatedInterf_AR + (1 - wt_AR)*actualInterf;
        
        % update the 'transition_prob_matrix' of 1st order
        nextInterfState = zeros(nrOfInterfLevels,1);
        nextInterfState(min(sum(actualInterf >= bin_edges), nrOfInterfLevels)) = 1;
        transition_prob_matrix(:, currentInterfLevel) = ...
            (transition_prob_matrix(:, currentInterfLevel)*interf_level_count(currentInterfLevel) + ...
            nextInterfState)/(interf_level_count(currentInterfLevel) + 1);
        interf_level_count(currentInterfLevel) = interf_level_count(currentInterfLevel) + 1;
        transition_cdf_matrix = cumsum(transition_prob_matrix);
        %     currentInterfLevel = (sum(actualInterf > bin_edges));

        % update the 'transition_prob_matrix' of 2nd order
        transition_prob_matrix_2nd(:,currentInterfLevel,pastInterfLevel) = ...
            (transition_prob_matrix_2nd(:,currentInterfLevel,pastInterfLevel)*interf_state_count(currentInterfLevel,pastInterfLevel) + ...
            nextInterfState)/(interf_state_count(currentInterfLevel,pastInterfLevel) + 1);
        interf_state_count(currentInterfLevel,pastInterfLevel) = interf_state_count(currentInterfLevel,pastInterfLevel) + 1;
        transition_cdf_matrix_2nd = cumsum(transition_prob_matrix_2nd);
        pastInterfLevel = currentInterfLevel;
        currentInterfLevel = min(sum(actualInterf >= bin_edges), nrOfInterfLevels);

    end
    
    temp_actualRU(outProbIdx,iter) = mean(resourceUsage_actual);  
    temp_RU_AR(outProbIdx,iter) = mean(resourceUsage_AR);    
    temp_RU_prop(outProbIdx,iter) = mean(resourceUsage_prop);
    temp_RU_prop_2nd(outProbIdx,iter) = mean(resourceUsage_prop_2nd);
   
    temp_meanOutage_AR(outProbIdx,iter) = mean(outageProb_AR);
    temp_meanOutage_prop(outProbIdx,iter) = mean(outageProb_prop);
    temp_meanOutage_prop_2nd(outProbIdx,iter) = mean(outageProb_prop_2nd);
    
    temp_outage_AR(outProbIdx,iter) = mean(outageProb_AR > tgtOutageProb);
    temp_outage_prop(outProbIdx,iter) = mean(outageProb_prop > tgtOutageProb);
    temp_outage_prop_2nd(outProbIdx,iter) = mean(outageProb_prop_2nd > tgtOutageProb);  

end

end

actualRU = mean(temp_actualRU,2);
RU_AR = mean(temp_RU_AR,2);
RU_prop = mean(temp_RU_prop,2);
RU_prop_2nd = mean(temp_RU_prop_2nd,2);

meanOutage_AR = mean(temp_meanOutage_AR,2);
meanOutage_prop = mean(temp_meanOutage_prop,2);
meanOutage_prop_2nd = mean(temp_meanOutage_prop_2nd,2);

outage_AR = mean(temp_outage_AR,2);
outage_prop = mean(temp_outage_prop,2);
outage_prop_2nd = mean(temp_outage_prop_2nd,2);

