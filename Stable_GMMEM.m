function [mean, var, mix_coef, E, likely] = Stable_GMMEM(input_vector,N,k,threshold,varargin)


    [m,n] = size(input_vector);

    if nargin >4
        mean = varargin{1};
        var = varargin{2};
        mix_coef = varargin{3};

    else
        mean = input_vector(randi(m,[1 k]),:);
        var = repmat(eye(1,1), [1 1 k]);
        a = rand(1,k);
        mix_coef = a./sum(a,2);

    end

    likely(1) = -inf;
    likely(2) = likelihood_log(input_vector, mean, var,mix_coef);

    l=1;
    E=[];
    while((l <= N+1) && (abs(likely(end) -likely(end-1)) > threshold))
        log_p = log_normal_pdf(input_vector, mean, var);
        log_E = log_p + repmat(log(mix_coef), [m,1]);
        normalization = log_exp_sum(log_E);
        log_E = log_E - (normalization*ones(1,k));
        %E = E./repmat(sum(E,2), [1,k]);
        E = exp(log_E);
        
        M = sum(E,1);
        for r = 1:k
           mean(r, :) = sum((E(:,r)*ones(1,n)).*input_vector,1)./M(r);
           deviation = input_vector - ones(m,1)*mean(r,:);
           var(:,:,r) = (((E(:,r)*ones(1,n)).*deviation)'*deviation)./M(r);
           %[a,b,c] = size(var);
           %var= var + ones(a,b,c)*0.01;
           %[a,b] = size(mean);
           %mean = mean + ones(a,b)*0.01;
           
           mix_coef(r) = M(r)./m;

        end
        likely(l+1)=likelihood_log(input_vector, mean, var,mix_coef);

        fprintf('N= %d loglikelihood %f\n', l-1, likely(end));
        l = l+1;
    end

    likely(1) =[];


end

    %{
    function p = normal(input, mean, var)

        [m,n] = size(input);
        c = size(var,3);
        p = zeros(m,c);

        for i = 1:c
            p(:,i) = logmvnpdf(input, mean(i,:), var(:,:,i));
        end


    end 
    %}
    
    function log_probs = log_normal_pdf(input,mean,var)
        [m,n] = size(input);
        comp = size(var,3);
        log_probs = zeros(m,comp);
        for t = 1 : comp
            log_probs(:,t) = log_mvn_pdf(input,mean(t,:),var(:,:,t));
        end
    end
    
    
    function log_lik = likelihood_log(input,mean,var,mix_coeff)

        [m,n] = size(input);
        log_p = log_normal_pdf(input,mean,var);
        log_lik = log_p + repmat(log(mix_coeff),[m,1]);
        log_lik = sum(log_exp_sum(log_lik));

end

%{
function likely = loglikelyhood(input, mean, var,mix_coef)

    p = normal(input, mean, var);
    temp2 = size(input);
    temp = size(mean);

    temp =temp(1);
    %p = p + 0.001*eye(temp2(1),temp(1));
    %p
    likely = sum(log(p*mix_coef'));

end 
%}

function log_probs = log_mvn_pdf(input,mean,var)

    [m,n] = size(input);
    input_m = input - ones(m,1)*mean;

    E = cholcov(var);
    input_inv = input_m / E;
    log_probs = -0.5*sum(input_inv.^2, 2) - sum(log(diag(E))) - n*log(2*pi)/2;

end

function y = log_exp_sum(input)
    [m,n] = size(input);
    input_m = max(input,[],2);
    input = input - input_m*ones(1,n);
    y = input_m + log(sum(exp(input),2));
end
