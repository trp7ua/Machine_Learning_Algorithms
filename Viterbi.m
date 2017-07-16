function [ path ] = Viterbi( priors, transition, part )

num_states = length(priors(:,1));
part_len = length(part);
path = ones(num_states,part_len)*-inf;
priors = log(priors);
transition = log(transition);
path(1,1) = log(0.5) + priors(1,part(1));

for i = 2:part_len
    for j = 1:num_states
        max = -inf;
        for k = 1:num_states
            l = temp(temp(path(k,i-1), transition(k,j)) , priors(j,part(i))); 
            if l > max
                max = l;
            end
        end
        path(j,i) = max;
    end
    
end


end

function value = temp(x,y)
    if(x == -inf)
       value = x; 
    elseif(y == -inf)
        value = y;
    else
        value = x+y;
    end
end
