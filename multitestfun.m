function [NMistake] = multitestfun(xts,yts,W,Nclass)

trsize  = size(xts,1);
ftsize  = size(xts,2);
Y       = zeros(trsize,1);

F       = zeros(Nclass,ftsize);
        
for i = 1:trsize
    for k = 1:Nclass
        F(k,:)     = xts(i,:);
    end

    [maxm, index]   = max(dot(W',F'));
    Y(i)            = index-1;
end
        


NMistake    = 0;

for i = 1:length(Y)
    if Y(i) ~= yts(i)
        NMistake = NMistake +1;
    end
end

end
