function [NMistake] = bintestfun(xdata,ydata,W)

Y           = sign(W*xdata');
NMistake    = 0;

for i = 1:length(Y)
    if Y(i) ~= ydata(i)
        NMistake = NMistake +1;
    end
end

