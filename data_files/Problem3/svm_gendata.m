function [X y] = svm_gendata(Np, Nn)
%% GENDATA  Generate data
%   [X y] = svm_gendata(NP, NN) generates NP positive and NN negative
%   data points.
%%


Xp = [2 -1 ; 2 1]/sqrt(2) * randn(2, Np) ;
Xp(1,:) =  Xp(1,:) + 2 ;
yp = ones(1, Np) ;

Xn = [2 -1 ; 2 1]/sqrt(2) * randn(2, Nn) ;
Xn(1,:) = Xn(1,:) - 2 ;
yn = - ones(1, Nn) ;

X = [Xp Xn] ;
y = [yp yn] ;

end
