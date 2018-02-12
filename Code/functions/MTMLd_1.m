function U = MTMLd_1(Gamma, V,Z,tau)
r = size(Z,2);
rv = 0;
for i = 1 : r
    rv = rv + Gamma(:,i)*V(:,i)';        
end
% U = (1/(2*tau)*rv+Z*V')*inv(V*V');
U = (1/(2*tau)*rv+Z*V')/(V*V');
end

