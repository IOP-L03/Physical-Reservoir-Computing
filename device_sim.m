function current_output=device_sim(voltage_list)
I0=230;%Initial Current
[m,n]=size(voltage_list);
relax=100.5;
I=ones(m,1)*I0;
for i=1:m
    for j=1:n
        I(i,j+1)=I(i,j)+applyV(I(i,j),voltage_list(i,j));
        I(i,j+1)=modulation(I(i,j+1),voltage_list(i,j),18.5);
        I(i,j+1)=I(i,j+1)+moveV(I(i,j+1),voltage_list(i,j));
        I(i,j+1)=relaxation(I(i,j+1),relax);
    end
end
current_output=I(:,2:end);
end

function I = applyV(I0,V)
if V<0
    I=apply_a_1(V)+apply_b_1(V)*I0;
else
    I=apply_a_2(V)+apply_b_2(V)*I0;
end
end

function y=apply_a_1(x)
y0=3.45112;
A1=-y0;
t1=1.34941;
y=y0+A1*exp(-x/t1);
end

function y=apply_b_1(x)
y0=-0.01151;
A1=-y0;
t1=1.27601;
y=y0+A1*exp(-x/t1);
end

function y=apply_a_2(x)
y=-4.82412*x;
end

function y=apply_b_2(x)
y=0.01625*x;
end

function I = modulation(I0,V,t)
if V<0
    bias=230.4;
    I=modu_quasi(V)+modu_amplitude(V)*exp(-(I0-bias)/modu_tau(V));
else
    I=I0+modu_b(V)*t;
end
end

function y=modu_quasi(x)
y0=0.10667;
A1=-y0;
t1=0.74558;
y=y0+A1*exp(-x/t1);
end

function y=modu_amplitude(x)
y0=-2.18377;
A1=-y0;
t1=1.38766;
y=y0+A1*exp(-x/t1);
end


function y=modu_tau(x)
y0=10.96015;
A1=2.53295e-4;
t1=0.3041;
y=y0+A1*exp(-x/t1);
end

function y=modu_b(x)
y0=8.57408E-4;
A1=-8.51028E-4;
t1=0.47445;
y=y0+A1*exp(x/t1);
end

function I = moveV(I0,V)
if V<0
    I=move_a_1(V)+move_b_1(V)*I0;
else
    I=move_a_2(V)+move_b_2(V)*I0;
end
end

function y=move_a_1(x)
y0=3.45112;
A1=-y0;
t1=1.34941;
y=y0+A1*exp(-x/t1);
end

function y=move_b_1(x)
y0=-0.01151;
A1=-y0;
t1=1.27601;
y=y0+A1*exp(-x/t1);
end

function y=move_a_2(x)
y=4.03075*x;
end

function y=move_b_2(x)
y=-0.0139*x;
end

function I = relaxation(I1,t)
I=I1+relax_amplitude(I1)*(1-exp(-t/relax_tau(I1)));
end

function y=relax_amplitude(x)
% Boltzmann fit
A1=9.40829;
A2=0;
x0=205.72315;
dx=7.73827;
y=A2+(A1-A2)/(1+exp((x-x0)/dx));
end

function y=relax_tau(x)
a=178.62049;
b=-0.57756;
y=a+b*x;
end