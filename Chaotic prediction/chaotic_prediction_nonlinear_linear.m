clc;
close all;
clear all;
Vmax=2;
Vmin=0;
datasize=2000;%Henon datasize
[x,y]=Henon(datasize+1);%generate Henon map
ratio=0.5;%train/datasize
n=25;%number of mask
m=50;%length of mask
%Henon_train
input_train=x(1:round(ratio*datasize));
target_train=x(2:round(ratio*datasize)+1);
%Henon_test
input_test=x(round(ratio*datasize)+1:datasize);
target_test=x(round(ratio*datasize)+2:datasize+1);
ntrain=length(input_train);%train datasize
ntest=length(input_test);%test datasize
mask=2*randi(2,n,m)-3;%generate mask
%train process
%mask process
train_mask=[];
for j=1:n
    for i=1:ntrain
        train_mask(j,(i-1)*m+1:m*i)=input_train(1,i)*mask(j,:);
    end
end
train_max=max(max(train_mask));
train_min=min(min(train_mask));
%voltage input
train_voltage=(train_mask-train_min)/(train_max-train_min)*(Vmax-Vmin)+Vmin;
%device output
current_output_linear=device_linear(train_voltage);
current_output_sim=device_sim(train_voltage);
%linear regression
a_linear=[];
a_sim=[];
states_linear=[];
states_sim=[];
for i=1:ntrain
    a_linear=current_output_linear(:, m*(i-1)+1:m*i);
    a_sim=current_output_sim(:, m*(i-1)+1:m*i);
    states_linear(:,i)=a_linear(:);
    states_sim(:,i)=a_sim(:);
end
input_linear=[ones(1,ntrain);states_linear];
input_sim=[ones(1,ntrain);states_sim];
weight_linear=target_train*pinv(input_linear);
weight_sim=target_train*pinv(input_sim);
%test process
%mask process
test_mask=[];
for j=1:n
    for i=1:ntest
        test_mask(j,(i-1)*m+1:m*i)=input_test(1,i)*mask(j,:);
    end
end
test_max=max(max(test_mask));
test_min=min(min(test_mask));
%voltage input
test_voltage=(test_mask-test_min)/(test_max-test_min)*(Vmax-Vmin)+Vmin;
%device output
current_output_linear=device_linear(test_voltage);
current_output_sim=device_sim(test_voltage);
%chaotic prediction
a_linear=[];
a_sim=[];
states_linear=[];
states_sim=[];
for i=1:ntest
    a_linear=current_output_linear(:, m*(i-1)+1:m*i);
    a_sim=current_output_sim(:, m*(i-1)+1:m*i);
    states_linear(:,i)=a_linear(:);
    states_sim(:,i)=a_sim(:);
end
input_linear=[ones(1,ntest);states_linear];
input_sim=[ones(1,ntest);states_sim];
output_linear=weight_linear*input_linear;
output_sim=weight_sim*input_sim;
NRMSE_linear=sqrt(mean((output_linear(10:end)-target_test(10:end)).^2)./var(target_test(10:end)));
NRMSE_sim=sqrt(mean((output_sim(10:end)-target_test(10:end)).^2)./var(target_test(10:end)));
sprintf('%s',['NRMSE_linear:',num2str(NRMSE_linear)])
sprintf('%s',['NRMSE_sim:',num2str(NRMSE_sim)])
% ----------------------PLOT----------------------
% time series
figure(1);
subplot(2,1,1);
plot(target_test(1:200), 'k', 'linewidth', 2);
hold on;
plot(output_linear(1:200), 'r', 'linewidth',1);
axis([0, 200, -2, 2])
str1 = '\color{black}Target';
str2 = '\color{red}Output_linear';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon', 'box', 'off');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);
subplot(2,1,2);
plot(target_test(1:200), 'k', 'linewidth', 2);
hold on;
plot(output_sim(1:200), 'r', 'linewidth',1);
axis([0, 200, -2, 2])
str1 = '\color{black}Target';
str2 = '\color{red}Output sim';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon', 'box', 'off');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.6]);
% 2D map
figure(2);
subplot(1,2,1);
plot(target_test(2:end), 0.3*target_test(1:end-1), '.k', 'markersize', 12);
hold on;
plot(output_linear(2:end), 0.3*output_linear(1:end-1), '.r', 'markersize', 12);
str1 = '\color{black}Target';
str2 = '\color{red}Output linear';
lg = legend(str1,str2);
set(lg, 'box', 'off');
ylabel('{\ity} (n)');
xlabel('{\itx} (n)');
axis([-2, 2, -0.4, 0.4]);
set(gca, 'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.3,0.45]);
subplot(1,2,2);
plot(target_test(2:end), 0.3*target_test(1:end-1), '.k', 'markersize', 12);
hold on;
plot(output_sim(2:end), 0.3*output_sim(1:end-1), '.r', 'markersize', 12);
str1 = '\color{black}Target';
str2 = '\color{red}Output sim';
lg = legend(str1,str2);
set(lg, 'box', 'off');
ylabel('{\ity} (n)');
xlabel('{\itx} (n)');
axis([-2, 2, -0.4, 0.4]);
set(gca, 'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.6,0.45]);