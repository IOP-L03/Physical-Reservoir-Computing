clc;
close all;
clear all;
Vmax=2;
Vmin=0;
wave_length=8;
%Sine
w1=sin(pi*2*(0:wave_length-1)/wave_length);
%Square
w2(1:wave_length/2) = 1;
w2(wave_length/2+1:wave_length) = -1;
type=2;
wave_num=300;%number of waveform
for i=1:wave_num
    w=randi(type);
    if w==1
        waveform(wave_length*(i-1)+1:wave_length*i)=w1;
        wave_label(wave_length*(i-1)+1:wave_length*i)=0;
    else
        waveform(wave_length*(i-1)+1:wave_length*i)=w2;
        wave_label(wave_length*(i-1)+1:wave_length*i)=1;
    end
end
ratio=0.5;%train/datasize
n=25;%number of mask
m=50;%length of mask
%Henon_train
input_train=waveform(1:round(ratio*wave_num)*wave_length);
target_train=wave_label(1:round(ratio*wave_num)*wave_length);
%Henon_test
input_test=waveform(round(ratio*wave_num)*wave_length+1:wave_num*wave_length);
target_test=wave_label(round(ratio*wave_num)*wave_length+1:wave_num*wave_length);
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
figure;
subplot(3, 1, 1);
plot(input_test, 'b', 'linewidth', 1);
hold on;
plot(input_test, '.r');
axis([0, wave_length*50, -1.2, 1.2])
ylabel('Input')
set(gca,'FontName', 'Arial', 'FontSize', 20);
subplot(3, 1, 2);
plot(target_test, 'k', 'linewidth', 2);
hold on;
plot(output_sim, 'r', 'linewidth',1);
axis([0, 400, -0.2, 1.2])
str1 = '\color{black}Target';
str2 = '\color{red}Output sim';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
subplot(3, 1, 3);
plot(target_test, 'k', 'linewidth', 2);
hold on;
plot(output_linear, 'r', 'linewidth',1);
axis([0, 400, -0.2, 1.2])
str1 = '\color{black}Target';
str2 = '\color{red}Output linear';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.6]);