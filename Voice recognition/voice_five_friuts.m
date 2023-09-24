clc
clear
close all;
%% blueberry
[blueberry,Fs_blue]=audioread('E:\Desktop\voice\zmz\blueberry.wav');
N = length(blueberry);
time=(0:N-1)/Fs_blue; 
subplot(5,3,1)
plot(time,blueberry)
xlabel('Time/s');
ylabel('Amplitude');
title('blueberry');

win=hanning(N);
nfft=30000;
Y1=fft(blueberry.*win,nfft);
f=(0:nfft/2)*Fs_blue/nfft;
subplot(5,3,2)
y=Y1(1:nfft/2+1);
plot(f,y);
title('frequency');

subplot(5,3,3)
y_abs=abs(Y1(1:nfft/2+1));
plot(f,y_abs);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('frequency');

%采样
df=1:40:10000;
dt=1:100:25000;
t=blueberry(dt,:);
sz_t(:,1)=t;
fr=y(df,:);
sz_f(:,1)=abs(fr);
new_bq(1:5,1)=[1,0,0,0,0];
%% lychee
[lychee,Fs_lychee]=audioread('E:\Desktop\zmz\lychee.wav');
N = length(lychee);
time=(0:N-1)/Fs_lychee;
subplot(5,3,4)
plot(time,lychee)
xlabel('Time/s');
ylabel('Amplitude');
title('lychee');

win=hanning(N);
nfft=30000;
Y1=fft(lychee.*win,nfft);
f=(0:nfft/2)*Fs_lychee/nfft;
subplot(5,3,5)
y=Y1(1:nfft/2+1);
plot(f,y);
title('wave');

subplot(5,3,6)
y_abs=abs(Y1(1:nfft/2+1));
plot(f,y_abs);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('frequency');

%采样
df=1:40:10000;
dt=1:100:25000;
t=lychee(dt,:);
sz_t(:,2)=t;
fr=y(df,:);
sz_f(:,2)=abs(fr);
new_bq(1:5,2)=[0,1,0,0,0];
%%  mango
[mango,Fs_mango]=audioread('E:\Desktop\zmz\mango.wav');
N = length(mango);
time=(0:N-1)/Fs_mango;
subplot(5,3,7)
plot(time,mango)
xlabel('Time/s');
ylabel('Amplitude');
title('mango');

win=hanning(N);
nfft=30000;
Y1=fft(mango.*win,nfft);
f=(0:nfft/2)*Fs_mango/nfft;
subplot(5,3,8)
y=Y1(1:nfft/2+1);
plot(f,y);
title('wave');

subplot(5,3,9)
y_abs=abs(Y1(1:nfft/2+1));
plot(f,y_abs);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('frequency');

%采样
df=1:40:10000;
dt=1:100:25000;
t=mango(dt,:);
sz_t(:,3)=t;
fr=y(df,:);
sz_f(:,3)=abs(fr);
new_bq(1:5,3)=[0,0,1,0,0];
%% pomegranate
[pomegranate,Fs_pomegranate]=audioread('E:\Desktop\zmz\pomegranate.wav');
N = length(pomegranate);
time=(0:N-1)/Fs_pomegranate;
subplot(5,3,10)
plot(time,pomegranate)
xlabel('Time/s');
ylabel('Amplitude');
title('pomegranate');

win=hanning(N);
nfft=30000;
Y1=fft(pomegranate.*win,nfft);
f=(0:nfft/2)*Fs_pomegranate/nfft;
subplot(5,3,11)
y=Y1(1:nfft/2+1);
plot(f,y);
title('wave');

subplot(5,3,12)
y_abs=abs(Y1(1:nfft/2+1));
plot(f,y_abs);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('frequency');


df=1:40:10000;
dt=1:100:25000;
t=pomegranate(dt,:);
sz_t(:,4)=t;
fr=y(df,:);
sz_f(:,4)=abs(fr);
new_bq(1:5,4)=[0,0,0,1,0];
%% shadock
[shadock,Fs_shadock]=audioread('E:\Desktop\zmz\shadock.wav');
shadock = shadock(:,1);
N = length(shadock);%求取抽样点数
time=(0:N-1)/Fs_shadock;
subplot(5,3,13)
plot(time,shadock)
xlabel('Time/s');
ylabel('Amplitude');
title('shadock');

win=hanning(N);
nfft=30000;
Y1=fft(shadock.*win,nfft);
f=(0:nfft/2)*Fs_shadock/nfft;
subplot(5,3,14)
y=Y1(1:nfft/2+1);
plot(f,y);
title('wave');

subplot(5,3,15)
y_abs=abs(Y1(1:nfft/2+1));
plot(f,y_abs);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('frequency');


df=1:40:10000;
dt=1:100:25000;
t=shadock(dt,:);
sz_t(:,5)=t;
fr=y(df,:);
sz_f(:,5)=abs(fr);
new_bq(1:5,5)=[0,0,0,0,1];


t_min=min(min(sz_t));
t_max=max(max(sz_t));
new_sz(1:250,1:5)=(sz_t-t_min)/(t_max-t_min)*255;
f_min=min(min(sz_f));
f_max=max(max(sz_f));
new_sz(251:500,1:5)=(sz_f-f_min)/(f_max-f_min)*255;
new_sz=round(new_sz);
sz_uint8=uint8(new_sz);


load('voice_sz.mat');
load('voice_bq.mat');
len_sz=size(sz,2);
len_bq=size(bq,2);
sz(:,len_sz+1:len_sz+5)=new_sz;
bq(:,len_bq+1:len_bq+5)=new_bq;
