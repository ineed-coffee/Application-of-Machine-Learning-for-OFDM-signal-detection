clear all
clc
close all
%-------PARAMETER---------
N_train = 600001;
N_cv = 100001;
N_test = 100001;
Total_sample=64*N_cv;
N = 64; % Total number of sub-carriers per block
N_d = 52; % number of sub-carriers that holds data info per block, 데이터가 들어가는 개수 52개, 나머지는 0으로 처리.
N_block = Total_sample/N;
cp_length = ceil(0.25*N); % cp_length는 1/4, 16으로 설정
M = 4;  % PSK Mode, QPSK, 4개
%-------------------------

%---Constellation Define--
for k=1:M 
    constellation(k,1) = cos((pi/M)+((2*pi*(k-1))/M))+(sin((pi/M)+((2*pi*(k-1))/M)))*sqrt(-1);  % QPSK, 4분할
end
%-------------------------

%---Channel Power Delay Profile---
pdp=zeros(cp_length,1);
for i=1:cp_length
     if (i>=1&&i<=8)
         pdp(i,1)=exp(-(i-1)/4);     % channel pdp는 expotenial, 지수함수로 설정 1~8로 설정
     end
end
pdp=pdp/sum(pdp);
%---------------------------------

%----------SNR setting-------------
snr_db=10;                              % SNR은 10dB ~ 35dB로 5dB씩 늘려가며 설정할 것
snr=10.^(snr_db/10);
sigma=sqrt(1./snr);
%----------------------------------

%-------------------------------TRANSMITTER--------------------------------
for k=1:N_d                            % Data symbol generation
    for i=1:N_block
        t=ceil(M*rand(1,1));           % 랜덤하게 값을 만들고
        m(k,i)=constellation(t);       % 52개의 자리에 삽입
    end
end

for k=1:N_block % OFDM symbol generation using IFFT
    x(:,k)=[zeros(N/2-N_d/2-1,1); m(1:N_d/2,k); 0; m(N_d/2+1:N_d,k); zeros(N/2-N_d/2,1)];   % 52개 데이터 넣고, 나머지는 0 삽입!
                                                                                            % 맨앞 시작 5개, 중간 1개, 맨 끝 6개 0으로.
    tx_symbol_without_cp(:,k)=ifft(x(:,k));         % ifft을 진행 
    tx_symbol_with_cp(:,k)= [tx_symbol_without_cp(N-cp_length+1:N,k);tx_symbol_without_cp(:,k)];
    
    for i=1:cp_length
        h(i,k)=sqrt(pdp(i))*(randn+sqrt(-1)*randn);     % 채널 특성 설정
    end
end
%--------------------------------------------------------------------------

for k=1:N_block-1   % 붙어있는 symbol 2개씩을 묶어서 tx_pair로 설정
    tx_pair(:,k) =  [tx_symbol_without_cp(:,k) ; tx_symbol_without_cp(:,k+1)];   
end

%-----------------------------------receiver-------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------------------------------Rx No cp ------------------------------

rx_signal_cp_not_added=zeros(N,N_block); % 일단 전체 0으로 설정
for i=1:cp_length % received signal
    for k=1:N_block
        if k==1     % 처음 들어오는 신호, 첫번째 symbol는 원래 tx 신호와 동일, 채널특성이 추가된 상태.
            rx_signal_cp_not_added(:,k)=rx_signal_cp_not_added(:,k)+h(i,k)*[zeros(i-1,1); tx_symbol_without_cp(1:(N-i+1),k)];      
        else
            if i==1     %다음 symbol인 경우, 처음 들어왔
                rx_signal_cp_not_added(:,k)=rx_signal_cp_not_added(:,k)+h(i,k)*tx_symbol_without_cp(:,k);
            else    % 밀려서 들어온는 신호.
                rx_signal_cp_not_added(:,k)=rx_signal_cp_not_added(:,k)+[h(i,k-1)*tx_symbol_without_cp(N-i+2:N,k-1);h(i,k)*tx_symbol_without_cp(1:(N-i+1),k)];
            end
        end
    end
end

%----------------------------------- Rx Yes cp ---------------------------

rx_signal_cp_added=zeros(N+cp_length,N_block); % 일단 전체 0으로 설정
for i=1:cp_length % received signal
    for k=1:N_block
        if k==1     % 처음 들어오는 신호, 첫번째 symbol는 원래 tx 신호와 동일, 채널특성이 추가된 상태.
            rx_signal_cp_added(:,k)=rx_signal_cp_added(:,k)+h(i,k)*[zeros(i-1,1); tx_symbol_with_cp(1:(N-i+1+cp_length),k)];      
        else
            if i==1     %다음 symbol인 경우, 처음 들어왔
                rx_signal_cp_added(:,k)=rx_signal_cp_added(:,k)+h(i,k)*tx_symbol_with_cp(:,k);
            else    % 밀려서 들어온는 신호.
                rx_signal_cp_added(:,k)=rx_signal_cp_added(:,k)+[h(i,k-1)*tx_symbol_with_cp(N-i+2+cp_length:N+cp_length,k-1);h(i,k)*tx_symbol_with_cp(1:(N-i+1+cp_length),k)];
            end
        end
    end
end

%----------------------------------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%여기까지 Received Signal%%%%%%%%%%%%%%%%%%%%%
%RX에 Noise 추가
for k=1:N_block
    AWGN_cp_x = sigma*(randn(N,1)+sqrt(-1)*randn(N,1))/(N*sqrt(2));    % AWGN, 노이즈  (N*sqrt(2)) 에서 sqrt(2)로 바꿈
    rx_signal_cp_not_added(:,k) = rx_signal_cp_not_added(:,k) + AWGN_cp_x;   % cp가 더해지지 않은 rx 신호에 AWGN 신호 추가.
    
    AWGN_cp_o = sigma*(randn(N+cp_length,1)+sqrt(-1)*randn(N+cp_length,1))/(N*sqrt(2));    % AWGN, 노이즈
    rx_signal_cp_added(:,k) = rx_signal_cp_added(:,k) + AWGN_cp_o;   % cp가 더해지지 않은 rx 신호에 AWGN 신호 추가.
end

%------- Data reshaping for Neural Network---------------------------------
X_material = zeros(2*(N-cp_length),N_block-1);              % reshape 할 때 48*2, 양쪽 symbol에서 겹치는 부분 제외하고,
Y_material = zeros(cp_length,N_block-1);                    % (16 x N_block-1), isi!

        % tx pair을 이용한 X_material, Y_material 생성
for k=1:N_block-1
    AWGN_NN = sigma*(randn(2*N-cp_length,1)+sqrt(-1)*randn(2*N-cp_length,1))/(N*sqrt(2));    % AWGN 생성
    for i=1:cp_length
        X_material(:,k)=X_material(:,k) + h(i,k+1)*[tx_pair(cp_length+2-i:N+1-i,k);tx_pair(N+cp_length+2-i:N+N+1-i,k)];
        if i==1
            Y_material(:,k)=Y_material(:,k) + h(i,k+1)*[tx_pair(N+1:N+cp_length+1-i,k)];
        else 
            Y_material(:,k)=Y_material(:,k) + h(i,k+1)*[tx_pair(N+N-i+2:N+N,k);tx_pair(N+1:N+cp_length+1-i,k)];
        end
    end
            % AWGN 추가
    X_material(:,k)=X_material(:,k)+AWGN_NN(1:2*(N-cp_length),1);
    Y_material(:,k)=Y_material(:,k)+AWGN_NN(2*(N-cp_length)+1:end,1);
end    
        
        
X_in = zeros(N_block-1,4*(N-cp_length)+2*cp_length);
for i=1:(N_block-1)
    for j=1:(N-cp_length)
        X_in(i,j) = real(X_material(j,i));
    end
    for k=1:(N-cp_length)
        X_in(i,(N-cp_length)+k) = imag(X_material(k,i));
    end
    for s=1:(N-cp_length)
        X_in(i,2*(N-cp_length)+s) = real(X_material(s+(N-cp_length),i));
    end
    for t=1:(N-cp_length)
        X_in(i,3*(N-cp_length)+t) = imag(X_material(t+(N-cp_length),i));
    end
    for p=1:cp_length
        X_in(i,4*(N-cp_length)+p) = real(h(p,i+1));
    end
    for q=1:cp_length
        X_in(i,4*(N-cp_length)+cp_length+q) = imag(h(q,i+1));
    end
end

Y_out = zeros(N_block-1,2*(cp_length));
for k=1:N_block-1
    for i=1:cp_length
        Y_out(k,i)= real(Y_material(i,k));
    end
    for j=1:cp_length
        Y_out(k,j+cp_length)=imag(Y_material(j,k));
    end 
end
X_in = N.*X_in;
Y_out = N.*Y_out;


% save('training_input_channel_8.mat','X_in','h')
% save('training_output_expected_channel_8.mat','Y_out')
% save('training_symbol_channel_8.mat','m')

% save('cv_input_channel_8.mat','X_in','h')
% save('cv_output_channel_8.mat','Y_out')

%%% 이름 16 채널로 바꿔야함 %%%
save('test_input_channel_8_10db.mat','X_in','h')
% save('test_output_expected_channel_8.mat','Y_out')
save('test_symbol_channel_8_10db.mat','m')
% save('test_rx_cp_x_channel_8_35dB.mat','rx_signal_cp_not_added','rx_signal_cp_added','h','m')