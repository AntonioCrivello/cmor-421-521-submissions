clear all
close all
data = readmatrix('cmor-421-521-submissions/homework-1/docs/timing.csv');

%Peak Bandwidth GB/s
peak_bandwidth = 59.7;

%Peak Performance for a single core GFLOPS
peak_performance = 166.4 / 8;

% Initialize matrices to store data for each block size
block_mat_4 = [];
block_mat_8 = [];
block_mat_16 = [];
block_mat_32 = [];
block_mat_64 = [];
block_mat_128 = [];
block_mat_256 = [];
block_mat_512 = [];
block_mat_1024 = [];

for i = 1:length(data)
    data_row = data(i,:);
    block_size = data_row(2);
    if block_size == 4
        block_mat_4 = [block_mat_4; data_row];
    elseif block_size == 8
        block_mat_8 = [block_mat_8; data_row];
    elseif block_size == 16
        block_mat_16 = [block_mat_16; data_row];
    elseif block_size == 32
        block_mat_32 = [block_mat_32; data_row];
    elseif block_size == 64
        block_mat_64 = [block_mat_64; data_row];
    elseif block_size == 128
        block_mat_128 = [block_mat_128; data_row];
    elseif block_size == 256
        block_mat_256 = [block_mat_256; data_row];
    elseif block_size == 512
        block_mat_512 = [block_mat_512; data_row];
    elseif block_size == 1024
        block_mat_1024 = [block_mat_1024; data_row];
    end

end

GFLOPs_naive_4 = zeros(length(block_mat_4(:,1)),1);
GFLOPs_blocked_4 = zeros(length(block_mat_4(:,1)),1);
GFLOPS_recursive_4 = zeros(length(block_mat_4(:,1)),1);
GFLOPS_recursive_intermediates_4 = zeros(length(block_mat_4(:,1)),1);

GFLOPs_naive_8 = zeros(length(block_mat_8(:,1)),1);
GFLOPs_blocked_8 = zeros(length(block_mat_8(:,1)),1);
GFLOPS_recursive_8 = zeros(length(block_mat_8(:,1)),1);
GFLOPS_recursive_intermediates_8 = zeros(length(block_mat_8(:,1)),1);

GFLOPs_naive_16 = zeros(length(block_mat_16(:,1)),1);
GFLOPs_blocked_16 = zeros(length(block_mat_16(:,1)),1);
GFLOPS_recursive_16 = zeros(length(block_mat_16(:,1)),1);
GFLOPS_recursive_intermediates_16 = zeros(length(block_mat_16(:,1)),1);

GFLOPs_naive_64 = zeros(length(block_mat_64(:,1)),1);
GFLOPs_blocked_64 = zeros(length(block_mat_64(:,1)),1);
GFLOPS_recursive_64 = zeros(length(block_mat_64(:,1)),1);
GFLOPS_recursive_intermediates_64 = zeros(length(block_mat_64(:,1)),1);

GFLOPs_naive_256 = zeros(length(block_mat_256(:,1)),1);
GFLOPs_blocked_256 = zeros(length(block_mat_256(:,1)),1);
GFLOPS_recursive_256 = zeros(length(block_mat_256(:,1)),1);
GFLOPS_recursive_intermediates_256 = zeros(length(block_mat_256(:,1)),1);

for i = 1:length(block_mat_4(:,1))
    n = block_mat_4(i,1);
    b = 4;

    CI_naive_4(i) = (2 * n^3) / ((n^3 + 3 * n^2) * 8);
    CI_blocked_4(i) = (2 * n^3) / ((2 * n^2 + 2 * n^3 / b) * 8);
    CI_recursive_4(i) = (2 / 3) * b / 8;
    CI_recursive_intermediates_4(i) = (2 / 3) * b / 8;

    GFLOPS_naive_4(i) = ((2 * n^3) / 1e9) / block_mat_4(i,3);
    GFLOPS_blocked_4(i) = ((2 * n^3) / 1e9) / block_mat_4(i,4);
    GFLOPS_recursive_4(i) = ((2 * n^3) / 1e9) / block_mat_4(i,5);
    GFLOPS_recursive_intermediates_4(i) = ((2 * n^3) / 1e9) / block_mat_4(i,6);
end

for i = 1:length(block_mat_8(:,1))
    n = block_mat_8(i,1);
    b = 8;
    
    CI_naive_8(i) = (2 * n^3) / ((n^3 + 3 * n^2) * 8);
    CI_blocked_8(i) = (2 * n^3) / ((2 * n^2 + 2 * n^3 / b) * 8);
    CI_recursive_8(i) = (2 / 3) * b / 8;
    CI_recursive_intermediates_8(i) = (2 / 3) * b / 8;

    GFLOPS_naive_8(i) = ((2 * n^3) / 1e9) / block_mat_8(i,3);
    GFLOPS_blocked_8(i) = ((2 * n^3) / 1e9) / block_mat_8(i,4);
    GFLOPS_recursive_8(i) = ((2 * n^3) / 1e9) / block_mat_8(i,5);
    GFLOPS_recursive_intermediates_8(i) = ((2 * n^3) / 1e9) / block_mat_8(i,6);
end

for i = 1:length(block_mat_16(:,1))
    n = block_mat_16(i,1);
    b = 16;

    CI_naive_16(i) = (2 * n^3) / ((n^3 + 3 * n^2) * 8);
    CI_blocked_16(i) = (2 * n^3) / ((2 * n^2 + 2 * n^3 / b) * 8);
    CI_recursive_16(i) = (2 / 3) * b / 8;
    CI_recursive_intermediates_16(i) = (2 / 3) * b / 8;

    GFLOPS_naive_16(i) = ((2 * n^3) / 1e9) / block_mat_16(i,3);
    GFLOPS_blocked_16(i) = ((2 * n^3) / 1e9) / block_mat_16(i,4);
    GFLOPS_recursive_16(i) = ((2 * n^3) / 1e9) / block_mat_16(i,5);
    GFLOPS_recursive_intermediates_16(i) = ((2 * n^3) / 1e9) / block_mat_16(i,6);
end

for i = 1:length(block_mat_32(:,1))
    n = block_mat_32(i,1);
    b = 32;
    
    CI_naive_32(i) = (2 * n^3) / ((n^3 + 3 * n^2) * 8);
    CI_blocked_32(i) = (2 * n^3) / ((2 * n^2 + 2 * n^3 / b) * 8);
    CI_recursive_32(i) = (2 / 3) * b / 8;
    CI_recursive_intermediates_32(i) = (2 / 3) * b / 8;

    GFLOPS_naive_32(i) = ((2 * n^3) / 1e9) / block_mat_32(i,3);
    GFLOPS_blocked_32(i) = ((2 * n^3) / 1e9) / block_mat_32(i,4);
    GFLOPS_recursive_32(i) = ((2 * n^3) / 1e9) / block_mat_32(i,5);
    GFLOPS_recursive_intermediates_32(i) = ((2 * n^3) / 1e9) / block_mat_32(i,6);
end

for i = 1:length(block_mat_64(:,1))
    n = block_mat_64(i,1);
    b = 64;

    CI_naive_64(i) = (2 * n^3) / ((n^3 + 3 * n^2) * 8);
    CI_blocked_64(i) = (2 * n^3) / ((2 * n^2 + 2 * n^3 / b) * 8);
    CI_recursive_64(i) = (2 / 3) * b / 8;
    CI_recursive_intermediates_64(i) = (2 / 3) * b / 8;

    GFLOPS_naive_64(i) = ((2 * n^3) / 1e9) / block_mat_64(i,3);
    GFLOPS_blocked_64(i) = ((2 * n^3) / 1e9) / block_mat_64(i,4);
    GFLOPS_recursive_64(i) = ((2 * n^3) / 1e9) / block_mat_64(i,5);
    GFLOPS_recursive_intermediates_64(i) = ((2 * n^3) / 1e9) / block_mat_64(i,6);
end

CI_vector = logspace(log10(1e-6), log10(4), 10000);
roofline = min(peak_performance, CI_vector * peak_bandwidth);

% Plot the roofline performance
figure(1)
hold on;
plot(CI_vector, roofline,'black-','LineWidth',1.2)

plot(CI_naive_8,GFLOPS_naive_8,'bx','DisplayName','Naive')

plot(CI_blocked_4,GFLOPS_blocked_4,'ro')
plot(CI_blocked_8,GFLOPS_blocked_8,'go')
plot(CI_blocked_16,GFLOPS_blocked_16,'bo')
plot(CI_blocked_32,GFLOPS_blocked_32,'mo')
hold off;
xlabel('Operational Intensity [FLOPS/byte]');
ylabel('Performance [GFLOPS]');
title('Roofline Plot for Naive and Blocked Matrix Multiplication');
legend('Roofline Bound','Naive Implementation', ...
       'Blocked Implementation, Block Size = 4', 'Blocked Implementation, Block Size = 8', ...
       'Blocked Implementation, Block Size = 16','Blocked Implementation, Block Size = 32');

figure(2)
hold on;
plot(CI_vector, roofline,'black-','LineWidth',1.2)

plot(CI_recursive_4,GFLOPS_recursive_4,'rx')
plot(CI_recursive_8,GFLOPS_recursive_8,'gx')
plot(CI_recursive_16,GFLOPS_recursive_16,'bx')
plot(CI_recursive_32,GFLOPS_recursive_32,'mx')

plot(CI_recursive_intermediates_4,GFLOPS_recursive_4,'ro')
plot(CI_recursive_intermediates_8,GFLOPS_recursive_8,'go')
plot(CI_recursive_intermediates_16,GFLOPS_recursive_16,'bo')
plot(CI_recursive_intermediates_32,GFLOPS_recursive_32,'mo')
hold off;
xlabel('Operational Intensity [FLOPS/byte]');
ylabel('Performance [GFLOPS]');
title('Roofline Plot for Recursive Matrix Multiplication');
legend('Roofline Bound','Recursive Implementation, Block Size = 4', ...
       'Recursive Implementation, Block Size = 8', 'Recursive Implementation, Block Size = 16', ...
       'Recursive Implementation, Block Size = 32', ...
       'Recursive Intermediates Implementation, Block Size = 4', ...
       'Recursive Intermediates Implementation, Block Size = 8', ...
       'Recursive Intermediates Implementation, Block Size = 16', ...
       'Recursive Intermediates Implementation, Block Size = 32');