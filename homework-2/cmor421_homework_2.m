clear all
close all
data = readmatrix('cmor-421-521-submissions/homework-2/docs/timing.csv');

% Matrix sizes
matrix_size = data(1:6:end, 1);
% Number of threads
thread_numbers = data(1:6, 2);

% Producing timing matrices from data
for i = 1:length(matrix_size)
    start_index = (i - 1) * 6 + 1;
    end_index = (start_index) + 5;
    parallel_blocked_timings(i,:) = ...
        data(start_index:end_index, 3);
    parallel_collapsed_blocked_timings(i,:) = ...
        data(start_index:end_index, 4);
    parallel_backsolve_static_timings(i,:) = ...
        data(start_index:end_index, 5);
    parallel_backsolve_dynamic_timings(i,:) = ...
        data(start_index:end_index, 6);
end

% Calculate Speedup S(n) = T(1) / T(n) for each algorithm
parallel_blocked_speedup = parallel_blocked_timings(:,1) ...
    ./ parallel_blocked_timings;
parallel_collapsed_blocked_speedup = parallel_collapsed_blocked_timings(:,1) ...
    ./ parallel_collapsed_blocked_timings;
parallel_backsolve_static_speedup = parallel_backsolve_static_timings(:,1) ...
    ./ parallel_backsolve_static_timings;
parallel_backsolve_dynamic_speedup = parallel_backsolve_dynamic_timings(:,1) ...
    ./ parallel_backsolve_dynamic_timings;

% Calculate Efficiency E(n) = S(n) / n for each thread count
parallel_blocked_efficiency = parallel_blocked_speedup ...
    ./ thread_numbers';
parallel_collapsed_blocked_efficiency = parallel_collapsed_blocked_speedup ...
    ./ thread_numbers';
parallel_backsolve_static_efficiency = parallel_backsolve_static_speedup ...
    ./ thread_numbers';
parallel_backsolve_dynamic_efficiency = parallel_backsolve_dynamic_speedup ...
    ./ thread_numbers';

% Generate figure for the speedup of the parallel blocked implementations
figure(1)
hold on
grid on
for i = 1:length(matrix_size)
    plot(thread_numbers, parallel_blocked_speedup(i,:), '-o','DisplayName',...
        ['Parallel Blocked Speed-up for Matrix Size: ', num2str(matrix_size(i))])
    plot(thread_numbers, parallel_collapsed_blocked_speedup(i,:),'-+','DisplayName',...
        ['Parallel Collapsed Blocked Speed-up for Matrix Size: ', num2str(matrix_size(i))])
end
title('Parallel Blocked Methods: Speed-up');
xlabel('Number of Threads')
ylabel('Speed-up Value')
legend('Location', 'northeastoutside');

% Generate figure for the efficiency of the parallel blocked implementations
figure(2)
hold on
grid on
for i = 1:length(matrix_size)
    plot(thread_numbers, parallel_blocked_efficiency(i,:), '-o','DisplayName',...
        ['Parallel Blocked Efficiency for Matrix Size: ', num2str(matrix_size(i))])
    plot(thread_numbers, parallel_collapsed_blocked_efficiency(i,:),'-+','DisplayName',...
        ['Parallel Collapsed Blocked Efficiency for Matrix Size: ', num2str(matrix_size(i))])
end
title('Parallel Blocked Methods: Efficiency');
xlabel('Number of Threads')
ylabel('Efficiency Value')
legend('Location', 'northeastoutside');

% Generate figure for efficiency versus matrix size of static scheduling
figure(3);
hold on;
grid on;
for i = 1:length(thread_numbers)
    plot(matrix_size, parallel_backsolve_static_efficiency(:,i), '-o', 'DisplayName', ...
        ['Static Scheduling Efficiency, Threads: ', num2str(thread_numbers(i))]);
end
title('Parallel Back Solve with Static Scheduling Efficiency Vs. Matrix Size');
xlabel('Matrix Size');
ylabel('Efficiency');
legend('Location', 'best');

% Generate figure for efficiency versus matrix size of dynamic scheduling
figure(4);
hold on;
grid on;
for i = 1:length(thread_numbers)
    plot(matrix_size, parallel_backsolve_dynamic_efficiency(:,i), '-+', 'DisplayName', ...
        ['Dynamic Scheduling Efficiency, Threads: ', num2str(thread_numbers(i))]);
end
title('Parallel Back Solve with Dynamic Scheduling Efficiency Vs. Matrix Size');
xlabel('Matrix Size');
ylabel('Efficiency');
legend('Location', 'best');






