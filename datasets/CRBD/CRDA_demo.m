clc;
clear all;
close all;

startImg = 1;
endImg = 281;

sigma = 0.1;
adjacentGTRows = 9;
m = 3;

%relative paths to GT data folder and evaluated method results
GT_folder = 'GT data';
results_folder = 'TMGEM results';

%extension of file which contains results of the evaluated method
extension = '.tmg';

CRDA_value = zeros((endImg - startImg + 1),1);

for iImg = startImg : endImg
   
    %load ground truth data
    path = sprintf('%s/crop_row_%03d.crp', GT_folder, iImg);
    GT_CRP = load(path);
    
    %load evaluated method results (TMGEM)
    path = sprintf('%s/crop_row_%03d%s', results_folder, iImg, extension);
    U = load(path);
    
    %calculate CRDA value
    CRDA_value((iImg - startImg + 1)) = CRDA(U, GT_CRP, m, sigma, adjacentGTRows);
    
    %calculate normalized cumulative histogram values
    histogram_step = 1/(endImg - startImg + 1);
    histogram_value = (histogram_step : histogram_step : 1) .* 100;
    
    %sort CRDA_values for normalized cumulative histogram
    CRDA_value_sorted = sort(CRDA_value, 'descend');
    
    %plot histogram
    plot(CRDA_value_sorted, histogram_value);
    grid on;
    
    %set title, xlabel and ylabel
    title('Normalized cumulative histogram');
    xlabel('CRDA');
    ylabel('Percentage of samples');
    
    %Add percentage sign to yticks    
    a=[cellstr(num2str(get(gca,'ytick')'))];
    pct = char(ones(size(a,1),1)*'%');
    new_yticks = [char(a),pct];
    set(gca,'yticklabel',new_yticks); 
end