function [ CRDA_value ] = CRDA( U, GT_CRP, m, sigma, adjacentGTRows)

% CRDA Crop Row Detection Accuracy
%   Evaluation of the crop row detection method by comparing the results
%   obtained by these method with groung truth.
%   See I. Vidoviæ, R. Cupec and Ž. Hocenski "Crop Row Detection by Global
%   Energy Minimization," submitted to Pattern Recognition (June 15, 2015).
%
% Syntax
%   CRDA_value = CRDA(U, GT_CRP, m, sigma, adjacentGTRows);
%   CRDA_value = CRDA(U, GT_CRP, m, sigma);
%   CRDA_value = CRDA(U, GT_CRP, m);
%   CRDA_value = CRDA(U, GT_CRP);
%
% Inputs:
%   - U: matrix of dimensions h x m, where h represents the number of image
%   rows and m represents the number of crop rows to be evaluated.
%   Elements of each matrix row (u_{v,1}, u_{v,2}, …, u_{v,m}) represent
%   the horizontal coordinates of m adjacent crop rows detected by the 
%   considered method in the v-th image row
%
%   - GT_CRP: Ground Truth Crop Row Parameters. Matrix of dimensions
%   (h-v0) x 2 where v0 represents the first image row where crop rows are
%   visible. Each matrix row contains crop row parameters c_v* and d_v*
%   where c_v* represents position of the central crop row relative to the
%   image center and d_v* represents distance between crop rows in the v-th
%   image row.
%
%   - m (optional): the number of crop rows to be evaluated. The default
%   value is 3. Must be an odd number.
%    
%   - sigma (optional): the desired accuracy needed for safe guidance of an
%   agricultural machine. The default value is 0.1, which means that the
%   matching score is greater than zero only if the horizontal distance
%   between a detected crop row and the corresponding ground truth curve
%   is less than 10% of the distance between adjacent crop rows.
%   
%   - adjacentGTRows (optional): number of created adjacent ground truth
%   crop rows. CRDA value is calculated for each m adjacent crop rows
%   and maximum of n (n = adjacentGTRows - m + 1) obtained CRDA values 
%   is considered as the final performance measure for a particular image.
%
%    - adjacentGTRows (optional): the ground truth values are generated for
%   multiple groups of m adjacent crop rows and CRDA is computed for each
%   group. The total number of adjacent crop rows in all groups is
%   specified by the parameter adjacentGTRows. Therefore, the number of
%   groups is n=adjacentGTRows-m+1. The maximum of n obtained CRDA values
%   is considered as the final performance measure for a particular image.
%   The default value is 9.
%
% Output:
%   - CRDA_value: Crop Row Detection Accuracy value.
%
%
% Copyright (c), Ivan Vidoviæ and Robert Cupec
% Faculty of Electrical Engineering Osijek
% J.J. Strossmayer University of Osijek
% Croatia
% ividovi2(at)etfos.hr; rcupec(at)etfos.hr
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this Software without restriction, subject to the following
% conditions:
% The above copyright notice and this permission notice should be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is," without warranty of any kind.
%
% Created: June 3, 2015
% Last modified: June 16, 2015
%

%% Check input arguments
if nargin < 2
    error('At least parameters U and GT_CRP must be defined!');
elseif nargin == 2
    m = 3;
    sigma = 0.1;
    adjacentGTRows = 9;
elseif nargin == 3
    sigma = 0.1;
    adjacentGTRows = 9;
elseif nargin == 4
    adjacentGTRows = 9;      
end

if (size(U,2) ~= m)
    error('Number of columns in matrix U must be equal to parameter m!');
end
if (mod(m,2) == 0)
    error('Number of crop rows to be evaluated must be odd number!');
end

%% Calculate CRDA
CRDA_value = 0;
imgHeight = 240;
imgWidth = 320;

halfWidth = imgWidth/2;

iShiftStart = - floor(adjacentGTRows / m);
iShiftEnd = floor(adjacentGTRows / m);

iRowStart = -(m - 1) / 2;
iRowEnd = (m - 1) / 2;

%first image row where crop rows are present
v0 = imgHeight - size(GT_CRP, 1);

%save c and d parameters to seperate arrays
cGT = GT_CRP(:,1);
dGT = GT_CRP(:,2);

for iShift = iShiftStart : iShiftEnd
    
    CRDA_temp = 0;
    
    for v = 1 : (imgHeight - v0)
        
        for iRow = iRowStart : iRowEnd
           
            uGT = cGT(v) + (iShift + iRow) * dGT(v) + halfWidth;
            u = U((v + v0), (iRow - iRowStart + 1));
            
            distance = 1 - (((uGT - u) / (sigma * dGT(v))) * ((uGT - u) / (sigma * dGT(v))));
            
            if(distance < 0)
                distance = 0;
            end
            
            CRDA_temp = CRDA_temp + distance;            
        end        
    end
    
    CRDA_temp = CRDA_temp / (m * (imgHeight - v0));
    
    if(CRDA_temp > CRDA_value)
        CRDA_value = CRDA_temp;
    end
end

end

