#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <tuple>
#include <vector>

using namespace std;

template <class T>
void coutV (string text, vector<T> someVector) {
    cout << endl << text << endl;
    
    for (int i = 0; i < someVector.size(); i++) {
        cout << someVector[i] << ", ";
    }
    
    cout << endl;
}

template <class T>
bool contains (vector<T> data, T element) {
    return find(data.begin(), data.end(), element) != data.end();
}

/* pearson, spearman */
float mean (vector<float> values) {
    float sum  = 0;
    int    size = values.size();
    for (int i = 0; i < size; i++) {
        sum += values[i];
    }
    
    return sum / size;
}

float pearson_numerator (vector<float> A, vector<float> B, float meanA, float meanB) {
    float numerator = 0;
    for (int i = 0; i < A.size(); i++) {
        numerator += (A[i] - meanA) * (B[i] - meanB);
    }
    
    return numerator;
}

float pearson_denominator (vector<float> A, vector<float> B, float meanA, float meanB) {
    float denominator1;
    float denominator1_sum = 0;
    float denominator2;
    float denominator2_sum = 0;
    
    for (int i = 0; i < A.size(); i++) {
        denominator1_sum += pow(A[i] - meanA, 2);
    }
    
    for (int i = 0; i < B.size(); i++) {
        denominator2_sum += pow(B[i] - meanB, 2);
    }
    
    denominator1 = pow(denominator1_sum, 0.5);
    denominator2 = pow(denominator2_sum, 0.5);
    
    if (denominator1 == 0 || denominator2 == 0)
        cout << endl << endl << "##### ERROR: Denominator equal to 0 - probable cause: all result values are equal" << endl << endl;
    
    return denominator1 * denominator2;
}

float pearson (vector<float> A, vector<float> B) {
    if (A.size() != B.size()) {
        cout << "ERROR - wrong vector lengths" << endl;
        return -1;
    }
    
    float meanA = mean(A);
    float meanB = mean(B);
    
    float numerator   = pearson_numerator(A, B, meanA, meanB);
    float denominator = pearson_denominator(A, B, meanA, meanB);
    
    return numerator / denominator;
}

vector<float> toRanks (vector<float> A) {
    vector<float> sorted = A;
    sort(sorted.begin(), sorted.end());
    
    vector<float> ranks;
        
    for (int i = 0; i < A.size(); i++) {
        vector<int> positions;
        
        for (int j = 0; j < A.size(); j++) {
            if (sorted[j] == A[i]) {
                positions.push_back(j);
            }
        }
        
        float sum = 0;
        float avg;
        
        for (int j = 0; j < positions.size(); j++) {
            sum += positions[j] + 1;
        }
        
        avg = sum / positions.size();
        
        ranks.push_back(avg);
        //ranks.push_back(positions[positions.size()-1] + 1); //libreoffice calc ranks
    }
    
    /*
    cout << "Ranking: " << endl;
    for (int i = 0; i < ranks.size(); i++) {
        cout << ranks[i] << ", ";
    }
    cout << endl << endl;
    */
    
    return ranks;
}

vector<float> toPositions (vector<float> data, bool moreIsBetter) {
    int dataPoints = data.size();
    
    vector<float> sorted = data;
    
    if (moreIsBetter) {
        sort(sorted.begin(), sorted.end(), greater<int>()); // greater<int>() - provides reversed order (descending)
    } else {
        sort(sorted.begin(), sorted.end());
    }
    
    vector<float> positions;
        
    for (int i = 0; i < dataPoints; i++) {        
        for (int j = 0; j < dataPoints; j++) {
            if (sorted[j] == data[i]) {
                positions.push_back(j + 1);
                break;
            }
        }
    }
    
    return positions;
}

float spearman (vector<float> A, vector<float> B) {
    vector<float> A_ranked = toRanks(A);
    vector<float> B_ranked = toRanks(B);
   
    return pearson(A_ranked, B_ranked);
}


/* rest */

vector<string> getFileNames (string path) {
    DIR *pDIR;
    struct dirent *entry;
    vector<string> fileNames;
    if (pDIR=opendir(path.c_str())) {
        while (entry = readdir(pDIR)) {
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                fileNames.push_back(entry->d_name);
            }
        }
        closedir(pDIR);
    }

    return fileNames;
}

tuple<vector<float>, vector<float>, vector<float>, vector<float>> getRanksData (vector<vector<float>> datasets) {
    vector<vector<float>> positions;
    vector<vector<float>> ranks;
    
    for (int i = 0; i < datasets.size(); i++) {
        positions.push_back(toPositions(datasets[i], true));
    }
    
    for (int i = 0; i < datasets.size(); i++) {
        ranks.push_back(toRanks(datasets[i]));
    }
    
    vector<float> positionAverages;
    vector<float> positionSums;
    vector<float> ranksAverages;
    vector<float> ranksSums;
    
    for (int parametersSet = 0; parametersSet < ranks[0].size(); parametersSet++) {
        float ranksSum = 0;
        float positionsSum = 0;
        
        for (int dataset = 0; dataset < ranks.size(); dataset++) {
            ranksSum += ranks[dataset][parametersSet];
            positionsSum += positions[dataset][parametersSet];
        }
        
        positionAverages.push_back(positionsSum / positions.size());
        positionSums.push_back(positionsSum);
        
        ranksAverages.push_back(ranksSum / ranks.size());
        ranksSums.push_back(ranksSum);
    }
    
    return make_tuple(positionAverages, positionSums, ranksAverages, ranksSums);
}

vector<int> getIndexesOfBest (vector<float> data, int nrOfBest, bool moreIsBetter) {
    vector<int> bestIndexes;
    
    int toFind = nrOfBest == -1 ? data.size() : nrOfBest;
    
    for (int i = 0; i < toFind; i++) {
        float max = moreIsBetter ? -1 : numeric_limits<int>::max();
        int maxIndex = moreIsBetter ? -1 : numeric_limits<int>::max();
        
        for (int j = 0; j < data.size(); j++) {
            if (!contains(bestIndexes, j) && (moreIsBetter ? data[j] > max : data[j] < max)) {
                max = data[j];
                maxIndex = j;
            }
        }
        
        bestIndexes.push_back(maxIndex);
    }
    
    return bestIndexes;
}

vector<int> getExcelRowNumbers (vector<int> data, int tests) {
    vector<int> rowNumbers;
    
    for (int i = 0; i < data.size(); i++) {
        rowNumbers.push_back((tests + 1) * (data[i] + 1) + 1);
    }
    
    return rowNumbers;
}

int main() {
    vector<vector<float>> results;
    std::ofstream outfile;    
    
    /*
    vector<float> facebook {1, 2, 2, 4};
    vector<float> digg     {4, 2, 2, 1};
    vector<float> irvine   {1, 2, 2, 4};
    vector<float> enron    {4, 2, 2, 1};
    
    results.push_back(facebook);
    results.push_back(digg);
    results.push_back(irvine);
    results.push_back(enron);
    
    vector<string> datasetNames = {"facebook", "digg", "irvine", "enron"};
    */
    
    
    vector<float> digg      {374,375,378,375,375,371,341,341,328,373,381,374,371,377,377,340,340,328,377,379,374,379,376,377,340,344,328,375,377,375,375,374,370,342,342,328,374,379,373,372,376,374,345,345,326,375,372,375,373,374,376,342,340,326,371,375,374,371,373,370,337,340,326,374,369,372,370,375,370,343,339,326,373,372,374,368,375,375,338,340,326,376,385,382,380,381,381,347,342,345,378,384,383,377,383,384,347,346,344,376,378,381,381,383,383,343,349,344,376,378,384,382,379,377,343,343,347,383,381,386,376,380,383,342,347,340,379,382,381,380,377,386,348,338,347,375,377,378,380,382,380,342,353,349,377,382,377,382,380,377,340,346,342,379,376,382,379,379,383,342,342,348,379,381,386,378,384,386,346,350,346,386,383,386,378,386,383,337,343,349,378,385,383,381,386,384,339,343,351,382,381,387,381,379,386,344,349,345,379,382,386,381,381,384,348,346,350,377,383,388,382,382,385,340,350,347,377,381,383,382,382,382,338,347,343,379,383,382,382,383,380,342,349,348,376,383,381,378,381,384,341,347,346};
    vector<float> enron     {936,934,938,934,934,938,937,938,937,934,935,938,937,936,938,938,938,937,936,936,937,936,936,936,937,939,938,935,938,937,934,938,937,938,938,936,934,936,937,935,937,937,938,938,938,935,936,935,933,937,937,937,937,939,933,933,933,933,934,935,937,937,937,934,934,933,935,935,937,937,937,937,933,934,936,934,935,938,938,937,937,935,937,939,937,936,938,940,940,939,938,937,936,937,937,939,939,941,940,935,936,938,936,936,939,939,941,941,938,939,936,937,939,941,939,939,940,937,940,939,938,936,940,939,941,940,938,936,939,937,940,940,939,940,939,935,936,939,935,938,938,940,940,939,937,938,939,936,937,939,939,940,940,936,936,938,934,936,940,939,939,939,937,937,939,936,938,940,938,941,941,936,938,939,938,939,940,939,941,941,934,937,940,938,937,940,941,940,941,937,941,940,934,940,939,940,940,941,937,940,940,939,939,939,940,941,941,939,939,939,938,940,940,939,941,941,936,936,937,935,937,938,939,941,941,936,937,939,937,939,939,940,941,941,936,939,938,935,939,938,940,941,941};
    //vector<float> facebook3 {249,247,251,247,246,250,227,230,218,250,247,247,247,252,246,220,230,213,248,248,249,249,247,250,226,229,215,248,248,249,248,248,246,219,228,218,243,248,249,249,247,247,225,232,218,250,250,245,250,245,247,220,228,217,250,249,251,249,253,249,227,230,220,252,250,248,250,249,249,226,230,223,248,250,249,248,250,249,220,229,219,251,250,247,250,247,246,215,221,229,249,251,251,253,248,252,217,226,228,246,248,253,247,252,249,216,224,232,250,246,250,250,249,250,215,222,228,249,249,250,249,250,251,218,222,229,250,251,251,249,254,247,215,226,229,250,251,253,250,251,251,220,227,231,249,250,254,250,252,251,221,223,228,251,248,253,250,246,252,225,222,234,253,246,252,250,248,251,215,219,226,252,252,252,246,247,252,217,220,221,251,249,252,245,249,251,217,225,226,249,251,247,247,250,254,219,219,230,249,247,251,250,248,253,216,221,226,246,253,251,250,250,249,220,226,230,250,249,249,250,250,251,213,217,231,250,251,251,248,248,252,221,217,228,252,249,252,251,252,249,220,220,228};
    vector<float> facebook  {354,356,359,361,359,357,301,304,271,358,360,357,359,357,355,293,307,273,358,356,358,359,360,361,301,313,277,359,361,358,361,362,360,304,311,278,360,358,359,361,359,353,305,304,280,358,355,356,362,354,358,296,304,278,360,357,354,361,360,354,302,310,275,361,359,354,359,360,354,301,307,278,357,357,355,354,359,358,301,312,281,362,362,364,361,362,363,288,300,294,365,357,361,360,362,358,287,299,305,359,358,363,358,361,364,288,296,303,357,362,361,358,364,361,293,303,301,363,362,362,356,362,364,295,297,306,364,365,362,361,361,362,297,298,305,356,364,363,360,365,359,289,298,307,364,360,361,361,362,361,290,306,309,359,364,359,355,363,360,289,298,303,363,360,361,355,357,366,284,286,306,362,361,365,361,363,359,289,294,307,362,367,363,359,362,365,295,292,300,362,366,362,358,366,365,290,302,308,359,363,364,361,361,364,290,292,308,359,363,358,360,366,368,289,304,312,364,362,362,361,365,364,295,297,302,358,357,360,357,358,364,292,294,307,362,363,363,356,364,362,291,297,305};
    //vector<float> facebook7 {436,439,428,433,439,435,343,355,312,439,434,432,440,437,433,349,351,312,436,435,432,438,436,436,344,359,320,437,434,428,436,436,434,343,353,316,437,435,430,439,435,434,342,354,315,435,437,435,438,432,433,349,358,312,433,430,430,429,434,431,351,355,317,439,433,430,431,436,433,343,355,319,436,435,431,436,435,430,347,355,322,436,441,441,441,437,440,337,339,353,438,441,444,439,439,437,334,346,349,436,441,443,439,437,441,335,343,351,440,440,441,439,440,444,338,341,346,439,442,442,444,443,443,343,343,352,436,441,441,443,439,441,336,345,355,440,443,439,440,442,437,340,344,346,440,440,441,436,437,441,346,346,351,436,445,440,438,441,442,339,347,361,439,444,442,441,438,447,329,339,359,434,440,445,432,441,442,332,341,349,441,442,442,437,439,442,332,342,361,440,437,443,441,440,444,334,352,349,442,442,444,441,440,441,337,344,353,441,442,441,440,440,444,337,340,344,442,437,444,436,444,439,336,327,357,437,438,440,443,441,440,334,341,354,440,443,443,436,445,440,343,339,359};
    
    vector<float> BA        {402.1,403.5,401.9,401.8,403.9,402.9,339.9,344.9,326.8,405.4,403.9,402.4,401.7,403.2,402.3,341.8,346.3,327.9,404.8,405,401,404.5,403.2,403,343.3,347.1,329.9,401.3,401.8,401.8,402.4,401.2,397.9,343.3,344.4,326.7,401.6,400.9,400.5,399.1,401.7,401.2,344.2,344.4,328,402.1,402.7,401.4,402.4,403.1,400.2,342,349.1,332,392.1,393.4,390.6,389.1,392.6,392.5,341,341.9,324.8,392.8,394.9,392.3,393.5,393.3,392.8,341.8,344.2,327.1,393,395.5,393.9,392.9,394.1,394.3,340.3,343.9,329.1,410,414.1,414,409.1,413.9,414.2,338.5,344.1,349,412.4,413.6,414.4,409.4,413.8,413.9,337.2,344.6,347.5,412.4,415.1,414.6,410.2,413.6,414.1,342.1,343.2,349,410.4,414.2,414.2,410.7,413.2,411.9,340.2,342.3,347.6,411.6,413,414.1,412.4,412.7,413.2,340.3,348.3,351.7,409.5,413.3,413.4,410.6,412.5,415.3,339.2,345.5,351.7,403.3,406.7,406.8,403.8,404.5,406.2,340.4,342,346.2,404.4,405.2,408.8,403.3,407.9,408,338.7,343.1,348.2,404.6,407.8,408.1,403,406.7,407.2,341.1,343.3,351.3,413.9,415.9,417.5,412.9,417.6,417.6,338.2,341.5,348.4,414.1,419,417.4,414.7,416.9,418.9,340.7,341.9,348.7,412.9,417.7,419.6,413.7,417.4,417.9,337.5,343.4,349.6,414.2,416.4,418.2,414.3,417.3,418.9,340.9,344.3,350.1,412.7,418.4,418.3,413.3,417.6,418.8,338.3,342.9,350.6,416.2,416.3,417.5,414.7,416.7,419.2,341.5,343.5,351.5,407.3,411.1,410.4,406.6,411.8,412.8,338.5,341,347,408.9,411.2,412.8,408.4,409.6,412.8,340,342.4,347.1,409.2,410.6,413.2,408.5,410.8,413.5,340,342.1,350};
    vector<float> ER        {238.6,237.2,237.5,238.3,237.3,237.8,202,205.6,196,238,240.8,238.8,237.9,237.6,238.1,202.5,206.7,196.3,239.7,239.7,239,238.6,238.2,239.1,206.5,207.9,197.8,239,238.6,238.4,236.3,238,238,203.8,206,197.2,238.1,240.4,238.1,239.3,237.9,237.4,203.9,205.9,199.2,237.3,240.2,237.3,237.9,238.4,237.3,204.9,207.1,199.4,233.2,234.3,232.7,232.6,234.4,232.7,202.3,204.4,195.4,234.3,233.4,232.9,235,234.8,233.9,202.9,205.3,196.3,234,234,233.8,235.5,235,234.4,204.6,205.9,198.5,241.3,245.2,244,242.7,246.1,244,200.6,203.6,207,241.8,245.5,245.6,242.3,246.3,244.9,200.4,203.6,208.1,243.1,244.2,245.8,242,244.6,243.3,200.7,204.4,210.2,242.8,244.8,244.6,242,244.7,244.4,202.1,203.5,208.9,243,244.1,245.5,244.2,243.6,244.3,200.4,204.6,210,243.2,244.6,245.2,244.3,244.6,244.5,203.2,206.2,210.5,239.5,240.9,241.4,239.2,238.9,241.3,201.2,202.4,208.4,238.8,242.7,243,239.4,242.8,242.3,202,204.3,206.7,240.9,240.8,242.4,240.2,240.5,242.8,202.3,206.1,207.9,242.6,244,247.7,242.9,245.4,247.2,201.5,202.9,207.2,243.8,247.4,246.3,243.8,246.3,248.1,201.2,205.2,207.4,242.7,247.1,248,243.4,245.7,246.9,200.3,204.5,209.7,243.6,245.6,247.1,243.1,244.9,248.3,201.1,205.5,208.3,245,246.1,246.7,245.3,246.9,246.3,201.2,204,209.2,245,245.9,247.7,244,245.5,248.1,201.2,204.7,210.8,240.1,243,244.8,241.4,243.4,244.9,202.1,202.8,209.4,242.1,242.9,245.5,243.6,244.9,244.8,201,205,207,241.9,244,245.8,241.5,242.8,244.6,202.9,204.2,207.6};
    vector<float> WS        {260.2,265.3,267.5,265.5,266.2,265.4,228,226.7,216.2,267.7,266.8,268,267,268,265,226.6,228.5,214.3,265,265.1,265.7,265.1,265.7,264.1,227.3,226.6,216.6,263.1,263,265.6,265.6,264.3,261.5,224.7,225.7,213.4,265.6,266.7,260.4,264,261.8,262.8,228.3,225.1,215,261.2,263.3,264.8,263.8,261.6,263.7,228.8,226.3,215.6,255.1,250.7,251.3,253.8,254.9,254.4,222.4,221.5,212.7,253.5,253.3,253.7,253.8,256.4,253.8,222.8,221.4,211.4,254.9,253.7,254.6,253.4,256.3,253.2,225.4,223.7,212.6,272.9,276.5,278.8,274.8,278.3,281.8,228,231.5,230.3,275.6,279.6,279.7,274.3,281,281.7,230.8,232.1,231.9,276.6,277.1,281.3,274.2,277.3,279.1,225.8,231.6,234.1,274.4,276.7,277.3,271.7,270.6,276.5,226.8,233.4,231.1,276.6,275.4,280.5,273.9,274.1,277.9,230.2,231.6,230.6,274.3,276.9,277.1,269.5,276.3,278.6,231.1,236.4,232.2,264.1,267.1,274,266.8,267.4,273.6,227,230.8,228.5,268.9,268.2,268,268.6,265.8,269.6,225.9,230.9,227.2,268.8,270.1,272.8,265.8,270.3,270.8,227.4,231,229.4,276.5,282.9,285.7,275.7,281.7,284,229.2,231.1,235.6,276.6,283.8,287.9,279.8,283.9,285.4,229.2,233.8,233.4,278.4,280.3,285.7,278.6,282.9,284,229.8,232.2,236.2,280.2,282.4,285,279,279.9,283.4,228.9,231.4,234.8,277.5,282.3,284.2,277.5,281.4,283.7,232.4,233.8,235.8,278.1,284.5,287.6,280,281.4,282.9,231.5,232,238,268.2,275.4,276.1,272.9,273.6,276.6,227.5,230.5,234.4,270.4,276.1,276.8,269.4,275.9,275.9,226.6,231.9,232.9,272.2,276.8,279.1,271.1,275.6,277.2,226.3,233.3,235.8};

    results.push_back(digg);
    results.push_back(enron);
    //results.push_back(facebook3);
    results.push_back(facebook);
    //results.push_back(facebook7);
    results.push_back(BA);
    results.push_back(ER);
    results.push_back(WS);
    
    vector<string> datasetNames = {"digg", "enron"/*, facebook 3%*/, "facebook"/*, facebook 7%*/, "BA", "ER", "WS"};
    
    
    vector<float> positionAverages;
    vector<float> positionSums;
    vector<float> ranksAverages;
    vector<float> ranksSums;
    
    tie(positionAverages, positionSums, ranksAverages, ranksSums) = getRanksData(results);
    
    
    //vector<int> bestRanksSumsIndexes = getIndexesOfBest(ranksSums, 10, true);
    //coutV("Best ranks sums indexes: ", bestRanksSumsIndexes);-
    
    int bestToFind = 10; // -1
    vector<int> bestPositionAverageIndexes = getIndexesOfBest(positionAverages, min((int)results[0].size(), bestToFind), false);
    coutV("Position on average best indexes: ", bestPositionAverageIndexes);
    
    vector<int> parametersSetsExcelRows = getExcelRowNumbers(bestPositionAverageIndexes, 10);
    coutV("Excel row numbers: ", parametersSetsExcelRows);
    
    for (int i = 0; i < bestPositionAverageIndexes.size(); i++) {
        int index = bestPositionAverageIndexes[i];
        
        //cout << "Sum of ranks: " << ranksSums[index] << endl;
        cout << "Avg position: " << positionAverages[index] << endl;
    }
    
    bool saveResultsCorrelation = true;
    string suffix = "TEST";
    
    if (saveResultsCorrelation) {
        // using ofstream constructors.
        outfile.open("results_correlation_" + suffix + "_.xls");

        outfile << "<?xml version='1.0'?>" << std::endl;
        outfile << "<Workbook xmlns='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
        outfile << " xmlns:o='urn:schemas-microsoft-com:office:office'" << std::endl;
        outfile << " xmlns:x='urn:schemas-microsoft-com:office:excel'" << std::endl;
        outfile << " xmlns:ss='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
        outfile << " xmlns:html='http://www.w3.org/TR/REC-html40'>" << std::endl;
        outfile << " <Worksheet ss:Name='Sheet1'>" << std::endl;
        outfile << "  <Table>" << std::endl;


        outfile << "   <Row>" << std::endl;
        outfile << "    <Cell></Cell>" << std::endl;
        for (int i=0; i<datasetNames.size(); i++) {
            outfile << "    <Cell><Data ss:Type='String'>" + datasetNames[i] + "</Data></Cell>" << std::endl;
        }
        outfile << "   </Row>" << std::endl;


        for (int i=0; i<datasetNames.size(); i++) {
            outfile << "   <Row>" << std::endl;
            outfile << "    <Cell><Data ss:Type='String'>" + datasetNames[i] + "</Data></Cell>" << std::endl;
            for (int j=0; j<datasetNames.size(); j++) {
                if (j > i) {
                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(pearson(results[i], results[j])) + "</Data></Cell>" << std::endl;
                } else {
                    outfile << "    <Cell></Cell>" << std::endl;
                }
            }
            outfile << "   </Row>" << std::endl;
        }


        outfile << "   <Row></Row>" << std::endl;
        outfile << "   <Row></Row>" << std::endl;
        outfile << "   <Row></Row>" << std::endl;


        outfile << "   <Row>" << std::endl;
        outfile << "    <Cell></Cell>" << std::endl;
        for (int i=0; i<datasetNames.size(); i++) {
            outfile << "    <Cell><Data ss:Type='String'>" + datasetNames[i] + "</Data></Cell>" << std::endl;
        }
        outfile << "   </Row>" << std::endl;


        for (int i=0; i<datasetNames.size(); i++) {
            outfile << "   <Row>" << std::endl;
            outfile << "    <Cell><Data ss:Type='String'>" + datasetNames[i] + "</Data></Cell>" << std::endl;
            for (int j=0; j<datasetNames.size(); j++) {
                if (j > i) {
                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(spearman(results[i], results[j])) + "</Data></Cell>" << std::endl;
                } else {
                    outfile << "    <Cell></Cell>" << std::endl;
                }
            }
            outfile << "   </Row>" << std::endl;
        }


        outfile << "  </Table>" << std::endl;
        outfile << " </Worksheet>" << std::endl;
        outfile << "</Workbook>" << std::endl;

        outfile.close();
    } else {
        /*
        cout << endl << endl << "Pearson: " << endl;
        cout << pearson(facebook, digg) << endl;
        cout << pearson(facebook, irvine) << endl;
        cout << pearson(facebook, enron) << endl;
        cout << pearson(digg, irvine) << endl;
        cout << pearson(digg, enron) << endl;
        cout << pearson(irvine, enron) << endl;


        cout << endl << endl << "Spearman: " << endl;
        cout << spearman(facebook, digg) << endl;
        cout << spearman(facebook, irvine) << endl;    
        cout << spearman(facebook, enron) << endl;
        cout << spearman(digg, irvine) << endl;
        cout << spearman(digg, enron) << endl;
        cout << spearman(irvine, enron) << endl;
        */
    }
   
    return 0;
}
