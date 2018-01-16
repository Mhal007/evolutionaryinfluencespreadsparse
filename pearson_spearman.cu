#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <sstream>
#include <dirent.h>

using namespace std;

double mean (vector<double> values) {
    double sum  = 0;
    int    size = values.size();
    for (int i = 0; i < size; i++) {
        sum += values[i];
    }
    
    return sum / size;
}

double pearson_numerator (vector<double> A, vector<double> B, double meanA, double meanB) {
    double numerator = 0;
    for (int i = 0; i < A.size(); i++) {
        numerator += (A[i] - meanA) * (B[i] - meanB);
    }
    
    return numerator;
}

double pearson_denominator (vector<double> A, vector<double> B, double meanA, double meanB) {
    double denominator1;
    double denominator1_sum = 0;
    double denominator2;
    double denominator2_sum = 0;
    
    for (int i = 0; i < A.size(); i++) {
        denominator1_sum += pow(A[i] - meanA, 2);
    }
    
    for (int i = 0; i < B.size(); i++) {
        denominator2_sum += pow(B[i] - meanB, 2);
    }
    
    //cout << denominator1_sum << endl;
    //cout << denominator2_sum << endl;
    
    denominator1 = pow(denominator1_sum, 0.5);
    denominator2 = pow(denominator2_sum, 0.5);
    
    return denominator1 * denominator2;
}

double pearson (vector<double> A, vector<double> B) {
    if (A.size() != B.size()) {
        cout << "ERROR - wrong vector lengths" << endl;
        return -1;
    }
    
    double meanA = mean(A);
    double meanB = mean(B);
    
    double numerator   = pearson_numerator(A, B, meanA, meanB);
    double denominator = pearson_denominator(A, B, meanA, meanB);
    
    //denominator
    
    //cout << "numerator: " << numerator << endl;
    
    return numerator / denominator;
}

vector<double> toRank (vector<double> A) {
    vector<double> sorted = A;
    sort(sorted.begin(), sorted.end());
    
    vector<double> rank;
    
    /*
    cout << endl << "Values: " << endl;
    for (int i = 0; i < A.size(); i++) {
        cout << A[i] << ", ";
    }
    cout << endl << endl;
    */
    
    /*
    cout << "Sorted: " << endl;
    for (int i = 0; i < sorted.size(); i++) {
        cout << sorted[i] << ", ";
    }
    cout << endl << endl;
    */
    
    for (int i = 0; i < A.size(); i++) {
        //cout << endl << "Value: " << A[i] << endl;
        
        
        vector<int> positions;
        //cout << "Positions: " << endl;
        for (int j = 0; j < A.size(); j++) {
            if (sorted[j] == A[i]) {
                positions.push_back(j);
                //cout << j << endl;
            }
        }
        
        double sum = 0;
        double avg;
        
        for (int j = 0; j < positions.size(); j++) {
            sum += positions[j] + 1;
        }
        
        //cout << "Sum: " << sum << endl;
        avg = sum / positions.size();
        //cout << "Avg: " << avg << endl;
        
        rank.push_back(avg);
        //rank.push_back(positions[positions.size()-1] + 1); //libreoffice calc rank
    }
    
    /*
    cout << "Ranking: " << endl;
    for (int i = 0; i < rank.size(); i++) {
        cout << rank[i] << ", ";
    }
    cout << endl << endl;
    */
    
    return rank;
}


double spearman (vector<double> A, vector<double> B) {
    vector<double> A_ranked = toRank(A);
    vector<double> B_ranked = toRank(B);
   
    return pearson(A_ranked, B_ranked);
}


vector<string> getFileNames () {
    DIR *pDIR;
    struct dirent *entry;
    vector<string> fileNames;
    if (pDIR=opendir("./networks/networks")) {
        while (entry = readdir(pDIR)) {
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                fileNames.push_back(entry->d_name);
            }
        }
        closedir(pDIR);
    }

    return fileNames;
}

int main() {
    // facebook
    vector<double> facebook {903,903,948,897,890,850,555,546,494,960,889,955,899,901,862,579,520,506,881,977,884,854,864,858,576,553,516,825,885,844,868,836,853,554,525,514,868,865,829,912,947,823,568,525,517,832,890,864,910,883,811,558,533,528,799,776,797,866,807,807,569,543,501,725,821,800,870,813,887,573,559,516,822,788,763,854,829,821,580,529,512,1005,1062,987,972,988,944,769,638,593,1036,1023,1073,973,1004,979,740,642,576,1027,1083,1052,959,976,952,699,635,600,978,981,982,1050,1011,961,755,669,591,996,994,988,981,995,968,702,647,589,939,992,1008,971,991,937,790,672,604,920,916,961,962,965,939,784,651,586,978,928,930,981,925,933,774,641,599,884,955,975,937,978,967,721,652,617,1055,970,1027,990,1033,1004,866,707,626,1012,1063,1005,1067,1034,980,821,700,631,1005,1041,1083,1014,1045,978,832,716,675,978,1012,1013,1043,926,980,824,688,631,1035,1039,1056,1091,967,1019,805,743,635,990,1004,1036,1024,1001,984,853,747,634,916,980,975,1035,1009,1004,840,754,671,953,939,999,1040,1015,984,828,704,637,938,927,942,997,984,953,810,729,661};
   
    // digg
    vector<double> digg {1271,1379,1402,1448,1378,1284,809,802,682,1372,1241,1393,1342,1333,1344,810,740,731,1337,1321,1371,1295,1351,1292,780,761,724,1333,1281,1348,1309,1423,1259,776,725,694,1233,1258,1354,1321,1286,1282,785,732,703,1242,1424,1211,1426,1374,1300,767,736,722,1205,1232,1193,1245,1204,1321,803,765,718,1207,1161,1158,1265,1256,1222,777,776,736,1291,1241,1340,1324,1331,1290,825,773,709,1564,1662,1767,1575,1622,1586,1043,964,815,1560,1531,1630,1680,1662,1565,1023,919,835,1462,1534,1666,1769,1564,1644,1029,961,833,1333,1556,1576,1561,1487,1587,1093,887,868,1436,1564,1584,1632,1516,1538,1062,960,846,1428,1528,1459,1731,1590,1645,1039,955,856,1291,1446,1483,1504,1459,1600,1125,971,846,1442,1404,1524,1580,1645,1549,1143,961,882,1426,1355,1385,1717,1596,1583,1032,920,860,1586,1481,1541,1566,1591,1519,1073,1038,901,1503,1613,1475,1654,1614,1466,1139,969,895,1517,1697,1521,1586,1681,1581,1114,1022,946,1411,1429,1403,1799,1550,1705,1170,980,948,1542,1486,1608,1602,1689,1596,1190,1093,921,1507,1686,1589,1600,1495,1479,1218,1002,916,1399,1403,1498,1498,1587,1441,1245,1029,881,1381,1415,1439,1507,1695,1535,1141,1047,906,1527,1417,1378,1565,1666,1528,1198,1046,900};
    
    //irvine
    vector<double> irvine {455,455,461,458,456,458,458,457,454,457,455,463,460,456,456,459,455,456,459,458,461,460,456,458,457,457,458,457,460,461,457,457,457,457,459,457,457,458,458,455,459,459,465,459,459,458,455,456,457,457,462,458,455,453,455,458,455,456,457,457,456,455,456,454,455,458,455,456,461,457,455,458,457,457,459,457,455,457,464,456,457,459,458,455,457,456,460,464,462,459,459,455,457,457,457,458,461,457,459,458,461,457,458,458,460,465,458,461,459,459,455,458,458,458,460,460,460,462,456,463,462,458,461,463,458,458,460,461,457,457,458,465,458,460,457,461,457,458,460,459,455,458,460,458,462,463,459,464,458,457,458,458,457,460,458,457,461,456,459,460,457,458,464,458,457,463,457,458,458,459,464,462,456,459,463,457,459,464,460,458,454,459,457,457,458,455,462,459,464,460,461,458,462,457,460,460,461,460,467,461,463,457,462,458,459,460,459,459,458,460,461,461,456,460,458,461,461,459,461,460,460,459,459,457,465,461,462,459,465,460,460,458,462,459,461,462,456,459,459,459,461,458,456};
   
    //enron
    vector<double> enron {8354,4991,8096,8139,7877,8498,7816,6887,7012,8470,8555,7629,8293,8104,8487,7900,6656,4418,8182,7949,4878,8603,8001,8278,7947,8035,7623,5639,5074,7512,8653,7797,8222,7994,4452,4543,7960,8303,8628,8200,8281,8045,7852,7962,4500,7641,7739,8060,8171,8135,8485,8013,7952,4990,8012,8184,8331,8060,8651,8062,8361,7865,4729,8197,5558,8385,8193,8412,8233,7973,7880,4117,8369,5352,5514,8190,8501,8148,8004,4681,4695,8159,8010,8283,8176,8562,8784,8043,8504,7997,8369,7893,7937,8528,8642,8776,8617,8253,7859,8145,8707,8612,8583,8774,8626,8534,8491,8046,6282,7624,8007,8106,8625,8241,8013,8022,8381,7668,8200,5839,8163,8038,8529,8282,8462,8404,8624,6622,5801,8337,8591,8155,8356,8609,8297,7705,8528,7489,8212,8201,8652,8587,8485,8146,8113,8075,8186,8473,8014,8710,8460,8540,7952,7717,8037,8551,8253,8695,8651,8236,8086,7709,8411,5012,7436,8747,8694,8651,8578,8626,8531,8615,7949,8538,8195,8246,8435,8062,8106,8165,8210,8208,8148,8683,8263,8642,8637,8614,8426,8221,8080,7724,8138,8099,8625,8544,8592,8531,7848,8575,5938,8086,8721,8659,8171,7411,8529,8180,7989,8118,8614,8293,8374,8603,8509,8434,8229,8248,8125,8167,8673,8753,8551,8677,8393,7839,7996,8330,8750,8409,8626,8687,8411,8598,5845,5555,8547,8640,8320,8659,8625,8472,820};
    
   
    //vector<double> AA {100, 100, 0.12, 5, 5};
    //toRank(AA);
    
    vector<string> fileNames = getFileNames();
    sort(fileNames.begin(), fileNames.end());
    
    cout << endl << endl;
    for (int i=0; i<fileNames.size(); i++) {
		string name = fileNames[i];
        cout << name << ", ";
		
		stringstream ssname(name);
		string token;
		getline(ssname, token, '-');
		getline(ssname, token, '-');
		
		cout << "N: " << token << endl;
    }
    cout << endl << endl;
	
    
    
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
   
    return 0;
}
