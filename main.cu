#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include <dirent.h>

using namespace std::chrono;
using namespace std;

void coutGPUStatus () {
    size_t freem, totalm;
    float free_m, total_m, used_m;

    cudaMemGetInfo((size_t*)&freem, (size_t*)&totalm);

    free_m  = (size_t) freem  / 1048576.0;
    total_m = (size_t) totalm / 1048576.0;
    used_m  = total_m - free_m;

    printf ( "## Total: %f MB. Used %f MB. Free: %f MB. \n", total_m, used_m, free_m);
}
void coutResult(int& generation, int& max_fitness_value) {
    cout << "Generation " << generation << ", currently best individual can activate " << max_fitness_value << " others" << endl;
}
void coutInfluenceArray (int N, vector<vector<float>>& influence) {
    cout << "Influence Array: \n";
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            float val = influence[i][j];
            if (val == 0) {
                cout << val << ",     ";
            } else if (int(val * 1000) % 100 == 0) {
                cout << val << ",   ";
            } else {
                cout << val << ",  ";
            }
        }
        cout << "\n";
    }
    cout << "\n\n";
}
void coutValuesVector (vector<float> &inf_values) {
    cout << "Values vector: (" << inf_values.size() << ")\n";
    for (int i=0; i<inf_values.size(); i++) {
        cout << inf_values[i] << ", ";
    }
    cout << "\n\n";
}
void coutColIndVector (vector<float> &inf_col_ind) {
    cout << "ColInd vector: (" << inf_col_ind.size() << ")\n";
    for (int i=0; i<inf_col_ind.size(); i++) {
        cout << inf_col_ind[i] << ", ";
    }
    cout << "\n\n";
}
void coutRowPtrVector (vector<float> &inf_row_ptr) {
    cout << "RowPtr vector: (" << inf_row_ptr.size() << ")\n";
    for (int i=0; i<inf_row_ptr.size(); i++) {
        cout << inf_row_ptr[i] << ", ";
    }
    cout << "\n\n";
}
void coutInfluenceArraySize (vector <vector<float>>& influence) {
    cout << "Size of influence array: " << influence.size()*influence.size()*sizeof(float) << "\n\n";
}
void coutVectorsSize (vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr) {
    cout << "Total size of vectors: "
         <<   inf_values.size()  * sizeof(float)
            + inf_col_ind.size() * sizeof(float)
            + inf_row_ptr.size() * sizeof(float) << "\n\n";
}
void coutPopulation (vector <vector<int>>& population) {
    cout << "Population:";
    for (int i=0; i<population.size(); i++) {
        cout << "\nIndiv: " << i << ":   ";

        for (int j=0; j<population[i].size(); j++) {
            if (population[i][j] < 10) {
                cout << population[i][j] << ",    ";
            }
            else if (population[i][j] < 100) {
                cout << population[i][j] << ",   ";
            }
            else if (population[i][j] < 1000) {
                cout << population[i][j] << ",  ";
            }
            else if (population[i][j] < 10000) {
                cout << population[i][j] << ", ";
            }
            else {
                cout << population[i][j] << ",";
            }
        }
    }
    cout << "\n\n";
}

void coutIndividual (vector <vector<int>>& population, int i) {
    cout << "Individual " << i << ":";
    for (int j=0; j<population[i].size(); j++) {
        if (population[i][j] < 10) {
            cout << population[i][j] << ",    ";
        }
        else if (population[i][j] < 100) {
            cout << population[i][j] << ",   ";
        }
        else if (population[i][j] < 1000) {
            cout << population[i][j] << ",  ";
        }
        else if (population[i][j] < 10000) {
            cout << population[i][j] << ", ";
        }
        else {
            cout << population[i][j] << ",";
        }
    }
    cout << "\n\n";
}

void coutProgressBar (bool code, int size) {
    string element = code ? "[#]" : "[ ]";

    for (size_t i=0; i<size; i++) {
        cout << element;
    }
    if (code) cout << "100%\n";
    else      cout << "0%\n";
}
void coutProgressBar (int x, int y, int size) {
    int val = std::round((float)x/y*size);

    for (int i=0; i<size; i++) {
        if(i<val) cout << "[#]";
        else      cout << "[ ]";
    }
    cout << val*size << "%" << endl;
}

float timeDiff (timespec &start, timespec &end) {
    if (end.tv_nsec-start.tv_nsec < 0) {
        return ((end.tv_sec - start.tv_sec - 1) * 1000000000 + (end.tv_nsec - start.tv_nsec + 1000000000)) / 1000000 / (float)1000;
    } else {
        return ((end.tv_sec - start.tv_sec)     * 1000000000 + (end.tv_nsec - start.tv_nsec)) / 1000000 / (float)1000;
    }
}

__device__ float getInfluenceValue (int N, int inf_values_size, float* d_inf_values, float* d_inf_col_ind, float* d_inf_row_ptr, int x, int y) {
    float infValue = 0;

    int min = d_inf_row_ptr[x];
    int max = x == N-1 ? inf_values_size-1 : d_inf_row_ptr[x+1]; //inf_values_size-1

    //printf("min: %d\n", min);
    //printf("max: %d\n", max);
    
    for (int i=min; i<max; i++) {
        //printf("val: %f\n", d_inf_values[i]);
    
        if (d_inf_col_ind[i] == y) {            
            infValue = d_inf_values[i];
            break;
        }
    }

    return infValue;
}

__global__ void warmUp ()    // GPU warm-up before the main calculations - for better, stable results
{
    int x = 0;
    for (size_t i=0; i<1000; i++) {
        x++;
    }
}

__global__ void InfluenceSpreadPopulationStep (bool *d_boolPopulMatrix, float *d_inf_values, float *d_inf_col_ind, float *d_inf_row_ptr, int N, int nrOfIndividuals, int inf_values_size, float threshold, bool *d_changed)
{
    //for (int q=0; q<1000; q++) { 
        int id = blockIdx.x * blockDim.x + threadIdx.x; // unique ID number
        int indiv_id = id / N;
        int node_id  = id % N;
        
        //printf("(Indiv, node): (%d, %d)\n", indiv_id, node_id);
        
        if (indiv_id < nrOfIndividuals && node_id < N && d_changed[indiv_id]) {
            float infValue = 0;                      // total value of influence on the node
            for (int i=0; i<N; i++) {
                if (d_boolPopulMatrix[indiv_id * N + i] && node_id != i) {  // if i-th element is active and is not the node
                    
                    //infValue += inf(i, node_id)
                    
                    float result = getInfluenceValue(N, inf_values_size, d_inf_values, d_inf_col_ind, d_inf_row_ptr, i, node_id);
                        
                    infValue += result;              // add i-th element influence on the node
                    //printf("Influence %d on %d is: %f\n", i, node_id, result);
                    //printf("\ninfValue: %f, id: %d", infValue, id);
                }
            }
                
            //printf("Total influence on %d is: %f\n", node_id, infValue);
                
            if (infValue >= threshold) {          // if total influence on the node is greater than or equal to the threshold value
                //printf("\ninfValue: %f", infValue);
                d_boolPopulMatrix[indiv_id * N + node_id] = true;           // activate the node
            }
        }
    //}
}

vector <vector<float>> readData (string dataset_name, int N, string _EXPERIMENT_ID) {
    vector <vector<float>> influence;
    
    // initialization of the influence vector
    for (int i=0; i<N; i++) {
        cout << "Initialization of the influence matrix, step: " << i*N << "/" << N*N << endl;
        vector<float> row(N, 0);
        influence.push_back(row);
    }

    // total number of interactions received by every node
    vector<float> received(N, 0);

    ifstream infile("./experiments_" + _EXPERIMENT_ID + "/" + dataset_name);
    string   line;
	
	
	int _csv_id_hack = -1;
	if (dataset_name.find(".csv") != std::string::npos) {
		_csv_id_hack = 0;
	}
	
    if (infile.good()) {
		int line_nr = 0;
		while (getline(infile, line)) {
			//cout << "Reading line nr: " << line_nr << endl;
			//cout << line << endl;
			istringstream iss(line);
			int a, b;
			if (!(iss >> a >> b)) { cout << "ERROR" << endl; break; } // error

			//cout << "a: " << a << ", b: " << b << endl;
			//cout << "a: " << a + _csv_id_hack << ", b: " << b + _csv_id_hack;
			//cout << ", N: " << N << endl;

			if (a != b && a + _csv_id_hack < N && b + _csv_id_hack < N) {
				influence[a + _csv_id_hack][b + _csv_id_hack] += 1;   // temp inf_values, calculating the total number of iteractions from "i" to "j"
				received [b + _csv_id_hack]                   += 1;

				//cout << "message from " << a + _csv_id_hack << " to " << b + _csv_id_hack << endl;
			}
			line_nr++;
		}

		infile.close();

		cout << "File reading finished successfully." << endl;

		ofstream outfile ("./experiments-counted/" + dataset_name + "_influenceCounted_" + to_string(N));

		if (outfile.good()) {
			// Influence value calculated as the ratio of iteractions from "i" node to "j" node, to the total number of iteractions to the "j" node.
			for (int i=0; i<N; i++) {
				for (int j=0; j<N; j++) {
					//cout << "Influence values calculations, step: " << i*N+(j+1) << "/" << N*N << endl;

					if (i == j) {
						outfile << i << " " << j << " " << -1 << "\n";
						influence[i][j] = -1;
					} else if (influence[i][j] > 0) {
						if (received[j] != 0) {
							influence[i][j] = influence[i][j] / received[j];
						} else if (influence[i][j] != 0) {
							cout << "Received array error";
						}

						/*cout << "saving inf values" << " from " << i << " to " << j << " it's: " << influence[i][j] << endl;*/
						outfile << i << " " << j << " " << influence[i][j] << "\n";
					} else {
						influence[i][j] = 0;
					}
				}
			}

			cout << "Compressed file saved successfully." << endl;

			outfile.close();
		} else {
        	throw std::invalid_argument("readData - File " + dataset_name + " not saved.");
		}
	} else {
        throw std::invalid_argument("readData - File " + dataset_name + " not found.");
	}

    return influence;
}

void arrayBoolToIndivVector (bool* individual, int size, vector<int>& intIndividual) {
    intIndividual.clear();
    intIndividual.swap(intIndividual);

    cout << endl << endl << "INDIVIDUAL: " << endl;

    for (int i=0; i<size; i++) {
        cout << individual[i] << ",";
        if (individual[i]) {
            intIndividual.push_back(i);
        }
    }
}

void  defineInfluenceArrayAndVectors (string dataset_name, int N, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, string _EXPERIMENT_ID) {
    //cout << "File reading started." << endl;

    ifstream infile("/experiments-counted/" + dataset_name + "_influenceCounted_" + to_string(N));

    if (infile.good()) { // reading the already calculated influence values
        int    line_nr = 0;
        string line;

        float last_a = -1;
        while (getline(infile, line)) {
            //cout << "Reading line nr: " << line_nr << endl;
            istringstream iss(line);
            float a, b, c;
            if (!(iss >> a >> b >> c)) { break; } // error

            if (c != 0) {
                if (a != last_a) {
                    inf_row_ptr.push_back(inf_values.size());
                    //cout << "add row ptr: " << inf_values.size() << endl;
                    last_a = a;
                }
                inf_values.push_back(c);
                //cout << "add value: " << c << endl;
                inf_col_ind.push_back(b);
                //cout << "add col ind: " << b << endl;
            }
            line_nr++;
        }

        infile.close();
    } else { // calculating influnce values
        infile.close();
        vector <vector<float>> influence = readData(dataset_name, N, _EXPERIMENT_ID);

        // inf_values, inf_col_ind, inf_row_ptr creation, based on the influence array
        for (int i=0; i<N; i++) {
            bool added = false;
            for (int j=0; j<N; j++) {
                //cout << "Influence " << i << " on " << j << " is: " << influence[i][j] << endl;
                if (influence[i][j] != 0) {
                    if (!added) {
                        inf_row_ptr.push_back(inf_values.size());
                        //cout << "add row ptr: " << inf_values.size() << endl;
                        added = true;
                    }
                    inf_values.push_back(influence[i][j]);
                    //cout << "add value: " << influence[i][j] << endl;
                    inf_col_ind.push_back(j);
                    //cout << "add col ind: " << j << endl;
                }
            }

            if (!added) {
                //inf_row_ptr.push_back(-1);
            }
        }


        //Removing influence matrix from memory
        /*influence.clear();
        influence.shrink_to_fit();
        vector<float>().swap(influence);*/

        /*cout << "\n\n size of influence array: " << sizeof(influence) + sizeof(float) * influence.capacity() * influence.capacity();
        cout << "\n\n Total size of vectors: "
                << sizeof(inf_values) + sizeof(float) * inf_values.capacity()
                    + sizeof(inf_col_ind) + sizeof(float) * inf_col_ind.capacity()
                    + sizeof(inf_row_ptr) + sizeof(float) * inf_row_ptr.capacity() << "\n\n";*/
    }
}

void  createPopulation (int nrOfIndividuals, int N, int toFind, vector <vector<int>>& population) {
    // creating random individuals within population
    for (int i = 0; i<nrOfIndividuals; i++) {
        vector<int> row;
        population.push_back(row);

        for (int j = 0; j<toFind; j++) {
            int rand_id = rand() % N;

            bool alreadyAdded = true;
            while (alreadyAdded) {
                alreadyAdded = false;
                for (int k=0; k<population[i].size(); k++) {
                    if (population[i][k] == rand_id) {
                        alreadyAdded = true;
                        rand_id      = rand() % N;
                    }
                }
            }
            //cout << "pushing: " << rand_id << endl;
            population[i].push_back(rand_id);
        }
    }
}

void populVectorToBoolMatrix (vector<vector<int>>& population, bool **boolPopulMatrix, int nrOfIndividuals, int toFind) {
    for (int i=0; i<nrOfIndividuals; i++) {
        for (int j=0; j<toFind; j++) {
            boolPopulMatrix[i][population[i][j]] = true;
        }
    }
}

void setFitnessFromBoolMatrix (bool **boolPopulMatrix, int nrOfIndividuals, int N, vector<int>& fitness) {
    int curr_fitness = 0;
    for (int i=0; i<nrOfIndividuals; i++) {
        for (int j=0; j<N; j++) {
            if(boolPopulMatrix[i][j]) {
                curr_fitness++;
            }
        }
        fitness.push_back(curr_fitness);
    }
}

void setPopulationFitness (vector<vector<int>>& population, int nrOfIndividuals, int N, int inf_values_size, float& threshold, int maxSteps, float *d_inf_values, float *d_inf_col_ind, float *d_inf_row_ptr, bool* d_boolPopulMatrix, bool* d_changed, int toFind, vector<int>& fitness) {
        
    bool boolPopulMatrix[nrOfIndividuals][N];

    
    for (int i=0; i<nrOfIndividuals; i++) {
        for (int j=0; j<N; j++) {
            boolPopulMatrix[i][j] = false;
        }
        for (int j=0; j<toFind; j++) {
            boolPopulMatrix[i][population[i][j]] = true;
        }
    }
    
    /*
    for (int i=0; i<nrOfIndividuals; i++) {
        for (int j=0; j<N; j++) {
            cout << boolPopulMatrix[i][j] << ", ";
        }
        cout << "\n";
    }
    */

    cudaError_t err;

    err = cudaMemcpy(d_boolPopulMatrix, boolPopulMatrix, sizeof(bool)*nrOfIndividuals*N, cudaMemcpyHostToDevice);    
    if (err != cudaSuccess) {
        cout << "Error copying boolPopulMatrix to d_boolIndiv." << endl;
        cudaFree(d_boolPopulMatrix);
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    int active  [nrOfIndividuals];
    int changed [nrOfIndividuals];
    for (int i=0; i<nrOfIndividuals; i++) {
        active[i]  = toFind;
        changed[i] = true;
    }   

    
    int step_counter = 0;
    bool atLeastOneChanged = true;
    while (step_counter < maxSteps && atLeastOneChanged)
    {
        //cout << "Step: " << step_counter << " / " << maxSteps << endl;
        
        err = cudaMemcpy(d_changed, changed, sizeof(bool)*nrOfIndividuals, cudaMemcpyHostToDevice);    
        if (err != cudaSuccess) {
            cout << "Error copying changed to d_changed." << endl;
            cudaFree(d_changed);
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
                            
        InfluenceSpreadPopulationStep << <nrOfIndividuals*N/192 + 1, min(192, nrOfIndividuals*N) >> >(d_boolPopulMatrix, d_inf_values, d_inf_col_ind, d_inf_row_ptr, N, nrOfIndividuals, inf_values_size, threshold, d_changed);
        //cudaDeviceSynchronize();
                
        err = cudaMemcpy(boolPopulMatrix, d_boolPopulMatrix, sizeof(bool)*nrOfIndividuals*N, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cout << "Error copying d_boolPopulMatrix to boolPopulMatrix" << endl;
            cudaFree(d_boolPopulMatrix);
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }    
              
        int curr_active;
        atLeastOneChanged = false;
        for (int i=0; i<nrOfIndividuals; i++) {
            curr_active = 0;
            for (int j=0; j<N; j++) {
                if (boolPopulMatrix[i][j]) {
                    curr_active++;
                }
            }
            changed[i] = curr_active != active[i];
            if (!atLeastOneChanged && changed[i]) {
                atLeastOneChanged = true;
            }
            active[i] = curr_active;
        }
        
        step_counter++;
    }    
            
    err = cudaMemcpy(boolPopulMatrix, d_boolPopulMatrix, sizeof(bool)*nrOfIndividuals*N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "Error copying d_boolPopulMatrix to boolPopulMatrix" << endl;
        cudaFree(d_boolPopulMatrix);
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }    
    
    int curr_fitness;
    for (int i=0; i<nrOfIndividuals; i++) {
        curr_fitness = 0;
        
        //cout << "###### START OF " << i << " ######" << endl;
        
        for (int j=0; j<N; j++) {
            if (boolPopulMatrix[i][j]) {
                curr_fitness++;
                //cout << "Activated " << j << endl;
            }
        }
        //cout << "curr_fitness: " << curr_fitness << endl;
        //cout << "toFind: " << toFind << endl;
            
        // acceptable `error`
        /*if (curr_fitness-toFind < 0) {
            cout << "# Crossover/mutation overlapping" << endl;    // can happen because of random crossover and mutation
            //coutIndividual(population, i);
        }*/
        
        //cout << "fitness Indiv: " << i << ":    " << curr_fitness-toFind << endl;
        //cout << "###### END OF " << i << " ######" << endl;
        fitness.push_back(curr_fitness-toFind);
    }
}

void  performPopulationSelection (vector<vector<int>>& population, int& nrOfIndividuals, int N, int inf_values_size, float& threshold, int& groupSize, int& maxSteps, float *d_inf_values, float *d_inf_col_ind, float *d_inf_row_ptr, bool* d_boolPopulMatrix, bool* d_changed, int& toFind, int& max_fitness_value, vector<int>& max_fitness_individual) {
    vector<int> fitness;
                
    setPopulationFitness(population, nrOfIndividuals, N, inf_values_size, threshold, maxSteps, d_inf_values, d_inf_col_ind, d_inf_row_ptr, d_boolPopulMatrix, d_changed, toFind, fitness);
    
    vector<vector<int>> newPopulation;

    while (newPopulation.size() != population.size()) {
        vector<int> newGroup;
        bool alreadyAdded[nrOfIndividuals];
        for (int i=0; i<nrOfIndividuals; i++) {
            alreadyAdded[i] = false;
        }

        for (int j=0; j<groupSize; j++) {
            int randIndiv = rand() % nrOfIndividuals;

            while (alreadyAdded[randIndiv]) {
                randIndiv = rand() % nrOfIndividuals;
            }
            newGroup.push_back(randIndiv);
        }

        int curr_best_fitness = -1;
        int curr_best_id      = -1;
        int currentFitness    = -1;
        
        for (int j=0; j<newGroup.size(); j++) {
            currentFitness = fitness[newGroup[j]];            

            if (currentFitness > curr_best_fitness) {
                curr_best_fitness = currentFitness;
                curr_best_id      = j;
            }
        }
        
        newPopulation.push_back(population[newGroup[curr_best_id]]);

        if (curr_best_fitness > max_fitness_value) {
            max_fitness_individual = population[newGroup[curr_best_id]];
            max_fitness_value = curr_best_fitness;
        }
    }
    population = newPopulation;
}


// TODO performCrossover on DEVICE (nrOfIndividuals/2 threads (from 0 to nr/2 - 1), ids: id*2, id*2+1
void  performCrossover (vector<vector<int>>& population, int& nrOfIndividuals, float& crossover_ratio, int& toFind) {
    float split_ratio = 0.5;
    float split_point = split_ratio*toFind;

    int id_first  = -1;
    int id_second = -1;

    for (int i=0; i<nrOfIndividuals; i++) {
        int cross = rand() % 100;
        if (cross < crossover_ratio * 100) {
            if (id_first == -1) {
                id_first = i;
            } else {
                id_second = i;
            }
        }
        if (id_second != -1) {
            for (int j=0; j<split_point; j++) {
                float temp = population[id_first][j];
                population[id_first][j] = population[id_second][j];
                population[id_second][j] = temp;
            }
            id_first = -1;
            id_second = -1;
        }
    } // allows to node doubling (fitness = -1 can happen)
}

// TODO performMutation on DEVICE
void  performMutation (vector<vector<int>>& population, int& nrOfIndividuals, float& mutation_ratio, float& mutation_potency, int& toFind, int N) {
    for (int i=0; i<nrOfIndividuals; i++) {
        int mutation = rand() % 100;
        if (mutation < mutation_ratio * 100) {
            for (int j=0; j<mutation_potency*toFind; j++) {
                population[i][rand() % toFind] = rand() % N;
            }
        }
    } // allows to node doubling (fitness = -1 can happen)
}

bool anyLimitReached(bool isRelativeResultLimit, int resultStep, float resultMinDiff, vector<int> &resultsBuffer, bool isGenerationsLimit, int generation, int generationsLimit, bool isTimeLimit, float timeLimit, int start, bool isResultLimit, int result, int resultLimit) {
    int   now  = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    float diff = (now - start) / 1000.0;
	
	/*cout << endl << endl << "Generation " << generation << " started.";
	cout << endl << "results buffer: ";
	for (int i = 0; i < resultsBuffer.size(); i++) {
		cout << resultsBuffer[i] << ", ";
	}
	cout << endl << "Previous result: " << result;
	if (generation > resultStep) {
		cout << endl << "Comparing with: " << resultsBuffer[0];
	} else {
		cout << endl << "No comparison, not enough results";
	}
 	cout << endl << endl;
    */
    
    bool anyLimit = 
       isRelativeResultLimit && generation >  resultStep && result < resultsBuffer[0] * (1 + resultMinDiff) 
    || isGenerationsLimit    && generation >= generationsLimit 
    || isResultLimit         && result     >= resultLimit
    || isTimeLimit           && diff       >= timeLimit;
	
	if (generation > 0) {
		resultsBuffer.push_back(result);
	}
	
	if (generation > resultStep) {
		resultsBuffer.erase(resultsBuffer.begin());
		//cout << endl << "Current resultsBuffer[0]: " << resultsBuffer[0] << endl;
	}
	
	return anyLimit;
}

void  performWarmup () {
    warmUp << <1024, 1024>> >();
}

void copyToDevice (vector<float>& element, int& size, float *d_element) {
    if (cudaMalloc(&d_element, size) != cudaSuccess) {
        cout << "Error allocating memory for d_element." << endl;
    }

    float* elementArray = &element[0];

    if (cudaMemcpy(d_element, elementArray, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cout << "Error copying influence to d_influence." << endl;
        cudaFree(d_element);
    }
}

void mallocOnDevice (int& size, bool* d_element) {
    if (cudaMalloc(&d_element, size) != cudaSuccess) {
        cout << "Error allocating memory for d_boolIndiv." << endl;
    }
}

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
    
    //cout << denominator1_sum << endl;
    //cout << denominator2_sum << endl;
    
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
    
    //denominator
    
    //cout << "numerator: " << numerator << endl;
    
    return numerator / denominator;
}

vector<float> toRank (vector<float> A) {
    vector<float> sorted = A;
    sort(sorted.begin(), sorted.end());
    
    vector<float> rank;
    
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
        
        float sum = 0;
        float avg;
        
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


float spearman (vector<float> A, vector<float> B) {
    vector<float> A_ranked = toRank(A);
    vector<float> B_ranked = toRank(B);
   
    return pearson(A_ranked, B_ranked);
}



// ### REFACTORING ###
// TODO Unify parameters order
// TODO Unify parameters and variables naming
// TODO Couts (introduce log file saving, main console only crucial information (phase)
// TODO Delete references for const variables

int main (int argc, char* argv[]) {
    srand (time(NULL));
    coutGPUStatus();

    string _EXPERIMENT_ID = argv[1];

    int tests = 100;
    
    bool  isTimeLimit = true;
    float timeLimit   = 6;  //seconds

    bool isGenerationsLimit = false;
    int  generationsLimit   = 5;

    bool isResultLimit = false;
    int  resultLimit   = 32;
    
    bool  isRelativeResultLimit = false;
    int   resultStep            = 10;
	float resultMinDiff         = 0.01;

    bool saveResults            = true;
    bool saveResultsCorrelation = true;



    
    /*   Parameters    */
    //int groupSize          = 20;                 // 10,    20,   30                        // 2,    5,    10,   20,  50
    //int nrOfIndividuals    = (int)ceil(N/10.0);  // N/20,  N/10, N/5                       // 100,  500   1k,   2k,  10k
    //float crossover_ratio  = 0.7;                // 0.5,   0.7,  0.9                       // 0.1,  0.3,  0.5,  0.7, 0.9
    //float mutation_potency = 0.01;               // 0.001, 0.01, 0.1                       // 0.01, 0.02, 0.05, 0.1, 0.2
    //float mutation_ratio   = 0.9;                // 0.75,  0.9,  0.95,                     // 0.1,  0.3,  0.5,  0.7, 0.9 

    int   a_groupSize       [3] = {10,    20,   30};   // 10,    20,   30
    int   a_nrOfIndividuals [3] = {12,    10,   8};    // N/12,  N/10, N/8
    float a_crossover_ratio [3] = {0.6,   0.7,  0.8};  // 0.6,   0.7,  0.8
    float a_mutation_potency[3] = {0.001, 0.01, 0.1};  // 0.001, 0.01, 0.1
    float a_mutation_ratio  [3] = {0.7,   0.8,  0.9};  // 0.7,   0.8,  0.9
	int parameters_sets = 3 * 3 * 3 * 3 * 3;

    // 3 wartości ^ 5 parametrów * 30 min * 3 datasety = 243 * 90min = 15 dni + 4,5h

    int percToFind  = 5;
	int maxSteps    = 10000;
    float threshold = 0.5;
	
	
    vector<string> datasets = getFileNames("./experiments_" + _EXPERIMENT_ID);
	
	vector<vector<float>> results;
	 for (int i=0; i<datasets.size(); i++) {
        vector<float> row(parameters_sets, -1);
        results.push_back(row);
    }
    
    //cout << endl << endl;
    for (int file_id=0; file_id<datasets.size(); file_id++) {
		int dataset_id = file_id;		//TODO to refactor
		
		string dataset_name = datasets[file_id];
        //cout << dataset_name << ", ";
		
		stringstream ssname(dataset_name);
		string token;
		getline(ssname, token, '-');
		getline(ssname, token, '-');
		
		//cout << "N: " << token << endl;
		
		int maxSize = stoi(token);
   		int N       = min(1000, maxSize);
		int toFind  = (int)ceil(float(percToFind * N) / 100.0);
    
		// using ofstream constructors.
		std::ofstream outfile("results_" + dataset_name + "_" + _EXPERIMENT_ID + "_" + ".xls");

		if (saveResults) {
			outfile << "<?xml version='1.0'?>" << std::endl;
			outfile << "<Workbook xmlns='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
			outfile << " xmlns:o='urn:schemas-microsoft-com:office:office'" << std::endl;
			outfile << " xmlns:x='urn:schemas-microsoft-com:office:excel'" << std::endl;
			outfile << " xmlns:ss='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
			outfile << " xmlns:html='http://www.w3.org/TR/REC-html40'>" << std::endl;
			outfile << " <Worksheet ss:Name='Sheet1'>" << std::endl;
			outfile << "  <Table>" << std::endl;
			outfile << "   <Row>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>Dataset</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>Test nr</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>groupSize</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>nrOfIndividuals</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>crossover_ratio</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>mutation_potency</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>mutation_ratio</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>Generations</Data></Cell>" << std::endl;
			outfile << "    <Cell><Data ss:Type='String'>Result</Data></Cell>" << std::endl;
			outfile << "   </Row>" << std::endl;
		}

		vector <float> inf_col_ind;
		vector <float> inf_row_ptr;
		vector <float> inf_values;

		defineInfluenceArrayAndVectors(dataset_name, N, inf_values, inf_col_ind, inf_row_ptr, _EXPERIMENT_ID);
        


        int inf_values_size = inf_values.size();

        //for (int i=0; i<inf_values_size; i++) {
        //    cout << "i: " << inf_values[i] << endl;
        //}

        cudaError_t err;
        
        
        

        float* d_inf_values;        
        err = cudaMalloc(&d_inf_values, sizeof(float) * inf_values.size());

        if (err != cudaSuccess) {
            cout << "Error allocating memory for d_inf_values." << endl;
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        float* inf_valuesArray = &inf_values[0];
        err = cudaMemcpy(d_inf_values, inf_valuesArray, sizeof(float)*inf_values.size(), cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            cout << "Error copying inf_valuesArray to d_inf_values." << endl;
            cudaFree(d_inf_values);
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }



        float* d_inf_col_ind;
        err = cudaMalloc(&d_inf_col_ind, sizeof(float) * inf_col_ind.size());

        if (err != cudaSuccess) {
            cout << "Error allocating memory for d_inf_col_ind." << endl;
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        float* inf_col_indArray = &inf_col_ind[0];
        err = cudaMemcpy(d_inf_col_ind, inf_col_indArray, sizeof(float)*inf_col_ind.size(), cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            cout << "Error copying inf_col_indArray to d_inf_col_ind." << endl;
            cudaFree(d_inf_col_ind);
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }



        float* d_inf_row_ptr;
        err = cudaMalloc(&d_inf_row_ptr, sizeof(float) * inf_row_ptr.size());

        if (err != cudaSuccess) {
            cout << "Error allocating memory for d_inf_row_ptr." << endl;
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        float* inf_row_ptrArray = &inf_row_ptr[0];
        err = cudaMemcpy(d_inf_row_ptr, inf_row_ptrArray, sizeof(float)*inf_row_ptr.size(), cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            cout << "Error copying influence to d_inf_row_ptr." << endl;
            cudaFree(d_inf_row_ptr);
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }



        bool  *d_boolIndiv;        
        err = cudaMalloc(&d_boolIndiv, sizeof(bool)*N);     

        if (err != cudaSuccess) {
            cout << "Error allocating memory for d_boolIndiv." << endl;
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        

        int parameters_set = 1;
        for (int groupSize_i = 0; groupSize_i < sizeof(a_groupSize)/sizeof(*a_groupSize); groupSize_i++) {
            int groupSize = a_groupSize[groupSize_i];

            for (int nrOfIndividuals_i = 0; nrOfIndividuals_i < sizeof(a_nrOfIndividuals)/sizeof(*a_nrOfIndividuals); nrOfIndividuals_i++) {
                int nrOfIndividuals = (int)ceil(N/a_nrOfIndividuals[nrOfIndividuals_i]);
                
                
                
                bool* d_boolPopulMatrix;       
                err = cudaMalloc(&d_boolPopulMatrix, sizeof(bool)*nrOfIndividuals*N);

                if (err != cudaSuccess) {
                    cout << "Error allocating memory for d_boolPopulMatrix." << endl;
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    return 0;
                }



                bool* d_changed;  
                err = cudaMalloc(&d_changed, sizeof(bool) * nrOfIndividuals);    
                if (err != cudaSuccess) {
                    cout << "Error allocating memory for d_changed." << endl;
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    return 0;
                }
                


                for (int crossover_ratio_i = 0; crossover_ratio_i < sizeof(a_crossover_ratio)/sizeof(*a_crossover_ratio); crossover_ratio_i++) {
                    float crossover_ratio = a_crossover_ratio[crossover_ratio_i];

                    for (int mutation_potency_i = 0; mutation_potency_i < sizeof(a_mutation_potency)/sizeof(*a_mutation_potency); mutation_potency_i++) {
                        float mutation_potency = a_mutation_potency[mutation_potency_i];

                        for (int mutation_ratio_i = 0; mutation_ratio_i < sizeof(a_mutation_ratio)/sizeof(*a_mutation_ratio); mutation_ratio_i++) {
                            float mutation_ratio  = a_mutation_ratio[mutation_ratio_i];
                            
                            float testsResultsSum     = 0;
                            float testsGenerationsSum = 0;
                            
                            for (int test = 0; test < tests; test++) {
                                vector <int> max_fitness_individual;
                                vector <vector<int>> population;

                                int max_fitness_value = -1;
                                int progressBarLength = 10;

                                int generation = 0;
                                vector<int> resultsBuffer;

                                createPopulation(nrOfIndividuals, N, toFind, population);
                                //coutPopulation(population);


                                performWarmup();


                                int start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                                int temp1;
                                int temp2 = start;        

                                while (!anyLimitReached(isRelativeResultLimit, resultStep, resultMinDiff, resultsBuffer, isGenerationsLimit, generation, generationsLimit, isTimeLimit, timeLimit, start, isResultLimit, max_fitness_value, resultLimit)) {

                                    //coutGPUStatus();

                                    //coutProgressBar(generation,generationsLimit,progressBarLength);

                                    temp1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                                    //cout << endl << "[progressBar]: " << (temp1 - temp2) / 1000.0 << "s"<< endl;


                                    performPopulationSelection(population, nrOfIndividuals, N, inf_values_size, threshold, groupSize, maxSteps, d_inf_values, d_inf_col_ind, d_inf_row_ptr, d_boolPopulMatrix, d_changed, toFind, max_fitness_value, max_fitness_individual);


                                    temp2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                                    //cout << endl << "[selection]: " << (temp2 - temp1) / 1000.0 << "s"<< endl;


                                    performCrossover(population, nrOfIndividuals, crossover_ratio, toFind);


                                    temp1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                                    //cout << endl << "[crossover]: " << (temp1 - temp2) / 1000.0 << "s"<< endl;


                                    performMutation(population, nrOfIndividuals, mutation_ratio, mutation_potency, toFind, N);


                                    temp2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                                    //cout << endl << "[mutation]: " << (temp2 - temp1) / 1000.0 << "s"<< endl;

                                    //coutResult(generation, max_fitness_value);

                                    generation++;
                                }

                                //coutProgressBar(true,progressBarLength);
                                
                                cout << endl << "[FINISHED] test:  " << test+1 << "/" << tests 
                                    << "  for parameters set nr:  " << parameters_set << "/" << parameters_sets 
                                    << "  for dataset_id:  " << dataset_id+1 << "/" << datasets.size() 
                                    << "  in: " << (temp2 - start) / 1000.0 << "s";
                                    
                                cout << endl;    
                                coutGPUStatus();
                                cout << endl;   
                                

                                if (saveResults) {
                                    outfile << "   <Row>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(parameters_set)    + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(test+1)            + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(groupSize)         + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(nrOfIndividuals)   + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(crossover_ratio)   + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(mutation_potency)  + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(mutation_ratio)    + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(generation)        + "</Data></Cell>" << std::endl;
                                    outfile << "    <Cell><Data ss:Type='Number'>" + to_string(max_fitness_value) + "</Data></Cell>" << std::endl;
                                    outfile << "   </Row>" << std::endl;
                                }

                                testsResultsSum     += max_fitness_value;
                                //cout << endl << "result " << test+1 << ": " << max_fitness_value << endl;
                                testsGenerationsSum += generation;

                                /*cout << "Best individual found: " << endl;
                                for (int i=0; i<max_fitness_individual.size(); i++) {
                                    cout << max_fitness_individual[i] << ", ";
                                }*/

                                //cout << endl << endl << "This group can activate " << max_fitness_value << " others";
                                //cout << endl << "Time elapsed: " << (temp2 - start) / 1000.0 << "s" << endl;

                                // clearing the memory 
                                // vector<float>().swap(inf_col_ind);
                                // vector<float>().swap(inf_row_ptr);
                                // vector<float>().swap(inf_values);
                                // cudaFree (?)

                            } // TEST
                            
                            float finalResult      = std::round(testsResultsSum     / tests);
                            float finalGenerations = std::round(testsGenerationsSum / tests);
                            
                            cout << endl << "Final result: " << finalResult << endl;
                            results[file_id][parameters_set-1] = finalResult;
                            
                            if (saveResults) {
                                outfile << "   <Row>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(parameters_set)   + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='String'>AVG                                </Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(groupSize)        + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(nrOfIndividuals)  + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(crossover_ratio)  + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(mutation_potency) + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(mutation_ratio)   + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(finalGenerations) + "</Data></Cell>" << std::endl;
                                outfile << "    <Cell><Data ss:Type='Number'>" + to_string(finalResult)      + "</Data></Cell>" << std::endl;
                                outfile << "   </Row>" << std::endl;
                            }

                            parameters_set++;
                        }
                    }
                }
            }
        }

		if (saveResults) {
			outfile << "  </Table>" << std::endl;
			outfile << " </Worksheet>" << std::endl;
			outfile << "</Workbook>" << std::endl;
		}
		outfile.close();
	}
	
	cout << endl << endl << "*** RESULTS ***" << endl;
	
	for (int i=0; i<datasets.size(); i++) {
		for (int j=0; j<parameters_sets; j++) {
			cout << results[i][j] << ", ";
		}
		cout << endl;
	}
	

    if (saveResultsCorrelation) {
        // using ofstream constructors.
        std::ofstream outfile("results_correlation_" + _EXPERIMENT_ID + "_.xls");

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
        for (int i=0; i<datasets.size(); i++) {
            outfile << "    <Cell><Data ss:Type='String'>" + datasets[i] + "</Data></Cell>" << std::endl;
        }
        outfile << "   </Row>" << std::endl;


        for (int i=0; i<datasets.size(); i++) {
            outfile << "   <Row>" << std::endl;
            outfile << "    <Cell><Data ss:Type='String'>" + datasets[i] + "</Data></Cell>" << std::endl;
            for (int j=0; j<datasets.size(); j++) {
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
        for (int i=0; i<datasets.size(); i++) {
            outfile << "    <Cell><Data ss:Type='String'>" + datasets[i] + "</Data></Cell>" << std::endl;
        }
        outfile << "   </Row>" << std::endl;


        for (int i=0; i<datasets.size(); i++) {
            outfile << "   <Row>" << std::endl;
            outfile << "    <Cell><Data ss:Type='String'>" + datasets[i] + "</Data></Cell>" << std::endl;
            for (int j=0; j<datasets.size(); j++) {
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
    }

    return 0;
}