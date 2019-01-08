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

vector<int> G_timestamps;

int getCurrentTime () {
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void F_TIME_START () {
	G_timestamps.push_back(getCurrentTime());
}

void F_TIME_END (string measuredName) {
	int start  = G_timestamps.back();
	int end    = getCurrentTime();
	float diff = (end - start) / 1000.0;

	G_timestamps.pop_back();

	cout << endl << "## [" << measuredName << "]: " << diff << "s" << endl << endl;
}

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

float getInfluenceValue (int N, int inf_values_size, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, int x, int y) {
    float infValue = 0;

    int min = inf_row_ptr[x];
    int max = x == N-1 ? inf_values_size-1 : inf_row_ptr[x+1]; //inf_values_size-1
    
    for (int i=min; i<max; i++) {
        if (inf_col_ind[i] == y) {            
            infValue = inf_values[i];
            break;
        }
    }

    return infValue;
}

void InfluenceSpreadPopulationStep (bool *dyn_activeNodesPerIndividual, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, int N, int nrOfChangedIndividuals, int inf_values_size, float INFLUENCE_THRESHOLD, vector<int>& changedIndividuals) {
	for (int indiv_id = 0; indiv_id < nrOfChangedIndividuals; indiv_id++) {
		for (int node_id = 0; node_id < N; node_id++) {		
			int indiv_index = changedIndividuals[indiv_id];
			float infValue = 0;  // total value of influence on the node
			
			for (int i=0; i<N; i++) {
				if (dyn_activeNodesPerIndividual[indiv_index * N + i] && node_id != i) {  // if i-th element is active and is not the node

					float result = getInfluenceValue(N, inf_values_size, inf_values, inf_col_ind, inf_row_ptr, i, node_id);

					infValue += result;              // add i-th element influence on the node
					//printf("Influence %d on %d is: %f\n", i, node_id, result);
					//printf("\ninfValue: %f, id: %d", infValue, id);
				}
			}

			//printf("\ninfValue: %f, id: %d", infValue, id);
			if (infValue >= INFLUENCE_THRESHOLD) {          // if total influence on the node is greater than or equal to the INFLUENCE_THRESHOLD value
				dyn_activeNodesPerIndividual[indiv_index * N + node_id] = true;           // activate the node
			}
		}
	}
}

vector <vector<float>> readData (string dataset_name, int N, string _EXPERIMENT_ID) {
    vector <vector<float>> influence;
    
    // initialization of the influence vector
    for (int i=0; i<N; i++) {
    	cout << endl << i << " out of " << N << endl;
        vector<float> row(N, 0);
        influence.push_back(row);

        if ((i + 1) * N % (N * N / 10) == 0) {
            cout << "[Initialization of the influence matrix]: " << float((i + 1) * N) / (N * N) * 100 << "%" << endl;
        }
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
			cout << "Reading raw data file, line nr: " << line_nr << endl;
			//cout << line << endl;
			istringstream iss(line);
			int a, b;
            
			if (!(iss >> a >> b)) { cout << "ERROR" << endl; break; } // error

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

						/*cout << i << "'s influence on " << j << " equals: " << influence[i][j] << endl;*/
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

void  defineInfluenceArrayAndVectors (string dataset_name, int N, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, string _EXPERIMENT_ID) {
    //cout << "File reading started." << endl;

    ifstream infile("./experiments-counted/" + dataset_name + "_influenceCounted_" + to_string(N));

    if (infile.good()) { // reading the already calculated influence values
        int    line_nr = 0;
        string line;

        float last_a = -1;
        while (getline(infile, line)) {
            cout << "Reading influence file, line nr: " << line_nr << endl;
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
                //cout << "Influence of " << i << " on " << j << " is equal to: " << influence[i][j] << endl;
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

        cout << "Creating individual " << i << " of " << nrOfIndividuals << endl;

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

void  createPopulationSample (int nrOfIndividuals, int N, int toFind, vector <vector<int>>& population) {
    // creating one individual - used as a sample e.g. for GPU vs CPU tests
	
	vector<int> row;
	population.push_back(row);
	
	for (int x = 0; x<toFind; x++) {
		population[0].push_back(x);
	}
}

void setPopulationFitness (vector<vector<int>>& population, int nrOfIndividuals, int N, int inf_values_size, float& INFLUENCE_THRESHOLD, int STEPS_MAX, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, int toFind, vector<int>& fitness, int THREADS_PER_BLOCK) {
    //bool activeNodesPerIndividual[nrOfIndividuals][N];

    bool *dyn_activeNodesPerIndividual = new bool[nrOfIndividuals*N];
    
    for (int i=0; i<nrOfIndividuals; i++) {
        for (int j=0; j<N; j++) {
        	int index = N * i + j;
        	dyn_activeNodesPerIndividual[index] = false;
        }
        for (int j=0; j<toFind; j++) {
        	int index = N * i + population[i][j];
        	dyn_activeNodesPerIndividual[index] = true;
        }
    }
        
    int active  [nrOfIndividuals];
    vector<int> changedIndividuals;

    for (int i=0; i<nrOfIndividuals; i++) {
        active[i] = toFind;
    	changedIndividuals.push_back(i);
    }

    int step_counter = 0;
    while (step_counter < STEPS_MAX && changedIndividuals.size() > 0) {
        //cout << "Step: " << step_counter << " / " << STEPS_MAX << endl;


    	int nrOfChangedIndividuals = changedIndividuals.size();
    	cout << "nrOfChangedIndividuals   " << nrOfChangedIndividuals << endl;

        F_TIME_START();
        InfluenceSpreadPopulationStep (dyn_activeNodesPerIndividual, inf_values, inf_col_ind, inf_row_ptr, N, nrOfChangedIndividuals, inf_values_size, INFLUENCE_THRESHOLD, changedIndividuals);
        F_TIME_END("host functions");


        changedIndividuals.clear();
        int curr_active;
        for (int i=0; i<nrOfIndividuals; i++) {
            curr_active = 0;

            for (int j=0; j<N; j++) {
            	int index = N * i + j;
                if (dyn_activeNodesPerIndividual[index]) {
                    curr_active++;
                }
            }

            if (curr_active != active[i]) {
            	changedIndividuals.push_back(i);
            }
            active[i] = curr_active;
        }
        
        step_counter++;
    }

    for (int i = 0; i < nrOfIndividuals; i++) {
        int individualFitness = 0;
        
        for (int j = 0; j < N; j++) {
        	int index = N * i + j;

            if (dyn_activeNodesPerIndividual[index]) {
                individualFitness++;
                //cout << "Activated " << j << endl;
            }
        }
        //cout << "individualFitness: " << individualFitness << endl;
        //cout << "toFind: " << toFind << endl;
            
        // acceptable `error`
        /*if (individualFitness-toFind < 0) {
            cout << "# Crossover/mutation overlapping" << endl;    // can happen because of random crossover and mutation
            //coutIndividual(population, i);
        }*/
        
        //cout << "fitness Indiv: " << i << ":    " << individualFitness-toFind << endl;
        fitness.push_back(individualFitness-toFind);
    }
}

void  performPopulationSelection (vector<vector<int>>& population, int& nrOfIndividuals, int N, int inf_values_size, float& INFLUENCE_THRESHOLD, int& groupSize, int& STEPS_MAX, vector<float>& inf_values, vector<float>& inf_col_ind, vector<float>& inf_row_ptr, int& toFind, int& max_fitness_value, vector<int>& max_fitness_individual, int THREADS_PER_BLOCK) {
    vector<int> fitness;

    F_TIME_START();
    setPopulationFitness(population, nrOfIndividuals, N, inf_values_size, INFLUENCE_THRESHOLD, STEPS_MAX, inf_values, inf_col_ind, inf_row_ptr, toFind, fitness, THREADS_PER_BLOCK);
    F_TIME_END("selection - fitness count");
    

    F_TIME_START();
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

    F_TIME_END("selection - population swapping");
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

bool anyLimitReached(int resultBufferSize, float resultMinDiff, vector<int> &resultsBuffer, int generation, int generationsLimit, float timeLimit, int COMPUTATION_START_TIME, int result, int resultLimit) {
    int   now  = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    float diff = (now - COMPUTATION_START_TIME) / 1000.0;
	
    bool anyLimit = 
	   (resultMinDiff    > 0 && generation >  resultBufferSize && result < resultsBuffer[0] * (1 + resultMinDiff))
    || (generationsLimit > 0 && generation >= generationsLimit)
    || (resultLimit      > 0 && result     >= resultLimit)
    || (timeLimit        > 0 && diff       >= timeLimit);
	
	if (generation > 0) {
		resultsBuffer.push_back(result);
	}
	
	if (generation > resultBufferSize) {
		resultsBuffer.erase(resultsBuffer.begin());
		//cout << endl << "Current resultsBuffer[0]: " << resultsBuffer[0] << endl;
	}
	
	return anyLimit;
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

vector<float> toRank (vector<float> A) {
    vector<float> sorted = A;
    sort(sorted.begin(), sorted.end());
    
    vector<float> rank;
        
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


int main (int argc, char* argv[]) {
    srand (time(NULL));
    coutGPUStatus();

    string _EXPERIMENT_ID = argv[1];

    int tests = 100;
    
    float timeLimit        = 6; //seconds
    int   generationsLimit = 0; //5;
    int   resultLimit      = 0; //32;
    int   resultBufferSize = 10;
	float resultMinDiff    = 0; //0.01;

    bool saveResults            = true;
    bool saveResultsCorrelation = true;

    float INFLUENCE_THRESHOLD = 0.5;
    int   N_MAX               = 1000;
	int   STEPS_MAX           = 10000;
    int   TO_FIND_PERCENTAGE  = 5;
    int   THREADS_PER_BLOCK   = 1024;

    /*   Parameters    */
    //int groupSize          = 20;                 // 10,    20,   30                        // 2,    5,    10,   20,  50
    //int nrOfIndividuals    = (int)ceil(N/10.0);  // N/20,  N/10, N/5                       // 100,  500   1k,   2k,  10k
    //float crossover_ratio  = 0.7;                // 0.5,   0.7,  0.9                       // 0.1,  0.3,  0.5,  0.7, 0.9
    //float mutation_potency = 0.01;               // 0.001, 0.01, 0.1                       // 0.01, 0.02, 0.05, 0.1, 0.2
    //float mutation_ratio   = 0.9;                // 0.75,  0.9,  0.95,                     // 0.1,  0.3,  0.5,  0.7, 0.9 

    vector<int>   a_groupSize        {10,    20,   30};   // 10,    20,   30
    vector<int>   a_nrOfIndividuals  {12,    10,   8};    // N/12,  N/10, N/8
    vector<float> a_crossover_ratio  {0.6,   0.7,  0.8};  // 0.6,   0.7,  0.8
    vector<float> a_mutation_potency {0.001, 0.01, 0.1};  // 0.001, 0.01, 0.1
    vector<float> a_mutation_ratio   {0.7,   0.8,  0.9};  // 0.7,   0.8,  0.9

	int parameters_sets = a_groupSize.size() * a_nrOfIndividuals.size() * a_crossover_ratio.size() * a_mutation_potency.size() * a_mutation_ratio.size();

    vector<string> datasets = getFileNames("./experiments_" + _EXPERIMENT_ID);

    /* DEBUG */
    int debug_nrOfIndividuals;
    bool debug = true;
    if (debug) {
    	tests = 10;

    	N_MAX = 1000;
    	THREADS_PER_BLOCK = 1024;
    	debug_nrOfIndividuals = -1; // -1 - the same as if it wasn't a debug mode (so devides N by a_nrOfIndividuals to get indivnr)
		
		// tests: 10, debug_nrOfIndividuals: -1, generationsLimit: 1, THREADS_PER_BLOCK: 1024, default parameters, facebook
		/* 100: 7 in 1ms, 500: 46 in 10ms, 1000: 88 in 53ms */

    	timeLimit        = 0;
    	generationsLimit = 5; //  5 - 80s
    	resultLimit      = 0;
    	resultMinDiff    = 0;

    	saveResults = true;//false;
    	saveResultsCorrelation = true;//false;

    	a_groupSize        = {20};
    	a_nrOfIndividuals  = {8};
    	a_crossover_ratio  = {0.7};
    	a_mutation_potency = {0.01};
    	a_mutation_ratio   = {0.9};
    	parameters_sets = a_groupSize.size() * a_nrOfIndividuals.size() * a_crossover_ratio.size() * a_mutation_potency.size() * a_mutation_ratio.size();

    	//datasets = {"facebook-46952"};
    	//datasets = {"BA-1000-1-3.csv"};
    	datasets = {"ER-1000-0.05-10.csv"};
		//datasets = getFileNames("./experiments_" + _EXPERIMENT_ID);
    }

    /*
    N = 1000
    INDIVIDUALS = 1000
		THREADS_PER_BLOCK = 192
			1    individuals - 0.056s
			10   individuals - 0.081s
			100  individuals - 0.265s
			1000 individuals - 2.483s

		THREADS_PER_BLOCK = 512
			1000 individuals - 2.423s

		THREADS_PER_BLOCK = 1024
			1000 individuals - 2.481s

	N = max (~47k for facebook)
		THREADS_PER_BLOCK = 512
			100 individuals - 5.08s
    */


	
	vector<vector<float>> results;
	 for (int i=0; i<datasets.size(); i++) {
        vector<float> row(parameters_sets, -1);
        results.push_back(row);
    }
    
    
    for (int file_id=0; file_id<datasets.size(); file_id++) {
		int dataset_id = file_id;		//TODO to refactor
		
		string dataset_name = datasets[file_id];
		
		stringstream ssname(dataset_name);
		string token;
		getline(ssname, token, '-');
		getline(ssname, token, '-');
		
        
		int maxSize = stoi(token);
   		int N       = min(N_MAX, maxSize);
		int toFind  = (int)ceil(float(TO_FIND_PERCENTAGE * N) / 100.0);
    
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
        int parameters_set = 1;

        for_each(a_groupSize.begin(), a_groupSize.end(), [&] (int groupSize) {
            for_each(a_nrOfIndividuals.begin(), a_nrOfIndividuals.end(), [&] (int nrOfIndividualsRaw) {
                int nrOfIndividuals = (int)ceil(N/nrOfIndividualsRaw);
                
                if (debug && debug_nrOfIndividuals != -1) {
                    nrOfIndividuals = debug_nrOfIndividuals;
                }

                for_each(a_crossover_ratio.begin(), a_crossover_ratio.end(), [&] (float crossover_ratio) {
                    for_each(a_mutation_potency.begin(), a_mutation_potency.end(), [&] (float mutation_potency) {
                        for_each(a_mutation_ratio.begin(), a_mutation_ratio.end(), [&] (float mutation_ratio) {
                            float testsResultsSum     = 0;
                            float testsGenerationsSum = 0;
                            float testsTimeSum        = 0;
                            
                            for (int test = 0; test < tests; test++) {
                                vector <int> max_fitness_individual;
                                vector <vector<int>> population;

                                int max_fitness_value = -1;
                                int progressBarLength = 10;

                                int generation = 0;
                                vector<int> resultsBuffer;

                                createPopulation(nrOfIndividuals, N, toFind, population);
                                //createPopulationSample(nrOfIndividuals, N, toFind, population);
                                //coutPopulation(population);


                                int COMPUTATION_START_TIME = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

                                while (!anyLimitReached(resultBufferSize, resultMinDiff, resultsBuffer, generation, generationsLimit, timeLimit, COMPUTATION_START_TIME, max_fitness_value, resultLimit)) {

                                    //coutGPUStatus();

                                	F_TIME_START();
                                    performPopulationSelection(population, nrOfIndividuals, N, inf_values_size, INFLUENCE_THRESHOLD, groupSize, STEPS_MAX, inf_values, inf_col_ind, inf_row_ptr, toFind, max_fitness_value, max_fitness_individual, THREADS_PER_BLOCK);
                                    F_TIME_END("selection");

                                	F_TIME_START();
                                    performCrossover(population, nrOfIndividuals, crossover_ratio, toFind);
                                    F_TIME_END("crossover");


                                	F_TIME_START();
                                    performMutation(population, nrOfIndividuals, mutation_ratio, mutation_potency, toFind, N);
                                    F_TIME_END("mutation");

                                    //coutResult(generation, max_fitness_value);

                                    generation++;
                                }
                                
                                int COMPUTATION_END_TIME = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
								int COMPUTATION_DURATION = COMPUTATION_END_TIME - COMPUTATION_START_TIME;

                                cout << endl << "[FINISHED] test:  " << test+1 << "/" << tests 
                                    << "  for parameters set nr:  " << parameters_set << "/" << parameters_sets 
                                    << "  for dataset_id:  " << dataset_id+1 << "/" << datasets.size() 
                                    << "  in: " << COMPUTATION_DURATION / 1000.0 << "s";
                                    
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

                                //cout << endl << "result " << test+1 << ": " << max_fitness_value << endl;
                                testsResultsSum     += max_fitness_value;
                                testsGenerationsSum += generation;
								testsTimeSum        += COMPUTATION_DURATION;

                                /*cout << "Best individual found: " << endl;
                                for (int i=0; i<max_fitness_individual.size(); i++) {
                                    cout << max_fitness_individual[i] << ", ";
                                }*/

                                //cout << endl << endl << "This group can activate " << max_fitness_value << " others";
                                //cout << endl << "Time elapsed: " << (time2 - COMPUTATION_START_TIME) / 1000.0 << "s" << endl;

                            } // TEST
                            
                            float finalResult      = std::round(testsResultsSum     / tests);
                            float finalGenerations = std::round(testsGenerationsSum / tests);
                            float finalTime        = std::round(testsTimeSum        / tests);
                            
                            cout << endl << "Final result avg: " << finalResult << " in avg " << finalTime / 1000.0 << "s" << endl;
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
                        });
                    });
                });
            });
        });

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
