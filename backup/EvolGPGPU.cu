#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <math.h>

using namespace std;
using namespace std::chrono;


#define TIMELIMIT false //10

#define MAX_STEPS_VAR 10000	

#define MAX_GEN 1000000
#define MIN_TIME 8
//#define MIN_GEN 50
#define SPLIT_RATIO 0.5					// defines the SPLIT (ratio of left and right parts)
#define THRESHOLD 0.5					// Default nodes' threshold value


__global__ void warmUp()	// GPU warm-up before main calculations for better, stable results
{
	int x = 0;
	for (int i = 0; i < 1000; i++)
	{
		x++;
	}
}

__global__ void populationInfluenceStep(float *inf, bool *state, float *changes, bool *continue_steps, int N)
{
	/*int indiv_id = blockIdx.x;		// investigated indiv. id: {0,...,INDIV_NR_VAR}
	int node_id = threadIdx.x;		// investigated node id: {0,...,N}
	*/

	int id = blockIdx.x * blockDim.x + threadIdx.x; // unique ID number
	int indiv_id = id / N;
	int node_id = id % N;

	//printf("indiv_id: %d", indiv_id);
	//printf(", continue: %d\n", continue_steps[indiv_id]);		// always FALSE (????)


	//if (continue_steps[INDIV_NR_VAR])	// checking the control number
	if (!state[indiv_id*N + node_id])
	{
		float node_inf_val = 0;		// total value of influence on the node

		for (int i = 0; i < N; i++)
		{
			if (state[indiv_id * N + i] && node_id != i)	// if i-th element is active and is not the node
			{
				node_inf_val += inf[i * N + node_id];				// add i-th element influence on the node
			}
		}
		
		//printf("Total influence on %d is: %f\n", node_id, node_inf_val);
		
		if (node_inf_val >= THRESHOLD)							// if total influence on the node is greater than or equal to the threshold value
		{
			state[indiv_id * N + node_id] = true;				// activate the node
			changes[indiv_id] = changes[indiv_id] + 1;		// increase the changes number, for CPU to see that this step changed current individual
		}   //akceptowalne wywlaszczanie na wspolnej zmiennej
	}
}



int main()
{
	int N;
	int SEEDS_NR;
	int SPLIT;
	int N_TESTS = 14;
	int N_VAR_A[N_TESTS];
	//int MIN_ACT;
	//FLOAT MAX_TIME;
	float GOAL;

	for (int i = 0; i < N_TESTS; i++)
	{
		N_VAR_A[i] = 100 * (i + 1);
	}


	int TESTRUNS = 1;
	int DATASET = 3;
	bool MEASURINGTIME = true;
	
	
	/*DATASET 1 ... TIMES */
	/*float GOALS[N_TESTS];
	GOALS[0] = 0.1
	GOALS[1] = 0.823
	GOALS[2] = 2.926
	GOALS[3] = 6.048
	GOALS[4] = 13.26
	GOALS[5] = 22.256
	GOALS[6] = 27.721
	GOALS[7] = 42.286
	GOALS[8] = 67.519
	GOALS[9] = 91.058
	GOALS[10] = 111.21
	GOALS[11] = 158.227
	GOALS[12] = 204.236
	GOALS[13] = 260.533*/

	/*DATASET 2 ... RESULTS *//*
	/*float GOALS[N_TESTS];
	GOALS[0] = 11;
	GOALS[1] = 21;
	GOALS[2] = 43;
	GOALS[3] = 79;
	GOALS[4] = 125;
	GOALS[5] = 172;
	GOALS[6] = 207;
	GOALS[7] = 232;
	GOALS[8] = 291;
	GOALS[9] = 337;
	GOALS[10] = 391;
	GOALS[11] = 440;
	GOALS[12] = 478;
	GOALS[13] = 557;*/

	/*DATASET 2 ... TIMES 
	float GOALS[N_TESTS];
	GOALS[0] = 0.034;
	GOALS[1] = 0.218;
	GOALS[2] = 0.637;
	GOALS[3] = 1.743;
	GOALS[4] = 4.059;
	GOALS[5] = 11.274;
	GOALS[6] = 18.257;
	GOALS[7] = 28.475;
	GOALS[8] = 44.965;
	GOALS[9] = 71.265;
	GOALS[10] = 75.038;
	GOALS[11] = 149.483;
	GOALS[12] = 162.765;
	GOALS[13] = 225.978;*/

	
	/*DATASET 3 ... RESULTS */
	float GOALS[N_TESTS];
	GOALS[0] = 18;
	GOALS[1] = 32;
	GOALS[2] = 60;
	GOALS[3] = 82;
	GOALS[4] = 115;
	GOALS[5] = 154;
	GOALS[6] = 169;
	GOALS[7] = 204;
	GOALS[8] = 243;
	GOALS[9] = 278;
	GOALS[10] = 309;
	GOALS[11] = 314;
	GOALS[12] = 359;
	GOALS[13] = 411;
	
	/*DATASET 3 ... TIMES 
	float GOALS[N_TESTS];
	GOALS[0] = 0.039;
	GOALS[1] = 0.271;
	GOALS[2] = 0.773;
	GOALS[3] = 2.199;
	GOALS[4] = 3.756;
	GOALS[5] = 7.861;
	GOALS[6] = 9.975;
	GOALS[7] = 20.916;
	GOALS[8] = 35.284;
	GOALS[9] = 51.509;
	GOALS[10] = 62.504;
	GOALS[11] = 82.568;
	GOALS[12] = 96.643;
	GOALS[13] = 136.718;*/
		
		
		
		
		
		

	float RESULTS[N_TESTS];


	for (int ns = 0; ns < 1;/*N_TESTS;*/ ns++)
	{
		N = 200;//N_VAR_A[ns];
		SEEDS_NR = (int)ceil(N/20.0);
		SPLIT = SPLIT_RATIO*SEEDS_NR;
		//MIN_ACT = GOALS[ns];
		//MAX_TIME = GOALS[ns];
		GOAL = 32;//GOALS[ns];
		

		float influence[N][N];		
		int connections[N];

		// Setting the inital values
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				influence[i][j] = 0;
			}
			connections[i] = 0;
		}


		const int max_chars_per_line_const = 512;
		const int max_tokens_per_line_const = 20;
		const char* const delimiter_const = " ";

		string dataset_path;

		switch (DATASET)
		{
			case 1:
			{
					  dataset_path = "../datasets/opsahl-ucforum/out.opsahl-ucforum";
					  break;
			}
			case 2:
			{
					  dataset_path = "../datasets/munmun_digg_reply/out.munmun_digg_reply";
					  break;
			}
			case 3:
			{
					  dataset_path = "../datasets/facebook-wosn-wall/out.facebook-wosn-wall";
					  break;
			}
			case 4:
			{
					  dataset_path = "../test";
					  break;
			}
			default:
			{
					   cout << "Wrong number, try again." << endl;
					   return 1;
			}
		}

		ifstream fin;
		fin.open(dataset_path);

		if (!fin.good())
		{
			cout << "File not found.";
			return 1;
		}

		while (!fin.eof())
		{
			char buf[max_chars_per_line_const];
			fin.getline(buf, max_chars_per_line_const);

			int n = 0;
			const char* token[max_tokens_per_line_const] = {};

			token[0] = strtok(buf, delimiter_const);
			if (token[0])
			{
				for (n = 1; n < max_tokens_per_line_const; n++)
				{
					token[n] = strtok(0, delimiter_const);
					if (!token[n]) break;
				}
				if (atoi(token[0]) != atoi(token[1]) && atoi(token[0]) - 1 < N && atoi(token[1]) - 1 < N)
				{
					influence[atoi(token[0]) - 1][atoi(token[1]) - 1] += 1;  // calculating the total number of iteractions from "i" to "j"
					connections[atoi(token[1]) - 1] += 1; // calculating the total number of received interactions by the "j" node
				}
			}
		}

		for (int i = 0; i < N; i++) // Influence value calculated as the ratio of iteractions from "i" node to "j" node, to the total number of received iteractions by the "j" node.
		{
			for (int j = 0; j < N; j++)
			{
				if (connections[j] != 0)
				{
					influence[i][j] = influence[i][j] / connections[j];
				}
			}
		}

		float *d_influence;

		if (cudaMalloc(&d_influence, sizeof(float)*N *N) != cudaSuccess)
		{
			cout << "Error allocating memory for d_influence." << endl;
			return 1;
		}


		// Copying initial values from Host to Device
		if (cudaMemcpy(d_influence, influence, sizeof(float)* N *N, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cout << "Error copying influence to d_influence." << endl;
			cudaFree(d_influence);
			return 1;
		}
		/*

		int INDIV_TESTS = 2;
		int GROUP_SIZE_TESTS = 1;
		int MERGE_RATIO_TESTS = 2;
		int MUTATION_RATIO_TESTS = 2;
		int MUTATION_SCALE_TESTS = 1;

		int INDIV_NR_VAR_A[INDIV_TESTS];
		int GROUP_SIZE_VAR_A[GROUP_SIZE_TESTS];
		float MERGE_RATIO_VAR_A[MERGE_RATIO_TESTS];
		float MUTATION_RATIO_VAR_A[MUTATION_RATIO_TESTS];
		float MUTATION_SCALE_VAR_A[MUTATION_SCALE_TESTS];


		INDIV_NR_VAR_A[0] = 50;			INDIV_NR_VAR_A[1] = 100;		//INDIV_NR_VAR_A[2] = 100;			INDIV_NR_VAR_A[3] = 100;			INDIV_NR_VAR_A[4] = 100;
		GROUP_SIZE_VAR_A[0] = 80;		//GROUP_SIZE_VAR_A[1] = 80;		//GROUP_SIZE_VAR_A[2] = 100;		GROUP_SIZE_VAR_A[3] = 100;			GROUP_SIZE_VAR_A[4] = 100;
		MERGE_RATIO_VAR_A[0] = 0.7;		MERGE_RATIO_VAR_A[1] = 0.9;		//MERGE_RATIO_VAR_A[2] = 100;		MERGE_RATIO_VAR_A[3] = 100;			MERGE_RATIO_VAR_A[4] = 100;
		MUTATION_RATIO_VAR_A[0] = 0.7;  MUTATION_RATIO_VAR_A[1] = 0.9;	//MUTATION_RATIO_VAR_A[2] = 0.9;		//MUTATION_RATIO_VAR_A[3] = 100;	MUTATION_RATIO_VAR_A[4] = 100;
		MUTATION_SCALE_VAR_A[0] = 0.01;	//MUTATION_SCALE_VAR_A[1] = 0.01; MUTATION_SCALE_VAR_A[1] = 1;		//MUTATION_SCALE_VAR_A[3] = 100;	MUTATION_SCALE_VAR_A[4] = 100;


		int INDIV_NR_VAR = INDIV_NR_VAR_A[0];
		int GROUP_SIZE_VAR = GROUP_SIZE_VAR_A[0];
		float MERGE_RATIO_VAR = MERGE_RATIO_VAR_A[0];
		float MUTATION_RATIO_VAR = MUTATION_RATIO_VAR_A[0];
		float MUTATION_SCALE_VAR = MUTATION_SCALE_VAR_A[0];*/


		warmUp << <1024, 1024 >> >();


		int test = 0;

		/*float RESULTS[INDIV_TESTS][GROUP_SIZE_TESTS][MERGE_RATIO_TESTS][MUTATION_RATIO_TESTS][MUTATION_SCALE_TESTS];
	
		for (int in = 0; in < INDIV_TESTS; in++)
		{
			INDIV_NR_VAR = INDIV_NR_VAR_A[in];
			for (int gs = 0; gs < GROUP_SIZE_TESTS; gs++)
			{
				GROUP_SIZE_VAR = GROUP_SIZE_VAR_A[gs];
				for (int mer = 0; mer < MERGE_RATIO_TESTS; mer++)
				{
					MERGE_RATIO_VAR = MERGE_RATIO_VAR_A[mer];
					for (int mur = 0; mur < MUTATION_RATIO_TESTS; mur++)
					{
						MUTATION_RATIO_VAR = MUTATION_RATIO_VAR_A[mur];
						for (int mus = 0; mus < MUTATION_SCALE_TESTS; mus++)
						{
							MUTATION_SCALE_VAR = MUTATION_SCALE_VAR_A[mus];

							RESULTS[in][gs][mer][mur][mus] = 0;
							*/
							
							RESULTS[ns] = 0;
							//OPTYMALNE
							/*int MAX_STEPS_VAR = 5;
							int INDIV_NR_VAR = 50;
							int GROUP_SIZE_VAR = 80;
							float MERGE_RATIO_VAR = 0.9;
							float MUTATION_RATIO_VAR = 0.9;
							float MUTATION_SCALE_VAR = 0.1;*/

							//MOJE
							int INDIV_NR_VAR = N/10;	
							int GROUP_SIZE_VAR = 20;
							float MERGE_RATIO_VAR = 0.7;
							float MUTATION_RATIO_VAR = 0.9;
							float MUTATION_SCALE_VAR = 0.01;



							INDIV_NR_VAR   = 1;
							GROUP_SIZE_VAR = 1;




							float durationTotal = 0;
							float sumOfActivated = 0;

							for (int avg = 0; avg < TESTRUNS; avg++)
							{
								// time - start
								high_resolution_clock::time_point beginning = high_resolution_clock::now();


								/* ----- population declaration ----- */
								float population[INDIV_NR_VAR][SEEDS_NR];		// whole population array
								bool state[INDIV_NR_VAR][N];					// every indiv. states 
								bool finalStates[INDIV_NR_VAR][N];				// defines return array of INDIV_NR_VAR x N states after influence spread process, revalueted within each generation
								float changes_before[INDIV_NR_VAR];				// used for counting changes in indiv. in phase of rating during selection
								float changes_after[INDIV_NR_VAR];				// as above, for comparison reasons
								bool continue_steps[INDIV_NR_VAR];				// 


								srand(time(NULL));
								/* ----- initial population determination ----- */
								for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv.
								{
									for (int j = 0; j < N; j++) // N nodes
									{
										state[i][j] = false;
										finalStates[i][j] = false;
									}
								}

								for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv.
								{
									for (int j = 0; j < SEEDS_NR; j++) // N nodes
									{
										population[i][j] = -1;

										int rand_id = rand() % N;
										while (state[i][rand_id])
										{
											rand_id = rand() % N;
										}
										state[i][rand_id] = true;
									}

									changes_before[i] = -1;
									changes_after[i] = 0;
									continue_steps[i] = true;
								}

								

								/* ----- selection ----- */

								// Allocating memory for GPU matrices
								bool *d_state;
								float *d_changes;
								bool *d_continue_steps;

								if (cudaMalloc(&d_state, sizeof(bool)*INDIV_NR_VAR *N) != cudaSuccess)
								{
									cout << "Error allocating memory for d_state." << endl;
									cudaFree(d_influence);
									return 1;
								}
								if (cudaMalloc(&d_changes, sizeof(float)*INDIV_NR_VAR) != cudaSuccess)
								{
									cout << "Error allocating memory for d_changes." << endl;
									cudaFree(d_influence); cudaFree(d_state);
									return 1;
								}
								if (cudaMalloc(&d_continue_steps, sizeof(bool)*INDIV_NR_VAR) != cudaSuccess)
								{
									cout << "Error allocating memory for d_continue_steps." << endl;
									cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes);
									return 1;
								}




								int gen = 0;
								int curr_max = 0;
								int curr_max_id = 0;
								int total_max = 0;
								float curr_time = 0;

								while (!TIMELIMIT && MEASURINGTIME && curr_max<GOAL || !TIMELIMIT && !MEASURINGTIME && curr_time / 1000 < GOAL || TIMELIMIT && MEASURINGTIME && curr_max<GOAL && curr_time / 1000 < GOAL)
								//while (curr_time / 1000 < MAX_TIME)
								//while (gen<MAX_GEN)
								//while (curr_time/1000 < MIN_TIME || curr_max<MIN_ACT && curr_time/1000 < MAX_TIME)
								{
									if (gen > 0)
									{
										/* ----- initial population determination ----- */
										for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv.
										{
											for (int j = 0; j < N; j++) // N nodes
											{
												finalStates[i][j] = false;
												state[i][j] = false;
											}

											for (int j = 0; j < SEEDS_NR; j++) // SEEDS_NR nodes
											{
												state[i][int(population[i][j])] = true;
											}

											changes_before[i] = -1;
											changes_after[i] = 0;
											continue_steps[i] = true;
										}
									}


									if (cudaMemcpy(d_state, state, sizeof(bool)*INDIV_NR_VAR *N, cudaMemcpyHostToDevice) != cudaSuccess)
									{
										cout << "Error copying state to d_state." << endl;
										cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
										return 1;
									}
									if (cudaMemcpy(d_changes, changes_after, sizeof(float)*INDIV_NR_VAR, cudaMemcpyHostToDevice) != cudaSuccess)
									{
										cout << "Error copying changes_after to d_changes." << endl;
										cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
										return 1;
									}
									if (cudaMemcpy(d_continue_steps, continue_steps, sizeof(bool)*INDIV_NR_VAR, cudaMemcpyHostToDevice) != cudaSuccess)
									{
										cout << "Error copying continue_steps to d_continue_steps." << endl;
										cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
										return 1;
									}


									// Individuals elauation process
									int step_counter = 0;
									int finished_counter = 0;
									bool changed;
									while (step_counter < MAX_STEPS_VAR && finished_counter < INDIV_NR_VAR) //nie wiadomo jaki dac warunek, aby nie analizowac "skoczonych" juz obliczen. mozna dodac tablice zliczajaca ilosc zmian. jesli ilosc zmian = ostatnia ilosc zmian, ustaw false. Do tego trzeba w petli odczytywac tablice stanow z GPU i jesli nie bylo zmian, to wyslac nowa
									{
										cout << endl << "Step: " << step_counter << "/" << MAX_STEPS_VAR << endl;
										changed = false;
										for (int i = 0; i < INDIV_NR_VAR; i++)
										{
											if (changes_before[i] == changes_after[i] && continue_steps[i])
											{
												continue_steps[i] = false;
												changed = true;
												finished_counter++;
											}
											else
											{
												changes_before[i] = changes_after[i];
											}
										}
										if (changed)
										{
											// Copying updated continue values from Host to Device
											if (cudaMemcpy(d_continue_steps, continue_steps, sizeof(bool)*INDIV_NR_VAR, cudaMemcpyHostToDevice) != cudaSuccess)
											{
												cout << "Error copying continue_steps to d_continue_steps." << endl;
												cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
												return 1;
											}
										}


										populationInfluenceStep << <INDIV_NR_VAR*N / 192 + 1, 192 >> >(d_influence, d_state, d_changes, d_continue_steps, N);

										//cudaDeviceSynchronize();


										if (cudaMemcpy(changes_after, d_changes, sizeof(float)*INDIV_NR_VAR, cudaMemcpyDeviceToHost) != cudaSuccess)
										{
											cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
											cout << "Error copying d_changes to changes" << endl;
											return 1;
										}



										step_counter++;


										/*cout << endl << "STEP: " << step_counter << endl;
										cout << "FINISHED: " << finished_counter << endl;
										cout << "changes after: " << endl;
										for (int i = 0; i < INDIV_NR_VAR; i++)
										{
										cout << changes_after[i] << endl;
										}*/
									}



									// Copy results from GPU to Host

									if (cudaMemcpy(finalStates, d_state, sizeof(bool)*INDIV_NR_VAR *N, cudaMemcpyDeviceToHost) != cudaSuccess)
									{
										cudaFree(d_influence); cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);
										cout << "Error copying d_state to finalStates" << endl;
										return 1;
									}


									// counting influence capabilities of all individuals

									int influence_capabilities[INDIV_NR_VAR];
									for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv
									{
										influence_capabilities[i] = SEEDS_NR * -1;
										cout << "###### START OF " << i << " ######" << endl;
										for (int j = 0; j < N; j++) // N nodes
										{
											if (finalStates[i][j])
											{
												//cout << "Activated " << j << endl;
												influence_capabilities[i]++;
											}
										}
										cout << "influence capabilities, of " << i << ": " << influence_capabilities[i] << endl;
										cout << "###### END OF " << i << " ######" << endl;
									}


									// selection combined with transferring state array to population array, for merging and mutation
									int transferred;

									for (int i = 0; i < INDIV_NR_VAR; i++) // INDIV_NR_VAR individuals
									{
										int curr_best_id = -1;
										int curr_best_value = -1;

										for (int j = 0; j < GROUP_SIZE_VAR; j++)
										{
											int rand_id = rand() % INDIV_NR_VAR;
											if (influence_capabilities[rand_id] > curr_best_value)
											{
												curr_best_value = influence_capabilities[rand_id];
												curr_best_id = rand_id;
											}
										}

										transferred = 0;
										
										for (int j = 0; j < N; j++)
										{
											if (state[curr_best_id][j])
											{
												population[i][transferred] = j;
												transferred++;
											}
										}
									}


									/* ----- new generation presentation ----- */
									/*cout << endl;
									for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv
									{
									for (int j = 0; j < SEEDS_NR; j++) // SEEDS_NR nodes
									{
									cout << population[i][j] << " ";
									}
									cout << endl;
									}*/



									// krzyzowanie // ew GPU: krzyzowanie thread_id*2 i thread_id*2+1, polowa watkow przy wywolaniu kernela

									int id_first = -1;
									int id_second = -1;
									for (int i = 0; i < INDIV_NR_VAR; i++)
									{
										int mer = rand() % 100;
										if (mer < MERGE_RATIO_VAR * 100)
										{
											if (id_first == -1)
											{
												id_first = i;
											}
											else
											{
												id_second = i;
											}
										}
										if (id_second != -1)
										{
											for (int j = 0; j < SPLIT; j++)
											{
												float temp = population[id_first][j];
												population[id_first][j] = population[id_second][j];
												population[id_second][j] = temp;
											}

											id_first = -1;
											id_second = -1;
										}
									}
									for (int i = 0; i < INDIV_NR_VAR; i++)
									{
										//mutacja
										int mut = rand() % 100;
										//bool alreadyIn = true;
										//int rand_id;
										if (mut < MUTATION_RATIO_VAR * 100)
										{
											for (int j = 0; j < MUTATION_SCALE_VAR * SEEDS_NR; j++)
											{
												population[i][rand() % SEEDS_NR] = rand() % N;
											}
											/* for (int j = 0; j < SEEDS_NR; j++)
											{
											population[i][j] = rand() % N;
											}*/
											/*while (alreadyIn)
											{
											alreadyIn = false;
											rand_id = rand() % N;

											for (int j = 0; j < SEEDS_NR; j++)
											{
											if (population[i][j] == rand_id)
											{
											alreadyIn = true;
											}
											}
											}

											population[i][rand() % SEEDS_NR] = rand_id;*/		// random node of the i-th individual switches to another random node (which is not already in the i-th indiv.)
										}
									}
									
									/* ----- new generation presentation ----- */
									/*cout << endl;
									for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv
									{
									for (int j = 0; j < SEEDS_NR; j++) // SEEDS_NR nodes
									{
									cout << population[i][j] << " ";
									}
									cout << endl;
									}*/

									/* ----- best in generation presentation ----- */
									curr_max = 0;
									curr_max_id = -1;
									for (int i = 0; i < INDIV_NR_VAR; i++) // 10 indiv
									{
										if (influence_capabilities[i] > curr_max)
										{
											curr_max = influence_capabilities[i];
											curr_max_id = i;
										}
									}

									if (curr_max > total_max)
									{
										total_max = curr_max;
									}

									cout << "The Best indiv. for gen " << gen << " can activate " << curr_max << " others." << endl;
									cout << "The Best UNTIL NOW can activate " << total_max << " others." << endl;
									
									
									cout << endl << endl;
									for (int j = 0; j < SEEDS_NR; j++) {
										cout << population[curr_max_id][j] << ", ";
									}
									
									cout << endl << endl;

									// time - after the step
									high_resolution_clock::time_point curr = high_resolution_clock::now();
									curr_time = std::chrono::duration_cast<std::chrono::milliseconds>(curr - beginning).count();
									cout << "Time: " << curr_time / 1000 << "s." << endl;


									gen++;
								}// zakonczenie petli


								cudaFree(d_state); cudaFree(d_changes); cudaFree(d_continue_steps);

								sumOfActivated += total_max;
								
								// time - finish
								high_resolution_clock::time_point ending = high_resolution_clock::now();
								float duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending - beginning).count();
								cout << "Test nr: " << test << endl;// ". Execution time: " << duration / 1000 << "s." << endl;
								durationTotal += duration;

								cout << "Run: " << avg << ". The Best indiv. can activate " << total_max << " others, after " << gen << " generations and " << duration/1000 << " time" << endl;
								cout << "Current avg: " << sumOfActivated / (avg + 1) << " in the avg time: " << durationTotal / (avg + 1) / 1000 << "s" << endl;
								
							} // 5 avgs

							cout << endl << endl << "Average max value: " << sumOfActivated / TESTRUNS << endl;
							cout << "Average time: " << durationTotal / 1000 / TESTRUNS << "s." << endl << endl << endl;
							

							if(MEASURINGTIME)
							{
								RESULTS[ns] = durationTotal / 1000 / TESTRUNS;
							}
							else
							{
								RESULTS[ns] = sumOfActivated / TESTRUNS;
							}

							//test++;
							//RESULTS[in][gs][mer][mur][mus] = sumOfActivated / TESTRUNS;
							//RESULTS[in][gs][mer][mur][mus] = durationTotal / 1000 / TESTRUNS;
						/*}
					}
				}
			}
		}*/
	}
	
	
	
	


	
	// using ofstream constructors.
	std::ofstream outfile("dataset" + to_string(DATASET) + "Times.xls");

	outfile << "<?xml version='1.0'?>" << std::endl;
	outfile << "<Workbook xmlns='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
	outfile << " xmlns:o='urn:schemas-microsoft-com:office:office'" << std::endl;
	outfile << " xmlns:x='urn:schemas-microsoft-com:office:excel'" << std::endl;
	outfile << " xmlns:ss='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
	outfile << " xmlns:html='http://www.w3.org/TR/REC-html40'>" << std::endl;
	outfile << " <Worksheet ss:Name='Sheet1'>" << std::endl;
	outfile << "  <Table>" << std::endl;
	outfile << "   <Row>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>N</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>Goal</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>Gen time</Data></Cell>" << std::endl;
	outfile << "   </Row>" << std::endl;


	for (int ns = 0; ns < N_TESTS; ns++)
	{
		outfile << "   <Row>" << std::endl;
		outfile << "    <Cell><Data ss:Type='Number'>" + to_string(N_VAR_A[ns]) + "</Data></Cell>" << std::endl;
		outfile << "    <Cell><Data ss:Type='Number'>" + to_string(GOALS[ns]) + "</Data></Cell>" << std::endl;
		outfile << "    <Cell><Data ss:Type='Number'>" + to_string(RESULTS[ns]) + "</Data></Cell>" << std::endl;
		outfile << "   </Row>" << std::endl;
	}
	outfile << "  </Table>" << std::endl;
	outfile << " </Worksheet>" << std::endl;
	outfile << "</Workbook>" << std::endl;
	outfile.close();
	
	
	
	
	
	
	
	/*
	// using ofstream constructors.
	std::ofstream outfile("gpu_final_estimation.xls");

	outfile << "<?xml version='1.0'?>" << std::endl;
	outfile << "<Workbook xmlns='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
	outfile << " xmlns:o='urn:schemas-microsoft-com:office:office'" << std::endl;
	outfile << " xmlns:x='urn:schemas-microsoft-com:office:excel'" << std::endl;
	outfile << " xmlns:ss='urn:schemas-microsoft-com:office:spreadsheet'" << std::endl;
	outfile << " xmlns:html='http://www.w3.org/TR/REC-html40'>" << std::endl;
	outfile << " <Worksheet ss:Name='Sheet1'>" << std::endl;
	outfile << "  <Table>" << std::endl;
	outfile << "   <Row>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>INDIV_NR</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>GROUP_SIZE</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>MERGE_RATIO</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>MUTATION_RATIO</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>MUTATION_SCALE</Data></Cell>" << std::endl;
	outfile << "    <Cell><Data ss:Type='String'>VALUE</Data></Cell>" << std::endl;
	outfile << "   </Row>" << std::endl;


	for (int in = 0; in < INDIV_TESTS; in++)
	{
		for (int gs = 0; gs < GROUP_SIZE_TESTS; gs++)
		{
			for (int mer = 0; mer < MERGE_RATIO_TESTS; mer++)
			{
				for (int mur = 0; mur < MUTATION_RATIO_TESTS; mur++)
				{
					for (int mus = 0; mus < MUTATION_SCALE_TESTS; mus++)
					{
						outfile << "   <Row>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(INDIV_NR_VAR_A[in]) + "</Data></Cell>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(GROUP_SIZE_VAR_A[gs]) + "</Data></Cell>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(MERGE_RATIO_VAR_A[mer]) + "</Data></Cell>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(MUTATION_RATIO_VAR_A[mur]) + "</Data></Cell>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(MUTATION_SCALE_VAR_A[mus]) + "</Data></Cell>" << std::endl;
						outfile << "    <Cell><Data ss:Type='Number'>" + to_string(RESULTS[in][gs][mer][mur][mus]) + "</Data></Cell>" << std::endl;
						outfile << "   </Row>" << std::endl;
					}
				}
			}
		}
	}
	outfile << "  </Table>" << std::endl;
	outfile << " </Worksheet>" << std::endl;
	outfile << "</Workbook>" << std::endl;
	outfile.close();*/

	system("pause");
	return 0;

	// 5 * 5 * 5 * 5 * 5 * 5 * 1s * 3proby = 46875 testow
	// ~13h 705
}
